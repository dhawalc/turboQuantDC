#!/usr/bin/env python3
"""Scale test for TurboRetrievalCache: needle-in-haystack at increasing context lengths.

Tests: 2K, 8K, 32K, 100K tokens with Qwen2.5-3B-Instruct (BnB 4-bit).
Measures: needle retrieval success, VRAM usage, prefill speed, generation speed.
"""

import argparse
import gc
import json
import os
import sys
import time
from typing import Any, Optional, Tuple

import torch
import numpy as np

os.environ["HF_HOME"] = "/media/dhawal/Beast/cache"
os.environ["TRANSFORMERS_CACHE"] = "/media/dhawal/Beast/cache/hub"

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from turboquantdc.turbo_retrieval_cache import TurboRetrievalCache


def get_gpu_memory():
    """Return (allocated_gb, reserved_gb, total_gb)."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return allocated, reserved, total
    return 0, 0, 0


def get_ram_usage():
    """Return resident RAM in GB."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / 1e9
    except ImportError:
        return 0


def build_context(target_tokens: int, tokenizer) -> tuple[str, int]:
    """Build synthetic context with a hidden needle at the middle."""
    NEEDLE = "The secret system unlock code is X-ALPHA-774."

    filler = "The system is functioning normally. All diagnostics report green metrics across the board. " * 5
    filler_tokens = len(tokenizer.encode(filler))

    num_paras = target_tokens // filler_tokens + 1
    needle_para = int(num_paras * 0.5)  # middle

    paragraphs = []
    for i in range(num_paras):
        if i == needle_para:
            paragraphs.append(f"\n[SYSTEM ALERT]\n{NEEDLE}\n[ALERT END]\n")
        paragraphs.append(filler)

    text = "\n".join(paragraphs)
    actual_tokens = num_paras * filler_tokens
    return text, actual_tokens


def monkey_patch_qwen2(model):
    """Monkey-patch Qwen2Attention for FAISS retrieval during decoding."""
    import transformers.models.qwen2.modeling_qwen2 as modeling_qwen2

    original_forward = modeling_qwen2.Qwen2Attention.forward

    def retrieve_forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        if "past_key_values" in kwargs and past_key_value is None:
            past_key_value = kwargs.pop("past_key_values")

        is_decoding = hidden_states.shape[1] == 1

        if past_key_value is None or not hasattr(past_key_value, "retrieve_and_attend"):
            return original_forward(
                self, hidden_states, attention_mask, position_ids, past_key_value,
                output_attentions, use_cache, cache_position, position_embeddings,
                **kwargs
            )

        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is not None:
            cos, sin = position_embeddings
            query_states, key_states = modeling_qwen2.apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not is_decoding:
            # PREFILL
            k_out, v_out = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=None)

            if attention_mask is not None and attention_mask.dim() == 4:
                attention_mask = attention_mask[:, :, :, -k_out.shape[2]:]

            from transformers.models.qwen2.modeling_qwen2 import repeat_kv
            import torch.nn.functional as F

            k_out = repeat_kv(k_out, self.num_key_value_groups)
            v_out = repeat_kv(v_out, self.num_key_value_groups)

            attn_output = F.scaled_dot_product_attention(
                query_states, k_out, v_out, attn_mask=attention_mask
            )
        else:
            # DECODING FAISS RETRIEVAL
            if past_key_value is not None:
                _, _ = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=None)

            attn_output_heads = []
            _debug_printed = getattr(self, '_debug_printed', False)
            for h in range(self.config.num_attention_heads):
                kv_h = h // self.num_key_value_groups
                q_h = query_states[:, h, :, :]
                if q_h.dim() == 3:
                    q_h = q_h.squeeze(1)

                res = past_key_value.retrieve_and_attend(self.layer_idx, kv_h, q_h)
                if not _debug_printed and self.layer_idx == 0 and h == 0:
                    seq_len = past_key_value.get_seq_length(self.layer_idx)
                    print(f"  [DEBUG] Decode layer=0 head=0: seq_len={seq_len}, k_effective={res.k_effective}, "
                          f"search_ms={res.search_time_ms:.1f}, attn_ms={res.attn_time_ms:.1f}")
                    self._debug_printed = True
                attn_output_heads.append(res.output.unsqueeze(1))

            attn_output = torch.cat(attn_output_heads, dim=1)
            attn_output = attn_output.unsqueeze(2)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, None

    class FAISSQwen2Attention(type(model.model.layers[0].self_attn)):
        def forward(self, *args, **kwargs):
            return retrieve_forward(self, *args, **kwargs)

    for layer in model.model.layers:
        layer.self_attn.__class__ = FAISSQwen2Attention


def run_test(model, tokenizer, target_tokens: int, results: list,
             prefill_only: bool = False, gen_tokens: int = 50):
    """Run a single needle-in-haystack test at given token count."""
    print(f"\n{'='*70}")
    print(f"  TESTING: {target_tokens:,} tokens")
    print(f"{'='*70}")

    # Clean GPU
    torch.cuda.empty_cache()
    gc.collect()
    alloc_before, _, _ = get_gpu_memory()
    ram_before = get_ram_usage()

    # Create cache
    # Use generous k for retrieval to maximize recall
    retrieval_k = min(8192, target_tokens + 1000)
    key_bits = 3
    val_bits = 3
    window = 2048
    cache = TurboRetrievalCache(
        num_layers=model.config.num_hidden_layers,
        num_kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.hidden_size // model.config.num_attention_heads,
        index_type="flat",
        nlist=16,
        m_subquantizers=16,
        key_bits=key_bits,
        val_bits=val_bits,
        window_size=window,
        k=retrieval_k,
    )
    print(f"  Cache config: k={retrieval_k}, window={window}, key_bits={key_bits}, val_bits={val_bits}")

    # Build context
    print(f"  Building context...")
    t0 = time.time()
    context_text, actual_tokens = build_context(target_tokens, tokenizer)
    build_time = time.time() - t0
    print(f"  Built context in {build_time:.1f}s (~{actual_tokens:,} tokens)")

    prompt = (
        f"You are a retrieval system. Read the log and answer the question.\n\n"
        f"--- LOG START ---\n{context_text}\n--- LOG END ---\n\n"
        f"Question: What is the secret system unlock code?\n"
        f"Answer: The secret system unlock code is"
    )

    # Tokenize
    print(f"  Tokenizing...")
    t0 = time.time()
    inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
    tok_time = time.time() - t0
    total_tokens_actual = inputs.input_ids.shape[1]
    print(f"  Tokenized {total_tokens_actual:,} tokens in {tok_time:.1f}s")

    # Chunked prefill
    chunk_size = 1024
    num_chunks = (total_tokens_actual + chunk_size - 1) // chunk_size

    print(f"  Prefilling {num_chunks} chunks...")
    t0 = time.time()
    last_output = None

    for chunk_idx in range(num_chunks):
        start_tok = chunk_idx * chunk_size
        end_tok = min(start_tok + chunk_size, total_tokens_actual)
        chunk_ids = inputs.input_ids[:, start_tok:end_tok]
        chunk_len = end_tok - start_tok

        past_seen = cache.get_seq_length(0)
        cache_position = torch.arange(past_seen, past_seen + chunk_len, device=model.device)
        position_ids = cache_position.unsqueeze(0)

        actual_attn_len = min(past_seen + chunk_len, 2048)
        attention_mask = torch.ones((1, actual_attn_len), dtype=torch.long, device=model.device)

        with torch.no_grad():
            last_output = model(
                input_ids=chunk_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )

        if (chunk_idx + 1) % max(1, num_chunks // 5) == 0 or (chunk_idx + 1) == num_chunks:
            alloc, _, _ = get_gpu_memory()
            print(f"    Chunk {chunk_idx+1}/{num_chunks} | VRAM: {alloc:.2f} GB")

    prefill_time = time.time() - t0
    prefill_speed = total_tokens_actual / prefill_time

    alloc_after, reserved_after, _ = get_gpu_memory()
    ram_after = get_ram_usage()

    print(f"  Prefill done: {prefill_time:.1f}s ({prefill_speed:.0f} tok/s)")
    print(f"  VRAM: {alloc_before:.2f} -> {alloc_after:.2f} GB (delta: {alloc_after - alloc_before:.2f} GB)")
    print(f"  RAM: {ram_before:.2f} -> {ram_after:.2f} GB (delta: {ram_after - ram_before:.2f} GB)")

    # Free tokenized input to reclaim memory
    del inputs, context_text, prompt
    gc.collect()

    if prefill_only:
        del last_output
        gen_time = 0
        gen_speed = 0
        generated_tokens = []
        answer = "PREFILL_ONLY"
        needle_found = False
        print(f"  Skipping generation (prefill-only mode)")
        result = {
            "target_tokens": target_tokens,
            "actual_tokens": total_tokens_actual,
            "needle_found": needle_found,
            "answer": answer,
            "prefill_time_s": round(prefill_time, 2),
            "prefill_speed_tok_s": round(prefill_speed, 0),
            "gen_time_s": 0,
            "gen_speed_tok_s": 0,
            "vram_before_gb": round(alloc_before, 2),
            "vram_after_gb": round(alloc_after, 2),
            "vram_delta_gb": round(alloc_after - alloc_before, 2),
            "ram_before_gb": round(ram_before, 2),
            "ram_after_gb": round(ram_after, 2),
            "ram_delta_gb": round(ram_after - ram_before, 2),
        }
        results.append(result)
        del cache
        torch.cuda.empty_cache()
        gc.collect()
        return result

    # Generation
    print(f"  Generating response...")

    # Use logits from the last prefill chunk to get the first generated token
    first_token_id = last_output.logits[:, -1, :].argmax(dim=-1).item()
    del last_output

    past_seen = cache.get_seq_length(0)
    cache_position = torch.tensor([past_seen], device=model.device)
    position_ids = cache_position.unsqueeze(0)

    generated_tokens = [first_token_id]
    current_ids = torch.tensor([[first_token_id]], device=model.device)
    t_start = time.time()

    with torch.no_grad():
        for _ in range(gen_tokens - 1):  # already have 1 token
            outputs = model(
                input_ids=current_ids,
                position_ids=position_ids,
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            next_token_id = outputs.logits[:, -1, :].argmax(dim=-1).item()
            generated_tokens.append(next_token_id)

            if next_token_id == tokenizer.eos_token_id:
                break

            current_ids = torch.tensor([[next_token_id]], device=model.device)
            cache_position += 1
            position_ids = cache_position.unsqueeze(0)

    gen_time = time.time() - t_start
    gen_speed = len(generated_tokens) / gen_time if gen_time > 0 else 0

    answer = tokenizer.decode(generated_tokens)
    needle_found = "X-ALPHA-774" in answer

    print(f"  Generated: {answer[:200]}")
    print(f"  Needle found: {'YES' if needle_found else 'NO'}")
    print(f"  Generation: {len(generated_tokens)} tokens in {gen_time:.2f}s ({gen_speed:.1f} tok/s)")

    result = {
        "target_tokens": target_tokens,
        "actual_tokens": total_tokens_actual,
        "needle_found": needle_found,
        "answer": answer[:300],
        "prefill_time_s": round(prefill_time, 2),
        "prefill_speed_tok_s": round(prefill_speed, 0),
        "gen_time_s": round(gen_time, 2),
        "gen_speed_tok_s": round(gen_speed, 1),
        "vram_before_gb": round(alloc_before, 2),
        "vram_after_gb": round(alloc_after, 2),
        "vram_delta_gb": round(alloc_after - alloc_before, 2),
        "ram_before_gb": round(ram_before, 2),
        "ram_after_gb": round(ram_after, 2),
        "ram_delta_gb": round(ram_after - ram_before, 2),
    }
    results.append(result)

    # Cleanup
    del cache
    torch.cuda.empty_cache()
    gc.collect()

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--scales", nargs="+", type=int, default=[2000, 8000, 32000, 100000])
    parser.add_argument("--max-only", type=int, default=None, help="Only run one test at this scale")
    parser.add_argument("--prefill-only", action="store_true", help="Skip generation, only measure prefill")
    parser.add_argument("--gen-tokens", type=int, default=50, help="Max tokens to generate")
    args = parser.parse_args()

    scales = [args.max_only] if args.max_only else args.scales

    print(f"Loading {args.model} in 4-bit quantized...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
        attn_implementation="eager",
    )
    model.eval()

    alloc, reserved, total = get_gpu_memory()
    print(f"Model loaded. VRAM: {alloc:.2f} GB allocated / {total:.1f} GB total")
    print(f"Config: {model.config.num_hidden_layers} layers, {model.config.num_key_value_heads} KV heads, "
          f"head_dim={model.config.hidden_size // model.config.num_attention_heads}")

    # Apply monkey-patch
    print("Monkey-patching Qwen2Attention for FAISS retrieval...")
    monkey_patch_qwen2(model)

    results = []

    for scale in scales:
        try:
            result = run_test(model, tokenizer, scale, results,
                              prefill_only=args.prefill_only,
                              gen_tokens=args.gen_tokens)
        except Exception as e:
            print(f"  FAILED at {scale:,} tokens: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                "target_tokens": scale,
                "error": str(e),
                "needle_found": False,
            })
            # Try to recover
            torch.cuda.empty_cache()
            gc.collect()

    # Print summary
    print(f"\n{'='*70}")
    print(f"  SCALE TEST SUMMARY")
    print(f"{'='*70}")
    print(f"{'Tokens':>10} | {'Needle':>8} | {'Prefill':>10} | {'Gen':>8} | {'VRAM':>8} | {'RAM':>8}")
    print(f"{'-'*10}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    for r in results:
        if "error" in r:
            print(f"{r['target_tokens']:>10,} | {'ERROR':>8} | {'N/A':>10} | {'N/A':>8} | {'N/A':>8} | {'N/A':>8}")
        else:
            print(f"{r['actual_tokens']:>10,} | {'YES' if r['needle_found'] else 'NO':>8} | "
                  f"{r['prefill_speed_tok_s']:>7,.0f}/s | "
                  f"{r['gen_speed_tok_s']:>5.1f}/s | "
                  f"{r['vram_delta_gb']:>5.2f} GB | "
                  f"{r['ram_delta_gb']:>5.2f} GB")

    # Save JSON results
    os.makedirs("/home/dhawal/turboQuantDC/benchmarks/results", exist_ok=True)
    with open("/home/dhawal/turboQuantDC/benchmarks/results/retrieval_cache_scale_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to benchmarks/results/retrieval_cache_scale_results.json")

    return results


if __name__ == "__main__":
    main()
