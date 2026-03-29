"""TurboQuant KV Cache Compression -- HuggingFace Integration Example.

This example shows how to use TurboQuantDC's compressed KV cache
with any HuggingFace model via the standard generate() API.

IMPORTANT: The drop-in cache uses MSE-only key reconstruction. The
paper's full unbiased estimator requires a custom attention kernel.
For generation tasks, use bits=4 for best quality. At bits=3 and
below, generation quality degrades due to error compounding across
autoregressive steps. The compression ratios and attention-level
quality metrics still match the paper, but end-to-end generation
output quality is limited by the MSE-only reconstruction.

Requirements:
    pip install turboquantdc transformers torch accelerate

Usage:
    python examples/hf_turboquant_example.py

Reference: TurboQuant paper (arxiv 2504.19874), ICLR 2026.
"""

from __future__ import annotations

import gc
import os
import sys
import time

import torch

# Allow running from repo root
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)


# ============================================================================
# Example 1: Basic usage with generate()
# ============================================================================

def example_basic_usage():
    """Show the simplest possible TurboQuant integration with HF generate()."""
    print("=" * 70)
    print("Example 1: Basic usage with generate()")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquantdc import TurboQuantCache

    model_name = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    prompt = "Explain the key insight behind vector quantization in three sentences."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    # Create TurboQuant cache -- this is the only change needed
    cache = TurboQuantCache(bits=3)

    print(f"Prompt: {prompt}")
    print(f"Generating with TurboQuant 3-bit KV cache...")
    start = time.perf_counter()

    output = model.generate(
        **inputs,
        max_new_tokens=150,
        past_key_values=cache,
        do_sample=False,
    )

    elapsed = time.perf_counter() - start
    new_tokens = output.shape[1] - inputs["input_ids"].shape[1]

    response = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

    print(f"\nResponse ({new_tokens} tokens in {elapsed:.2f}s, {new_tokens/elapsed:.1f} tok/s):")
    print(response)

    # Show memory savings
    savings = cache.memory_savings()
    print(f"\nKV Cache Memory:")
    print(f"  Compressed:  {savings['total_compressed_bits'] / 8 / 1024:.1f} KB")
    print(f"  FP16 equiv:  {savings['total_fp16_bits'] / 8 / 1024:.1f} KB")
    print(f"  Compression: {savings['overall_compression_ratio']:.2f}x")
    print()

    del model, tokenizer, cache
    gc.collect()
    torch.cuda.empty_cache()

    return response


# ============================================================================
# Example 2: FP16 vs TQ-3 vs TQ-2 quality comparison
# ============================================================================

def example_quality_comparison():
    """Compare output quality across bit-widths."""
    print("=" * 70)
    print("Example 2: FP16 vs TQ-4 vs TQ-3 vs TQ-2 Output Quality")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquantdc import TurboQuantCache

    model_name = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    prompt = "What are the three laws of thermodynamics? Be concise."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    print(f"Prompt: {prompt}\n")

    configs = [
        ("FP16 (baseline)", None),
        ("TQ-4 (4-bit)", 4),
        ("TQ-3 (3-bit)", 3),
        ("TQ-2 (2-bit)", 2),
    ]

    results = []
    for name, bits in configs:
        cache = TurboQuantCache(bits=bits) if bits else None

        output = model.generate(
            **inputs,
            max_new_tokens=200,
            past_key_values=cache,
            do_sample=False,
        )

        response = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        results.append((name, response, cache))

        ratio_str = ""
        if cache is not None:
            savings = cache.memory_savings()
            ratio_str = f"  [{savings['overall_compression_ratio']:.1f}x compression]"

        print(f"--- {name}{ratio_str} ---")
        print(response[:300])
        print()

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    return results


# ============================================================================
# Example 3: Memory savings measurement
# ============================================================================

def example_memory_savings():
    """Measure actual VRAM savings from TurboQuant cache compression."""
    print("=" * 70)
    print("Example 3: Memory Savings Measurement")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquantdc import TurboQuantCache

    model_name = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Build a longer prompt to make KV cache memory significant
    filler = (
        "The quarterly financial review meeting covered several topics including "
        "budget allocations for the upcoming fiscal year and departmental spending. "
    ) * 20
    prompt = filler + "Summarize the above in one sentence."
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    print(f"Input length: {input_len} tokens")
    print()

    # Measure for each bit-width
    print(f"{'Config':<12} {'KV Cache':<12} {'FP16 Equiv':<12} {'Ratio':<8} {'Saved':<10}")
    print("-" * 56)

    for bits in [4, 3, 2]:
        cache = TurboQuantCache(bits=bits)

        model.generate(
            **inputs,
            max_new_tokens=50,
            past_key_values=cache,
            do_sample=False,
        )

        savings = cache.memory_savings()
        compressed_mb = savings["total_compressed_bits"] / 8 / 1024 / 1024
        fp16_mb = savings["total_fp16_bits"] / 8 / 1024 / 1024
        saved_mb = fp16_mb - compressed_mb
        ratio = savings["overall_compression_ratio"]

        print(f"TQ-{bits}        {compressed_mb:>8.2f} MB   {fp16_mb:>8.2f} MB   {ratio:.1f}x     {saved_mb:.2f} MB")

    print()
    print("Note: Memory savings scale linearly with sequence length.")
    print("At 32K tokens with 28 layers, 3-bit saves ~1.5 GB of VRAM.")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Example 4: Using with different models
# ============================================================================

def example_different_models():
    """Show TurboQuant works with different HF models."""
    print("=" * 70)
    print("Example 4: Different Models")
    print("=" * 70)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquantdc import TurboQuantCache

    # TurboQuant works with any model that uses standard HF attention.
    # The cache auto-detects head_dim and num_heads on first use.
    models_to_test = [
        ("Qwen/Qwen2.5-3B-Instruct", "What is 2+2? Answer with just the number."),
    ]

    # Check if additional models are available
    try:
        from transformers import AutoConfig
        for extra_model, extra_prompt in [
            ("Qwen/Qwen2.5-0.5B-Instruct", "Name three primary colors."),
        ]:
            try:
                _ = AutoConfig.from_pretrained(extra_model)
                models_to_test.append((extra_model, extra_prompt))
            except Exception:
                pass
    except Exception:
        pass

    for model_name, prompt in models_to_test:
        print(f"\n--- {model_name} ---")
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
            )

            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            cache = TurboQuantCache(bits=3)
            output = model.generate(
                **inputs,
                max_new_tokens=50,
                past_key_values=cache,
                do_sample=False,
            )

            response = tokenizer.decode(
                output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True,
            )
            savings = cache.memory_savings()

            print(f"  Prompt: {prompt}")
            print(f"  Response: {response[:200]}")
            print(f"  Layers: {savings['num_layers']}, Compression: {savings['overall_compression_ratio']:.1f}x")

            del model, tokenizer, cache
            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"  Skipped: {e}")

    print()


# ============================================================================
# Example 5: Streaming + TurboQuant for models that don't fit
# ============================================================================

def example_streaming_inference():
    """Show layer-streaming inference with TurboQuant cache compression.

    This demonstrates the StreamingInferenceEngine, which loads one
    transformer layer at a time onto the GPU. Combined with TurboQuant
    cache compression, it enables running models on limited VRAM.
    """
    print("=" * 70)
    print("Example 5: Streaming Inference + TurboQuant")
    print("=" * 70)

    from turboquantdc.streaming import StreamingInferenceEngine

    model_name = "Qwen/Qwen2.5-3B-Instruct"

    print(f"Loading {model_name} in streaming mode...")
    print("(One layer at a time on GPU, TurboQuant 3-bit KV cache)")
    print()

    try:
        engine = StreamingInferenceEngine(model_name, bits=3)
        engine.load_model_streaming()

        prompt = "What is the capital of France?"
        print(f"Prompt: {prompt}")

        output = engine.generate(prompt, max_new_tokens=50)
        print(f"Response: {output}")

        report = engine.memory_report()
        print(f"\nMemory report:")
        for key, val in report.items():
            if isinstance(val, float):
                print(f"  {key}: {val:.2f}")
            else:
                print(f"  {key}: {val}")

        del engine
        gc.collect()
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Streaming inference not available: {e}")
        print("(Requires enough RAM to hold model weights on CPU)")

    print()


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("TurboQuantDC -- HuggingFace Integration Examples")
    print("=" * 70)
    print()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Examples require a GPU.")
        print("Some examples may fail or run very slowly on CPU.")
        print()

    # Run each example, catching errors so all get attempted
    examples = [
        ("Basic Usage", example_basic_usage),
        ("Quality Comparison", example_quality_comparison),
        ("Memory Savings", example_memory_savings),
        ("Different Models", example_different_models),
        ("Streaming Inference", example_streaming_inference),
    ]

    for name, fn in examples:
        try:
            fn()
        except Exception as e:
            print(f"\n[Example '{name}' failed: {e}]")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 70)
    print("All examples complete.")
