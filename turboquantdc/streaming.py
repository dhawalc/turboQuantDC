"""Streaming Layer Inference Engine for TurboQuantDC.

Enables running models of ANY size on fixed VRAM by streaming one transformer
layer at a time from CPU to GPU. The key insight: only ONE layer's weights
need to be on GPU at any moment, while the compressed KV cache stays resident.

VRAM budget = sizeof(one_layer) + sizeof(embeddings) + sizeof(lm_head)
            + sizeof(TQ_KV_cache) + sizeof(activations)

For Qwen2.5-3B (36 layers, d=2048): ~150MB/layer -> ~500MB peak
For a hypothetical 200B (80 layers, d=16384): ~3GB/layer -> ~8GB peak

The tradeoff is speed: PCIe bandwidth limits throughput to ~2-5 tok/s
for large models, but correctness is maintained because each layer
forward pass is identical to the non-streaming version.

Usage:
    from turboquantdc.streaming import StreamingInferenceEngine

    engine = StreamingInferenceEngine("Qwen/Qwen2.5-3B-Instruct", bits=3)
    engine.load_model_streaming()
    output = engine.generate("Explain general relativity:", max_new_tokens=50)
    print(output)
    print(engine.memory_report())
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F

from .hf_integration import TurboQuantCache


class StreamingInferenceEngine:
    """Layer-streaming inference with TurboQuant KV cache compression.

    Loads one transformer layer at a time from CPU to GPU, computes the
    forward pass, compresses the KV cache, and evicts the layer weights.

    This enables running models of ANY size on a fixed VRAM budget.
    The tradeoff is speed: PCIe bandwidth limits throughput.

    Args:
        model_name: HuggingFace model name or path.
        bits: TurboQuant bit-width for KV cache (2, 3, 4).
        device: GPU device string (e.g. "cuda", "cuda:0").
        dtype: Weight dtype (torch.float16 or torch.bfloat16).
    """

    def __init__(
        self,
        model_name: str,
        bits: int = 3,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16,
    ):
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for streaming inference")

        self.model_name = model_name
        self.bits = bits
        self.device = torch.device(device)
        self.dtype = dtype

        # Populated by load_model_streaming()
        self.config = None
        self.tokenizer = None
        self.embed_tokens = None
        self.rotary_emb = None  # Rotary position embeddings (tiny, on GPU)
        self.final_norm = None
        self.lm_head = None
        self.layers: List[Any] = []  # CPU-resident transformer layers
        self.num_layers: int = 0
        self.tq_cache: Optional[TurboQuantCache] = None

        # Architecture info (populated during load)
        self.hidden_size: int = 0
        self.num_attention_heads: int = 0
        self.num_kv_heads: int = 0
        self.head_dim: int = 0
        self.vocab_size: int = 0

        # Memory tracking
        self._peak_vram: int = 0
        self._layer_size_bytes: int = 0
        self._embed_size_bytes: int = 0
        self._lm_head_size_bytes: int = 0
        self._model_total_bytes: int = 0

        # Timing
        self._load_time: float = 0.0
        self._tokens_generated: int = 0
        self._generation_time: float = 0.0

    def load_model_streaming(self) -> None:
        """Load model with layers on CPU, embeddings and lm_head on GPU.

        The model is loaded entirely to CPU first, then we selectively move
        the small embedding, rotary_emb, norm, and lm_head modules to GPU
        while keeping the bulk of the weights (transformer layers) on CPU.
        """
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        start = time.time()

        # Load config for architecture info
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.hidden_size = self.config.hidden_size
        self.num_attention_heads = self.config.num_attention_heads
        self.num_kv_heads = getattr(
            self.config, "num_key_value_heads", self.num_attention_heads
        )
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.vocab_size = self.config.vocab_size
        self.num_layers = self.config.num_hidden_layers

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load the full model to CPU with the correct dtype flag
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=self.dtype,
            device_map="cpu",
            low_cpu_mem_usage=True,
        )
        model.eval()

        # Compute total model size for reporting
        self._model_total_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        )

        # Extract the model backbone
        backbone = self._get_backbone(model)

        # Move embeddings to GPU (vocab * hidden_size * 2 bytes)
        self.embed_tokens = backbone.embed_tokens.to(self.device)
        self._embed_size_bytes = sum(
            p.numel() * p.element_size() for p in self.embed_tokens.parameters()
        )

        # Move rotary position embeddings to GPU (tiny: just frequency tensors)
        if hasattr(backbone, "rotary_emb"):
            self.rotary_emb = backbone.rotary_emb.to(self.device)

        # Move final layer norm to GPU (tiny: 2 * hidden_size bytes)
        self.final_norm = backbone.norm.to(self.device)

        # Move lm_head to GPU (vocab * hidden_size * 2 bytes, or tied with embed)
        self.lm_head = model.lm_head.to(self.device)
        self._lm_head_size_bytes = sum(
            p.numel() * p.element_size() for p in self.lm_head.parameters()
        )

        # Keep layers on CPU -- store references
        self.layers = list(backbone.layers)
        for layer in self.layers:
            layer.to("cpu")
            layer.eval()

        # Measure one layer's size
        self._layer_size_bytes = sum(
            p.numel() * p.element_size()
            for p in self.layers[0].parameters()
        )

        # Initialize TurboQuant KV cache
        self.tq_cache = TurboQuantCache(bits=self.bits, seed=42)

        # Clean up the shell model to free CPU memory
        del model
        del backbone
        gc.collect()

        self._load_time = time.time() - start

        # Record baseline VRAM after loading embeddings
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self._peak_vram = torch.cuda.max_memory_allocated(self.device)

    def _get_backbone(self, model: Any) -> Any:
        """Extract the transformer backbone from an HF model.

        Different model families use different attribute names for the
        main transformer block. We try the common ones.
        """
        # Qwen2, Llama, Mistral, Gemma, etc.
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model
        # GPT-2, GPT-Neo
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer
        # Fallback: try common names
        for attr in ("model", "transformer", "backbone"):
            if hasattr(model, attr):
                sub = getattr(model, attr)
                if hasattr(sub, "layers") or hasattr(sub, "h"):
                    return sub
        raise ValueError(
            f"Cannot find transformer backbone in {type(model).__name__}. "
            f"Supported: Qwen2, Llama, Mistral, Gemma, GPT-2."
        )

    @torch.inference_mode()
    def forward_one_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        cache_position: torch.Tensor,
    ) -> torch.Tensor:
        """Move one layer to GPU, compute forward, compress KV, evict.

        This is the core streaming operation. For each layer:
        1. Transfer weights CPU -> GPU
        2. Run the layer's forward pass (attention + FFN)
        3. KV cache is managed by HF's attention via our TurboQuantCache
        4. Transfer weights GPU -> CPU and free GPU memory

        Args:
            layer_idx: Index of the transformer layer.
            hidden_states: Current hidden states on GPU, (batch, seq, hidden).
            position_ids: Position IDs on GPU, (batch, seq).
            attention_mask: Causal attention mask on GPU.
            position_embeddings: (cos, sin) tuple from rotary_emb on GPU.
            cache_position: Cache position indices on GPU, (seq,).

        Returns:
            Updated hidden_states tensor on GPU.
        """
        # 1. Move layer weights CPU -> GPU
        layer = self.layers[layer_idx]
        layer.to(self.device)

        # 2. Forward pass -- HF layer calls attention which calls
        #    past_key_values.update() -> TurboQuantCache.update()
        hidden_states = layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=self.tq_cache,
            use_cache=True,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

        # Some layers return a tuple, some a tensor
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        # 3. Move layer weights GPU -> CPU and free GPU memory
        layer.to("cpu")

        # Track peak VRAM
        current_peak = torch.cuda.max_memory_allocated(self.device)
        if current_peak > self._peak_vram:
            self._peak_vram = current_peak

        return hidden_states

    @torch.inference_mode()
    def generate_token(
        self,
        input_ids: torch.Tensor,
        past_seq_len: int = 0,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> torch.Tensor:
        """Generate one token by streaming through all layers.

        Handles the full forward pass: embed -> rotary -> layer loop -> norm -> head.

        Args:
            input_ids: Input token IDs, shape (batch, seq).
            past_seq_len: Number of previously cached tokens.
            temperature: Sampling temperature (1.0 = no change).
            top_k: Top-k filtering (0 = greedy).

        Returns:
            Next token ID, shape (batch, 1).
        """
        input_ids = input_ids.to(self.device)
        batch_size, seq_len = input_ids.shape

        # Compute cache_position and position_ids
        cache_position = torch.arange(
            past_seq_len, past_seq_len + seq_len, device=self.device
        )
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)

        # 1. Embed input tokens
        hidden_states = self.embed_tokens(input_ids)

        # 2. Compute rotary position embeddings (cos, sin)
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 3. Build causal attention mask
        attention_mask = self._build_causal_mask(
            batch_size, seq_len, past_seq_len, hidden_states.dtype
        )

        # 4. Stream through all transformer layers
        for layer_idx in range(self.num_layers):
            hidden_states = self.forward_one_layer(
                layer_idx,
                hidden_states,
                position_ids,
                attention_mask,
                position_embeddings,
                cache_position,
            )

        # 5. Final layer norm
        hidden_states = self.final_norm(hidden_states)

        # 6. Project to vocabulary -- only need the last position
        logits = self.lm_head(hidden_states[:, -1:, :])  # (batch, 1, vocab)

        # 7. Sample next token
        if temperature <= 0 or top_k == 0:
            # Greedy
            next_token = logits.argmax(dim=-1)  # (batch, 1)
        else:
            logits = logits / temperature
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k, dim=-1)
                probs = F.softmax(top_k_logits, dim=-1)
                sampled_idx = torch.multinomial(probs.squeeze(1), 1)
                next_token = top_k_indices.squeeze(1).gather(-1, sampled_idx)

        return next_token

    def _build_causal_mask(
        self,
        batch_size: int,
        seq_len: int,
        past_seq_len: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build a 4D causal attention mask.

        Creates the mask in the format HF attention expects:
        shape (batch, 1, seq_len, total_len) with 0.0 for attend
        and -inf for masked positions.

        Args:
            batch_size: Batch dimension.
            seq_len: Current sequence length (query length).
            past_seq_len: Number of tokens already in the KV cache.
            dtype: Tensor dtype for the mask.

        Returns:
            4D causal mask tensor.
        """
        total_len = past_seq_len + seq_len

        # Start with all attend (zeros)
        mask = torch.zeros(
            batch_size, 1, seq_len, total_len,
            device=self.device, dtype=dtype,
        )

        if seq_len > 1:
            # During prefill: apply causal mask to the new positions
            # Each position i in the query can attend to past positions [0..past+i]
            # but not to future positions [past+i+1..total)
            causal_mask = torch.triu(
                torch.full(
                    (seq_len, seq_len), torch.finfo(dtype).min,
                    device=self.device, dtype=dtype,
                ),
                diagonal=1,
            )
            mask[:, :, :, past_seq_len:] = causal_mask.unsqueeze(0).unsqueeze(0)

        return mask

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> str:
        """Full text generation with streaming inference.

        Args:
            prompt: Input text prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature (1.0 = neutral, <1 = sharper).
            top_k: Top-k filtering (0 = greedy decoding).

        Returns:
            Generated text (prompt + completion).
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Model not loaded. Call load_model_streaming() first."
            )

        # Reset KV cache and VRAM tracking for this generation
        self.tq_cache = TurboQuantCache(bits=self.bits, seed=42)
        torch.cuda.reset_peak_memory_stats(self.device)

        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(self.device)
        all_token_ids = input_ids.clone()

        gen_start = time.time()

        # Prefill: process the entire prompt at once
        next_token = self.generate_token(
            input_ids, past_seq_len=0,
            temperature=temperature, top_k=top_k,
        )
        all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)
        past_seq_len = input_ids.shape[1]

        # Decode: generate one token at a time
        for step in range(max_new_tokens - 1):
            # Check for EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break

            next_token = self.generate_token(
                next_token, past_seq_len=past_seq_len,
                temperature=temperature, top_k=top_k,
            )
            all_token_ids = torch.cat([all_token_ids, next_token], dim=-1)
            past_seq_len += 1

        self._generation_time = time.time() - gen_start
        self._tokens_generated = all_token_ids.shape[1] - input_ids.shape[1]
        self._peak_vram = torch.cuda.max_memory_allocated(self.device)

        # Decode all tokens to text
        output_text = self.tokenizer.decode(
            all_token_ids[0], skip_special_tokens=True
        )
        return output_text

    def memory_report(self) -> Dict[str, Any]:
        """Report VRAM usage breakdown.

        Returns:
            Dict with memory statistics in MB including embeddings, one layer
            size, KV cache, peak VRAM, model total, compression factor, and
            generation throughput.
        """
        MB = 1024 * 1024

        embed_mb = self._embed_size_bytes / MB
        lm_head_mb = self._lm_head_size_bytes / MB
        one_layer_mb = self._layer_size_bytes / MB
        model_total_mb = self._model_total_bytes / MB
        peak_vram_mb = self._peak_vram / MB

        # KV cache memory from TurboQuant
        kv_cache_mb = 0.0
        if self.tq_cache is not None and self.tq_cache.is_initialized:
            savings = self.tq_cache.memory_savings()
            kv_cache_mb = savings["total_compressed_bits"] / 8 / MB

        compression_factor = (
            model_total_mb / peak_vram_mb if peak_vram_mb > 0 else 0.0
        )

        tok_per_sec = 0.0
        if self._generation_time > 0 and self._tokens_generated > 0:
            tok_per_sec = self._tokens_generated / self._generation_time

        return {
            "embeddings_mb": round(embed_mb, 1),
            "lm_head_mb": round(lm_head_mb, 1),
            "one_layer_mb": round(one_layer_mb, 1),
            "kv_cache_mb": round(kv_cache_mb, 3),
            "peak_vram_mb": round(peak_vram_mb, 1),
            "model_total_mb": round(model_total_mb, 1),
            "compression_factor": round(compression_factor, 2),
            "num_layers": self.num_layers,
            "tokens_generated": self._tokens_generated,
            "tokens_per_sec": round(tok_per_sec, 2),
            "load_time_sec": round(self._load_time, 1),
            "bits": self.bits,
        }

    def architecture_info(self) -> Dict[str, Any]:
        """Return model architecture details."""
        return {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_attention_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "dtype": str(self.dtype),
        }
