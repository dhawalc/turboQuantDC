# Cache Architecture Review (Opus 4.7)

Date: 2026-04-27
Reviewer: Opus 4.7 (1M context)
Scope: 20 cache implementations under `turboquantdc/`
Demo target: Qwen3.6-27B on RTX 4090 via vLLM with TurboQuant 3-bit KV
Method: Read-only review, ~6 hour ship horizon

---

## Production-Ready Caches (load-bearing for vLLM integration)

Only ONE cache is the right horse for the demo. Everything else is either
research, infrastructure for layer streaming, or broken.

- **`turboquantdc/generation_cache.py`** (re-exports `generation_core.GenerationCache`)
  — Production HF-compatible KV cache with per-layer FP16 anchor schedule
  (`fixed` / `boundary` / `gradient`), FP16 hot window, ResidualQuant keys,
  MSE values, optional Triton fused kernels. HANDOFF.md explicitly marks this
  as STABLE. `__init__.py` exposes presets (`balanced`, `aggressive`,
  `lossless`, `hybrid_max_quality`). It is the only cache that:
  - Has been hammered by 246 sweep configs (per its own docstring)
  - Cleanly handles GQA (it is shape-agnostic on `num_heads`; whatever the
    model passes for dim 1 is what gets quantized — head_dim is the only
    fixed quantity)
  - Has matching transformers `get_mask_sizes` semantics for 5.3 and 5.5
  - Is what `run_70b.py` actually calls in production

- **`turboquantdc/token_eviction.py`** (`EvictionCache`)
  — Subclass-style extension of `_CompressedLayer` with attention-guided
  eviction. Used by `run_70b.py` and `KVManager` as the aggressive-budget
  fallback. Quality is gated by recency+structural scoring, but for the
  3-bit, 27B, single-stream demo, it is only needed if KV budget falls below
  ~4 GB. For Qwen3.6-27B at 32K context with 3-bit, `GenerationCache`
  alone fits — eviction is not on the critical path. Keep, but mark
  **SECONDARY**.

**Recommendation for the demo: ship `GenerationCache.from_preset("balanced")`
or `from_preset("hybrid_max_quality")` with `num_layers=` plumbed through.**

For vLLM specifically, `vllm_integration.py` already exists with
`TurboQuantAttentionBackend` and `TurboQuantCacheManager` — but note the
limitations under CRITICAL findings below. The cache manager's `block_size`
argument is currently a no-op (line 504: "Not used to reshape buffers here").
Real PagedAttention support is unimplemented.

---

## Broken / Don't Use

- **`turboquantdc/v2_cache.py`** (`TurboQuantV2Cache`) — Confirmed broken per
  HANDOFF "PCA whitening amplifies noise in low-variance dimensions". The
  whitening at lines 213-217 (`safe_eigs.clamp(min=1e-12)` divided into
  `target_var`) is exactly the failure mode: a tiny eigenvalue divides into
  `1/d` and produces a huge `_whiten_scale`, blowing up quantization error
  on low-variance PCA components. Also has a per-batch / per-head Python
  for-loop in `_flush_buffer` (lines 458-492) — guaranteed slow on GPU.
  ARCHIVE.

- **`turboquantdc/retrieval_cache.py`** (`RetrievalKVCache`) — Confirmed
  broken per HANDOFF "FAISS undertrained, sliding-window loses distant
  tokens". The `_create_index` at line 121 uses `effective_nlist = min(self.nlist, max(1, n_train // 39))`
  which silently degrades to a single cluster for n < 40. Beyond ~2K tokens
  the IVF index is undertrained (FAISS recommends 30-256x training points
  per centroid; 39 is below the floor). Per-layer per-head FAISS index
  storage (`indexes[layer][head]`) does not share state across requests and
  has no batched update path. ARCHIVE.

- **`turboquantdc/cross_layer_predict.py`** — This is not a cache. It is a
  **research script** (`if __name__ == "__main__"` style — see line 13:
  `"python -m turboquantdc.cross_layer_predict"`). Hardcoded to
  `Qwen/Qwen2.5-3B-Instruct` and `/media/dhawal/Beast/cache/hub/` (lines
  31-33). Does not implement any cache. ARCHIVE.

- **`turboquantdc/cache_distillation.py`** — Standalone analysis tool, not a
  KV cache. The `DistillAndCompressCache` wrapper requires you to have all
  KV in memory before distillation (forward-pass distillation), which
  defeats the purpose of streaming. HANDOFF marks it "STANDALONE". ARCHIVE.

- **`turboquantdc/temporal_delta.py`** — Author's own docstring (lines 3-4)
  says "EXPERIMENTAL STATUS: Marginal." Lines 16-29 catalog the trilemma
  that makes it strictly worse than `GenerationCache`. ARCHIVE.

- **`turboquantdc/sparse_loading.py`** — Predicts active FFN neurons for
  weight streaming. NOT a KV cache. Belongs to the streaming-engine concern,
  not the production cache stack. ARCHIVE under `research_artifacts/streaming/`.

---

## Research Artifacts (move to `docs/research_artifacts/`)

These are intellectually interesting but NOT load-bearing for the vLLM demo.
They share the failure mode of "wrap an inner cache and add a heuristic" but
each has either a HANDOFF caveat, a known-bad axis, or a degraded code path.

- **`adaptive_hf_cache.py`** — FP16-anchor-only variant of the same idea
  GenerationCache already supports natively via `anchor_strategy="fixed"`.
  Redundant. Keeps separate `FP16Layer` and `TurboQuantLayer` classes that
  duplicate logic. Move to `research_artifacts/early_anchor_experiments/`.

- **`adaptive_generation_cache.py`** — Tier-0/1/2/3 importance-driven
  refinement system. Spec is interesting but every flush invokes a Python
  loop over tiers (`_compress_mixed_tiers`, lines 329-349), and the
  `tier_distribution` / `effective_bits` paths assume a static schedule
  that doesn't survive batch_size > 1 cleanly (only repeat-interleaves
  the FP16 anchor layers — see lines 812-823 — silently no-ops on
  `_AdaptiveLayer`). ARCHIVE.

- **`ultimate_cache.py`** — Pre-`GenerationCache` ancestor. Same anchor
  mechanism, same FP16-window mechanism, but uses `PolarQuant` internally
  (Stage 1 only, no fused triton). GenerationCache subsumes this 1:1 with
  better defaults. ARCHIVE.

- **`self_correcting_cache.py`** — Wraps an inner cache and re-quantizes
  high-norm tokens periodically (norm-correction "I-frames"). The norm
  drift it tries to correct is already mitigated by `use_norm_correction=True`
  on the production `_CompressedLayer`. The wrapper adds Python-level
  bookkeeping per-step (`tokens_since_refresh`) which doesn't pay for
  itself in measurable PPL improvement. ARCHIVE.

- **`streaming.py`** (`StreamingInferenceEngine`) — Layer-streaming engine,
  CPU↔GPU per layer. Drops to ~2-5 tok/s. For a 27B model on a 24GB 4090,
  the model fits in VRAM at 4-bit (BnB) with KV compression — streaming
  is unnecessary and would fight vLLM's own scheduler. Move to
  `research_artifacts/streaming/`.

- **`streaming_70b.py`** (`StreamingModel`, `LayerGPUCache`,
  `AsyncPrefetcher`, `MemoryPlanner`) — The 70B-tier of the streaming
  engine. Shares the same vLLM-incompatibility (vLLM owns layer placement;
  this engine owns layer placement; pick one). Used by `run_70b.py` for
  the standalone demo, NOT by vLLM. Keep for the standalone demo path,
  archive everything but `MemoryPlanner` (which is just sizing math).

- **`ultra_streaming.py` / `ultra_streaming_kv.py` / `ultra_streaming_weights.py`**
  — `KVManager` is a strategy-selector that picks between `GenerationCache`
  and `EvictionCache` (lines 137-158 of `ultra_streaming_kv.py`). This is
  policy code, not a cache. `WeightManager` in `ultra_streaming_weights.py`
  is a generic LRU cache for weights. Both belong with the streaming
  research, not the production cache surface. ARCHIVE.

- **`cross_layer_kv.py`** (`CrossLayerKVCache`, `_SharedResourceLayer`) —
  Shares one rotation matrix across `group_size` layers. The author's own
  `correlation_report` (line 251) prints "Delta coding NOT viable" on real
  data. The shared-resource path saves O(d²) bytes per layer (a few KB on
  a 27B model) — irrelevant given KV cache is gigabytes. Pure research
  artifact. ARCHIVE.

- **`chunked_prefill.py`** (`ChunkedPrefillEngine`) — Functions correctly
  for HF `generate()`-style use. For vLLM, prefill chunking is handled by
  vLLM's scheduler, not at the cache layer. Keep as a standalone utility,
  archive from cache concerns. Move to `tools/` or `research_artifacts/`.

- **`temporal_decay.py`** (`TemporalDecayCache`) — Three-tier hot/warm/cold
  TurboQuantKVCache. Quality vs. simpler `GenerationCache` is unproven on
  Qwen models per HANDOFF (whose published numbers all use GenerationCache
  variants). ARCHIVE.

---

## CRITICAL findings in load-bearing files

### 1. vLLM PagedAttention block layout NOT implemented
**File:** `turboquantdc/vllm_integration.py:487-543`
**Issue:** `TurboQuantCacheManager.allocate(block_size=16)` accepts the vLLM
block size but the comment at line 500-503 admits: *"Not used to reshape
buffers here — kept for API compatibility and future paged cache support"*.
Buffers are allocated as flat `(max_seq_len, num_kv_heads, head_dim)`
tensors — the contiguous-per-block layout vLLM PagedAttention requires is
absent. `store(slot_idx, ...)` writes into a flat slice, not into a
2-D `[block_id, slot_in_block]` lookup.

**Impact for the 6-hour demo:** vLLM's BlockManager and AttentionMetadata
expect `kv_cache` shaped as `(num_blocks, block_size, num_kv_heads, head_dim)`
(or the FP8/INT8 PagedAttention variant). Wiring `TurboQuantCacheManager`
into vLLM as-is will fail at the first attention forward pass because the
slot indexing is incompatible.

**Fix:** Either (a) re-layout the buffers as `(num_blocks, block_size, H, D)`
and add a slot→(block, offset) translator in `store`/`fetch`, OR (b) skip
vLLM's PagedAttention path entirely and integrate at the `AttentionImpl`
level after vLLM has already gathered the KV per-request. The monkey-patch
path in the docstring (lines 27-54) is the latter, but it bypasses the
`CacheManager` entirely.

### 2. Buffers are per-instance, not request-shareable
**File:** `turboquantdc/vllm_integration.py:194-199, 460-471`
**Issue:** `_compressed_key_store: List[List[CompressedKey]]` and the per-layer
`_key_quantizers / _value_quantizers` are instance-bound. There is no
mechanism for sharing prefix KV across requests (vLLM's prefix caching) and
no concept of a request_id or sequence_id in the API surface. Two concurrent
requests would either share state (corruption) or each need its own
backend instance (defeats compression).

**Fix:** vLLM integration must key its storage by `(layer_idx, block_id)`,
not `(layer_idx, slot_idx)`. Block ids are owned by vLLM's BlockManager
and shareable across sequences.

### 3. Quantizer state allocated CPU-side via scipy in hot path
**File:** Tracing `GenerationCache → _CompressedLayer → LloydMaxCodebook`
**Issue:** First `update()` call lazily creates `LloydMaxCodebook(d, bits)`
via `solve_lloyd_max` which uses `scipy.integrate.quad`. This is a
~10-50ms one-time cost per (d, bits) pair. Fine for HF generate() startup,
but in vLLM the first request on a fresh cache will block the worker thread.

**Fix:** Pre-warm the codebook cache at engine init. Easy: instantiate one
`GenerationCache` with the model's known `num_layers`, call a 1-token
`update()` in a sentinel pass, discard.

---

## HIGH

### H1. `GenerationCache` returns full dequantized cache on every `update()`
**File:** `turboquantdc/generation_core.py:275-295`
**Observation:** The HF Cache protocol requires `update()` to return
`(all_keys, all_values)`. `GenerationCache` complies by dequantizing the
entire compressed prefix and concatenating it with the FP16 window every
forward step. At 32K context this is a 32K × 8 (kv-heads) × 128 (d) × 2
(K+V) × 2 bytes = 128 MB tensor materialized per layer per token, and
N_layers × tokens times during prefill. This is the root reason GenerationCache
won't deliver native vLLM throughput — it materializes the dense KV
which vLLM is trying to avoid via PagedAttention.

**Fix path:** vLLM bypasses HF Cache. The `vllm_integration.TurboQuantAttentionBackend`
already has the right shape (compress on store, estimate IP on query). The
fix is to NOT route vLLM through `GenerationCache.update()` — use
`compute_attention()` directly. This is a vLLM hook, not a `GenerationCache`
fix.

### H2. GQA double-quantization risk in vLLM path
**File:** `turboquantdc/vllm_integration.py:586-587`
**Observation:** `k_flat = keys.reshape(N * H, D)` flattens KV-heads into
the batch dim before quantizing. For Qwen3.6-27B with `num_attention_heads=64`,
`num_kv_heads=8`, the cache stores 8 K and 8 V vectors per token. The flatten
is correct ONLY if the caller passes raw KV (not the GQA-broadcasted version).
If a caller mistakenly passes the broadcast-to-attention-heads tensor, the
cache will quantize the same K vector 8 times with the same seed — wasted
compute, no quality loss, but huge memory blow-up.

**Mitigation:** Add a shape assertion `assert keys.shape[-2] == self.num_kv_heads`
in `compress_kv()` and `store()`.

### H3. `EvictionCache` does not support batch_size > 1 in its eviction path
**File:** `turboquantdc/token_eviction.py` (`_EvictableLayer` extends
`_CompressedLayer`)
**Observation:** Eviction decisions are made per-token-position (single
sequence). The structural-importance scorer has no `batch_idx` plumbing.
For batched inference (vLLM serves concurrent requests), each sequence
needs its own eviction state, but `_EvictableLayer` keeps a single
`_position_importance` array. Eviction will mix sequences.

**Mitigation for the demo:** Don't use `EvictionCache` with batch > 1.
Stick to `GenerationCache` for vLLM.

### H4. `_FP16AnchorLayer` and `FP16Layer` reorder() handling does not match
batch_size > 1 semantics
**File:** `turboquantdc/adaptive_generation_cache.py:630-632`,
`turboquantdc/adaptive_hf_cache.py:90-92`
**Observation:** `reorder(beam_idx)` calls `index_select(0, beam_idx)` on
the FP16 buffers but the compressed entries (in `_AdaptiveLayer`) get
silently no-op'd with a comment "(beam search with adaptive cache is rare in
practice)". This is a correctness bug. For vLLM the bug doesn't trigger
(no beam search), but it disqualifies the cache for any beam-search caller.

### H5. `streaming_70b.AsyncPrefetcher` does not use pinned memory
**File:** `turboquantdc/streaming_70b.py:189-216`
**Observation:** `layer.to(self.device, non_blocking=True)` requires the
source tensors to be in pinned memory for `non_blocking` to actually be
async. The CPU-resident layers loaded via `from_pretrained(device_map="cpu")`
are NOT pinned by default. `non_blocking=True` silently degrades to a
synchronous transfer. The "double-buffered prefetch" in the docstring is
thus mostly synchronous in practice.

**Fix:** After `model.eval()`, call `module._apply(lambda t: t.pin_memory())`
on the layers that will be streamed.

---

## MEDIUM

### M1. Mutable defaults in dataclass
**File:** `turboquantdc/v2_cache.py:58-82`
**Observation:** `V2Config` uses `field(default_factory=lambda: [...])` — this
is correct, NOT a mutable default. False alarm avoided. (Noting because a
naive scan would flag it.)

### M2. `LloydMaxCodebook` instances allocated per-layer per-bits combination
**File:** `turboquantdc/generation_layers.py` (lazy init in `_CompressedLayer`)
**Observation:** Every layer's codebook is independently constructed even
though `(d, bits)` is constant across layers. The result is identical math
but N_layers wasted scipy.integrate calls. `cross_layer_kv.py` documented
this exact insight (lines 16-23), but production `GenerationCache` does not
share codebooks. Memory cost is negligible (a few KB per codebook); the
real cost is startup latency and the scipy import per-process.

**Fix:** Module-level `@functools.lru_cache` on the codebook constructor
keyed by `(d, bits, device)`.

### M3. `print()` in production cache code paths
**File:** Multiple — verify with `grep -n "print(" turboquantdc/generation_*.py`
**Observation:** The production `generation_core.py` itself is clean, but
`streaming.py` and `streaming_70b.py` use `print()` for progress reporting.
Wrap these in `logging` calls so vLLM's log handler can route them.

### M4. `Any` overuse in cache APIs
**File:** Multiple — e.g. `vllm_integration.py:331` returns `Dict[str, object]`,
`cross_layer_kv.py` uses `Dict[str, Any]` for stats.
**Observation:** Not blocking, but `TypedDict`s for `CompressedKey` and
`CompressedValue` would catch field-name typos at lint time. The current
`CompressedKey = Dict[str, torch.Tensor]` alias has no compile-time
guarantee on which keys exist.

### M5. Dual classes named `FP16Layer` in different modules
**File:** `turboquantdc/adaptive_hf_cache.py:42`, `turboquantdc/ultimate_cache.py:35`,
`turboquantdc/generation_layers.py` (`_FP16Layer`)
**Observation:** Three FP16 layer classes with overlapping but
non-identical APIs. `__init__.py` exposes `FP16Cache` (from `layer_adaptive`)
which is yet another. None of them subclass a shared protocol. New
contributors will burn an hour figuring out which to inherit from.

**Fix:** Consolidate to `_FP16Layer` in `generation_layers.py` and have
the others import from there. Already mostly true for production code.

### M6. `ChunkedPrefillEngine` recreates the cache on every prefill
**File:** `turboquantdc/chunked_prefill.py:160-162`
**Observation:** `self.cache = TurboQuantCache(bits=self.bits, ...)` resets
state on each `prefill()` call. For batched prefill use cases (which is
exactly what vLLM does), this would clobber the previous request's KV.
Fine for the standalone demo (single document at a time), wrong for vLLM.

---

## LOW

- L1. `streaming_70b.py:233`: `OVERHEAD_GB = 1.5` is a magic number; should
  be a configurable class attribute or derived from `torch.cuda.mem_get_info()`.
- L2. `retrieval_cache.py:121`: silent `effective_nlist = min(nlist, n_train // 39)`
  — should warn or log when degrading to FlatIP.
- L3. `temporal_delta.py:4-29`: docstring is a 25-line "this doesn't work"
  caveat. If the file ships, the caveat should be a `DeprecationWarning` at
  import time, not buried in a docstring.
- L4. `cross_layer_predict.py:31-33`: hardcoded paths `/media/dhawal/Beast/cache/hub/`
  — would crash on any other machine. Belongs in a benchmark script under
  `benchmarks/` or `tools/`, not in `turboquantdc/`.

---

## TL;DR for the 6-hour ship

Drop everything but `generation_cache.py` (production), `token_eviction.py`
(secondary), `vllm_integration.py` (the surface area) and the `kv_cache.py`
+ `hf_integration.py` HF compat shims. Move the other 16 files in this
review into `docs/research_artifacts/<topic>/` to clean the public namespace.
For Qwen3.6-27B at 32K context on a 4090 with 3-bit, `GenerationCache`
alone fits and is the only cache with proven Qwen-on-real-PPL numbers.

The vLLM integration in `vllm_integration.py` is **not yet
PagedAttention-compatible** — the `block_size` argument is a no-op and the
buffers are flat. Either (a) reshape buffers to `(num_blocks, block_size, H, D)`
in the next 2 hours, or (b) bypass `TurboQuantCacheManager` and use the
`AttentionImpl` monkey-patch path documented in the module docstring.
Path (b) is the lower-risk demo strategy.

