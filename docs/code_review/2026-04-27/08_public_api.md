# Public API & Integration Review (Opus 4.7)

Reviewer scope: public API, packaging, vLLM/HF integration, README, CI.
Date: 2026-04-27
Repo state: master @ 5306ec7, v0.3.0, scipy 1.17.1 / transformers 5.5.0 / Python 3.13 in dev shell.

Bottom line: **`turboquantdc/vllm_integration.py` is a docstring-only sketch with zero `import vllm` calls and a monkey-patch example that targets a vLLM internal layout that does not exist in vLLM 0.6+ (and especially not in the V1 engine that ships with 0.7+/0.8+/0.9+). It cannot be wired into vLLM tonight without surgery.** The `hf_integration.TurboQuantCache` IS a viable, end-to-end-tested path and is the recommended fallback (it actually works against transformers 5.5.0 today). For shipping in 6 hours, do not touch vLLM — use the HF path or the existing `run_70b.py` launcher.

---

## Minimum-Viable vLLM Wiring (the most important section)

### Recommendation: do NOT use vLLM tonight. Use HF transformers instead.

The vLLM integration in this repo is theatre. It can be made real, but not in 6 hours by one person against vLLM 0.7+ with a V1 engine and AWQ-INT4 weight loading. The layout `llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers` exists in older legacy V0 vLLM (~0.5.x). It does NOT exist in the V1 engine that has been default since vLLM 0.8 (Q4 2025). Modern vLLM dispatches attention through `vllm/v1/attention/backends/` with a `MultiHeadAttention` op, not via per-layer `attn_layer.self_attn.attn.forward` callables that can be monkey-patched.

If the user insists on vLLM 0.19+ (which I cannot verify is even a real released version — vLLM is currently at ~0.10.x as of late 2026), the patch would have to:
1. Subclass `vllm.attention.backends.abstract.AttentionBackend` AND `AttentionImpl` AND `AttentionMetadata`.
2. Register the backend via `VLLM_ATTENTION_BACKEND=TURBOQUANT` env var or a private registry shim.
3. Implement `forward_decode` and `forward_prefill` over paged blocks (not flat tensors as the current `compute_attention` assumes).
4. Update `kv_cache_dtype` enum and the cache engine to allocate the 6-tensor compressed layout in paged-block form.

That is a 2–4 week project, not a 6-hour project.

### Files needed (HF fallback — what to actually use tonight)

```
turboquantdc/__init__.py                 -- top-level exports
turboquantdc/hf_integration.py           -- TurboQuantCache (the real integration)
turboquantdc/generation_core.py          -- GenerationCache (better quality, has mean-removal)
turboquantdc/generation_layers.py        -- _CompressedLayer (used by GenerationCache)
turboquantdc/generation_strategy.py      -- anchor schedule helpers
turboquantdc/codebook.py                 -- LloydMaxCodebook
turboquantdc/rotation.py                 -- WHT/QR rotation
turboquantdc/polarquant.py               -- Stage 1
turboquantdc/qjl.py                      -- Stage 2 (used by TurboQuantEstimator only)
turboquantdc/estimator.py                -- TurboQuantEstimator
turboquantdc/e8_lattice.py               -- E8Quantizer (the v0.3.0 quality lever)
turboquantdc/triton_kernels.py           -- optional fast path
run_70b.py                               -- existing one-shot launcher; works
```

`turboquantdc/vllm_integration.py` should NOT be on the critical path tonight.

### Integration point (HF path)

The integration point is `transformers.GenerationMixin.generate(..., past_key_values=cache)`. The cache class is `turboquantdc.GenerationCache` (preferred — has mean-removal and E8) or `turboquantdc.TurboQuantCache` (HF subclass-compatible duck type). HF transformers calls `cache.update(key_states, value_states, layer_idx)` for each layer per forward pass.

### Sequence of calls (concrete code that will run on Qwen3.6-27B-AWQ-INT4)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from turboquantdc import GenerationCache

MODEL = "cyankiwi/Qwen3.6-27B-AWQ-INT4"  # already in the local HF cache

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
)
num_layers = model.config.num_hidden_layers  # 64 for Qwen3.5-27B; verify for 3.6

cache = GenerationCache(
    key_bits=3,
    val_bits=3,
    fp16_window=128,
    anchor_strategy="boundary",
    num_layers=num_layers,
    use_residual_quant=True,
    center_before_quantize=True,        # the "mean removal fix" from README
    quantizer_type="e8",                # near-lossless 3-bit per HANDOFF
)

inputs = tok("Explain quantum computing:", return_tensors="pt").to(model.device)
out = model.generate(
    **inputs,
    past_key_values=cache,
    max_new_tokens=256,
)
print(tok.decode(out[0], skip_special_tokens=True))
```

This is the smallest viable path. It already works for Qwen2.5-7B / 14B / Gemma-2-9B per HANDOFF.md. The change for Qwen3.6-27B-AWQ-INT4 is: pass the right `num_layers` (the AWQ wrapper preserves config) and confirm `head_dim` is 256 (Qwen3-style) and is power-of-2 (it is) so WHT path works.

### Risks / unknowns for shipping tonight

1. **AWQ-INT4 weight quant interaction**: TurboQuantDC quantizes K/V activations, AWQ quantizes weights. They are independent in theory, but `model.generate()` with a custom `past_key_values` has not been tested against an AWQ-loaded model in this repo. I see no AWQ-specific test in `tests/`. There is a non-zero chance the AWQ kernel shape assertions reject the dequantized tensor dtype returned by `_dequantize_all()` (it casts back to `self._dtype` which is the model's K/V dtype).
2. **Qwen3.6 architecture is unknown**: `cyankiwi/Qwen3.6-27B-AWQ-INT4` exists in the user's HF cache but `huggingface_hub.model_info` returns 401 (private/gated) — I could not verify the architecture. The closest cached model is `Qwen/Qwen3.5-27B`: 64 layers, 4 KV heads, head_dim 256. The static lookup in `vllm_integration.py` says `qwen3.5-27` is num_layers=62 and num_kv_heads=8 — both wrong. If `Qwen3.6` has the same layout as `Qwen3.5`, then **do not rely on `get_turboquant_config`** — read `model.config` directly.
3. **head_dim 256 + WHT**: 256 IS a power of 2, so WHT path should work. But d=256 is not power-of-2-favored in any benchmark in HANDOFF.md (most benchmarks at d=128). Validation pre-flight required.
4. **Layer 0 anchor**: HANDOFF says "Layer 0: always needs FP16 anchor". `anchor_strategy="boundary"` covers layers 0,1,N-2,N-1 — good. Do not use `anchor_strategy="fixed"` with `anchor_interval=12` for the production run.
5. **Memory budget**: Qwen3-27B-AWQ-INT4 is ~14 GB weights on disk. With 24 GB VRAM and 4 KV heads at d=256, KV at FP16 = 2 * 64 * 4 * 256 * 2 bytes = 256 KB per token. At 3-bit + anchors, ~50 KB per token. Budget for 200K tokens KV is ~10 GB, fits.
6. **transformers 5.5 Cache protocol drift**: The repo's HF caches lack `early_initialization` and `has_previous_state` methods that the 5.5 `Cache` base class declares. They duck-type, so this works *until* HF adds an `isinstance(Cache)` check. Today (5.5.0) it works — `model.generate(past_key_values=GenerationCache(...))` runs.
7. **No vLLM in `.venv-vllm`**: The venv at `/home/dhawal/turboQuantDC/.venv-vllm` does not have vLLM installed. Even if you wanted to try the monkey-patch, you'd have to `pip install vllm` first, and at that point you'd discover the V1 engine layout mismatch.

### If the user truly wants vLLM, the 4-hour cap is "host the API"

Option: run `vllm serve cyankiwi/Qwen3.6-27B-AWQ-INT4` with stock vLLM (no TurboQuant) and use the OpenAI-compatible API for end-user serving. Then run a separate HF-transformers process for the TurboQuant 3-bit comparison. Two side-by-side servers, no integration. This is honest and shippable in 30 minutes.

---

## __init__.py Audit

### What's exported (113 names in `__all__`)

Stage modules: `LloydMaxCodebook`, `PolarQuant`, `QJL`, `TurboQuantEstimator`, `TurboQuantKVCache`. OK.

Caches: `GenerationCache`, `HybridCache`, `TurboQuantCache` (HF), `LayerAdaptiveKVCache`, `FP16Cache`, `AsymmetricKVCache`, `AsymmetricTurboQuantCache`, `AsymmetricTurboQuantLayer`, `EvictionCache`, `SelfCorrectingCache`, `UltraValueCache`, `ResidualVQCache`, `ResidualQuantCache`, `AdaptiveBitsCache`, `AdaptiveGenerationCache`, `ExpectedAttentionCache`, `ChannelAdaptiveCache`, `CrossLayerKVCache`, `TurboQuantV2Cache`, `AttentionGatedCache`, `RetrievalKVCache`, `TemporalDecayCache`. **22 cache classes is a discoverability problem.** A new user reading the README will not know which one to use. The README example uses `GenerationCache` with kwargs that don't exist (see CRITICAL section).

vLLM: `TurboQuantAttentionBackend`, `TurboQuantCacheManager`, `get_turboquant_config`. These import cleanly (no vllm dependency at import time) but as noted, do not actually integrate with vLLM.

### What should be exported but isn't

- `TurboQuantLayer` (from hf_integration) — useful for users wiring custom attention. Not exported.
- `_compute_attention_entropy` IS exported but with a leading underscore — that's a code smell (private name in public API).

### What shouldn't be exported

- `_compute_attention_entropy` — leading underscore, listed in `__all__`. Either rename or hide.
- `run_model` — pulls `from run_70b import run_model` at call time. `run_70b.py` is in the repo root, NOT in the package. After `pip install turboquantdc`, this function will raise `ModuleNotFoundError: No module named 'run_70b'`. **This is a packaging bug.**
- `compress_model`, `effective_bpw`, `estimate_compressed_size` — weight compression API mixed in with KV API; should be a sub-namespace.

### Discoverability

The README points users at `GenerationCache` (correct) but the docstring on `__init__.py` lists modules that don't all exist any more (`kv_cache`, `vllm_integration` listed; `e8_lattice`, `residual_quant`, `generation_core` not). Update the module docstring.

`hf_integration.TurboQuantCache` is hard to find. It is exported but not mentioned by name in the README quickstart. If you want users to actually use it, the README should say "use `TurboQuantCache` for HuggingFace compatibility" — instead it shows `GenerationCache` with broken kwargs.

---

## CLAUDE.md Staleness (cite line numbers)

`/home/dhawal/turboQuantDC/CLAUDE.md` describes a project that does not exist any more. The repo is at v0.3.0 with 1,818+ tests; CLAUDE.md describes "Phase 1 (Core Algorithm) has not started".

Specific lines that are wrong:

- **Line 11**: "All source files in `turboquantdc/` and `tests/` are empty stubs. `setup.py` is empty. Phase 1 (Core Algorithm) has not started." — flat false. There are 70+ source modules, 44 test files, and `setup.py` has 75 lines.
- **Line 17–18**: "Dependencies (not yet in a requirements.txt — create one when starting) `pip install torch scipy`" — `requirements.txt` exists and pyproject.toml + setup.py specify deps.
- **Line 30–33**: References `cd reference/tonbistudio-ref && python validate.py` as a current dev workflow. This is research-archive code; it is not part of the v0.3.0 development loop.
- **Line 40**: "Python: 3.12" — the working venv uses Python 3.13 (Anaconda). Minor.
- **Lines 66–101**: The "Module Pipeline" diagram is largely obsolete. It lists `kv_cache.py` as the integration target, but the production cache today is `generation_core.GenerationCache` with mean-removal + E8 lattice — neither in the diagram.
- **Lines 109–115 ("Success Metrics")**: The metrics are paper targets; the project has hit them and moved beyond them per HANDOFF.md (E8 lattice +0.001% PPL on 3B). CLAUDE.md still presents them as goals.
- **Lines 119–130 ("Implementation Rules")**: "Don't try to integrate with vLLM/SGLang until the core algorithm is validated" — the project has now done both rounds and is actively trying to integrate with vLLM. The rule is stale.
- **Line 144 onwards ("Workflow")**: References `PLAN.md` as "the latest task checklist". PLAN.md was last modified 28 Mar 2026; HANDOFF.md (15 Apr) is the current source of truth. CLAUDE.md does not mention HANDOFF.md.

Recommendation: replace CLAUDE.md content with a thin wrapper pointing to HANDOFF.md as the current state, plus the actual API surface (GenerationCache, TurboQuantCache, E8). 30 minutes of work.

---

## Packaging Audit (setup.py, pyproject.toml, requirements*.txt)

### CRITICAL packaging issues

1. **`run_model` will crash after pip install** (`turboquantdc/__init__.py:234`):
   ```python
   from run_70b import run_model as _run_model
   ```
   `run_70b.py` is at the repo root, NOT inside the `turboquantdc/` package. The wheel does not include it (verified: `unzip -l dist/turboquantdc-0.3.0-py3-none-any.whl | grep run_70b` is empty). Any user who runs `from turboquantdc import run_model; run_model()` after a clean `pip install` gets `ModuleNotFoundError: No module named 'run_70b'`. Either move `run_70b.py` into the package as `turboquantdc/run_70b.py` and add a console_scripts entry, or remove `run_model` from `__all__`.

2. **CUDA `.cu` source files are missing from the wheel** (`MANIFEST.in:6`):
   ```
   recursive-include turboquantdc *.py
   ```
   This pattern only includes `.py` files. The CUDA kernels at `turboquantdc/cuda/dequantize.cu` and `turboquantdc/cuda/wht.cu` ARE referenced by `turboquantdc/cuda/build.py:65` (`os.path.join(src_dir, "dequantize.cu")`) but they are NOT shipped in the wheel. I verified by `unzip -l` of `dist/turboquantdc-0.3.0-py3-none-any.whl`: no `.cu` files.
   
   Result: any user who installs from PyPI and tries to use the raw CUDA path falls back to "kernel compilation failed". The local dev environment hides this because the kernels were JIT-built once and cached in `~/.cache/turboquantdc_cuda/`. Fix: add `recursive-include turboquantdc *.cu *.cuh *.h *.hpp` to MANIFEST.in.

3. **scipy upper bound is too tight**: `scipy>=1.10.0,<1.15.0` in setup.py, pyproject.toml, requirements.txt. The currently working dev environment has scipy 1.17.1. The pin would force pip to reject scipy 1.15+ even though the code uses `scipy.integrate.quad` and `scipy.special.gamma` which have been stable since scipy 0.x. The pin appears to have been added defensively but is wrong. Either remove the upper bound or extend to `<2.0.0`.

4. **No torch upper bound, no CUDA toolchain pin**: `torch>=2.0.0` accepts any torch (CPU, ROCm, CUDA 11, CUDA 12, etc). Triton kernels and the JIT CUDA kernels assume CUDA 12+ and SM 80+ (A100, RTX 3090, RTX 4090). On a clean install with `torch+cu118`, the triton kernels will mis-target. Acceptable for "alpha" release, but not for a production claim.

### HIGH packaging issues

5. **`extras_require={"base": _base, ...}` repeats the install_requires**: `pip install turboquantdc[base]` is a no-op — `_base` is already in `install_requires`. Only matters for documentation hygiene.

6. **`faiss-gpu` wheel doesn't exist for modern Python on PyPI**: `_faiss = ["faiss-gpu"]` (setup.py:24-26). On Python 3.12+, `pip install faiss-gpu` fails. The code already handles the import error with try/except in `__init__.py:148-156`, but the extra is broken. Either pin `faiss-gpu==1.7.2` (last available) or remove the extra.

7. **Two parallel package metadata files** (`setup.py` + `pyproject.toml`) with mostly-but-not-quite-identical content. Modern packaging is `pyproject.toml` only. The duplication is a maintenance trap (e.g., the `_dev` extras differ subtly: setup.py has `_dev`, pyproject does not list it under `[project.optional-dependencies]`).

8. **`requirements_demo.txt` exists but is not referenced anywhere** (`/home/dhawal/turboQuantDC/requirements_demo.txt`). Dead file.

### MEDIUM packaging issues

9. The `[project.urls]` Homepage points to `https://github.com/turboquantdc/turboquantdc` but the actual repo is `https://github.com/dhawalc/turboQuantDC` (per README link). The wrong URL goes onto PyPI metadata.

10. `package_dir` is implicit (find_packages). With `find_packages(exclude=["tests*", ...])`, the `turboquantdc/cuda/` and `turboquantdc/kernels/` subpackages are auto-detected. `kernels/` is empty, so an empty package gets shipped. Trivial but messy.

11. `Development Status :: 3 - Alpha` is in pyproject.toml. After 1,818 tests and ~5 weeks of validation, this should be at least `4 - Beta`.

---

## CRITICAL findings

### CRIT-1: README quickstart code does not run

`/home/dhawal/turboQuantDC/README.md:140-150` shows:
```python
cache = GenerationCache(
    num_layers=36,
    num_heads=2,        # NOT a kwarg
    head_dim=128,       # NOT a kwarg
    bits=3,             # NOT a kwarg (it's key_bits/val_bits)
    mean_removal=True,  # NOT a kwarg (it's center_before_quantize)
    residual_quant=True,# NOT a kwarg (it's use_residual_quant)
    fp16_window=128,
    anchor_layers=[0, 1, -2, -1],  # NOT a kwarg (it's anchor_strategy="boundary")
    device="cuda"       # NOT a kwarg
)
```

I tested it. It raises `TypeError: GenerationCache.__init__() got an unexpected keyword argument 'num_heads'`. **The very first code block in the README fails on the first kwarg checked.**

The actual signature (`generation_core.py:157-172`) is:
```python
def __init__(self, key_bits=4, val_bits=3, fp16_window=64, anchor_interval=12,
             anchor_strategy="fixed", num_layers=None, seed=42,
             use_norm_correction=True, use_residual_quant=True,
             rotation_type=None, use_triton=..., center_before_quantize=True,
             quantizer_type="lloyd_max")
```

Fix the README to:
```python
cache = GenerationCache(
    num_layers=36,
    key_bits=3,
    val_bits=3,
    fp16_window=128,
    anchor_strategy="boundary",
    use_residual_quant=True,
    center_before_quantize=True,
)
```

### CRIT-2: vLLM integration is non-functional

`turboquantdc/vllm_integration.py` does not import vLLM. The classes work standalone (verified) but they are not vLLM backends. They are stubs with vLLM-shaped names. The docstring example at lines 25–54 shows a monkey-patch path against `llm.llm_engine.model_executor.driver_worker.model_runner.model.model.layers` — this attribute path exists in legacy V0 vLLM (~0.5.x) and was removed in the V1 engine refactor. The example is fictional for current vLLM.

The `compute_attention` implementation (line 260) does a full materialised `(Q, K)` softmax and a `(Q, K) @ (K, d)` matmul. This is NOT compatible with vLLM's paged attention scheduler, which expects per-block kernels operating on indexed cache slots. Even if the monkey-patch path is fixed for current vLLM, this implementation cannot be plugged into the production attention path.

`TurboQuantCacheManager` allocates `(max_seq_len, num_kv_heads, d)` per layer (line 519). vLLM's CacheEngine allocates `(num_blocks, block_size, num_kv_heads, d)`. These shapes are incompatible.

### CRIT-3: `turboquantdc.run_model` raises `ModuleNotFoundError` after pip install

(See packaging item #1 above.) `__init__.py:234`: `from run_70b import run_model as _run_model`. `run_70b.py` is at the repo root, never gets into the wheel. Anyone who follows the docstring "One-line API to run any 70B model" gets a crash on first call. This is a regression-test gap and a docs gap.

### CRIT-4: CUDA `.cu` source files missing from packaged wheel

(See packaging item #2 above.) Verified via `unzip -l dist/turboquantdc-0.3.0-py3-none-any.whl`. Any clean `pip install turboquantdc` followed by use of the JIT CUDA path will fall through to the "compilation failed" branch and silently degrade to PyTorch-only (which works, but defeats the marketing claim of CUDA kernels).

---

## HIGH

### HIGH-1: vllm_integration `_MODEL_CONFIGS` has wrong values for Qwen3.5-27B

`turboquantdc/vllm_integration.py:795`:
```python
"qwen3.5-27":  {"num_layers": 62, "num_kv_heads": 8,  "head_dim": 256},
```

Actual `Qwen/Qwen3.5-27B/config.json` (verified locally):
```
num_hidden_layers: 64   (NOT 62)
num_key_value_heads: 4  (NOT 8)
head_dim: 256           (correct)
```

Two of three fields are wrong. The HF-config fallback (`_try_load_hf_config`) would catch this on the public model name but fails for the user's actual target `cyankiwi/Qwen3.6-27B-AWQ-INT4` because that repo is private/gated (401 from `model_info`). For tonight's target, **the config will be wrong**. Recommend reading `model.config` from the loaded model instead of the static lookup.

### HIGH-2: HF Cache subclass status — duck-typed only

`turboquantdc.TurboQuantCache` (`hf_integration.py:435`) is NOT a subclass of `transformers.cache_utils.Cache`. It duck-types the protocol. I verified `isinstance(cache, Cache) == False`.

`transformers >= 5.0` has tightened isinstance checks in some `generate()` paths (e.g., `_get_initial_cache_position`). On 5.5.0 it currently works (verified end-to-end with `cache.update(k, v, layer_idx=0)`), but the cache is missing `early_initialization` and `has_previous_state` methods that the current `Cache` base class declares. Add a proper subclass plus the two missing methods (both can be no-ops). 30 minutes.

### HIGH-3: vllm_integration `compute_attention` does not support GQA

Lines 305-322: scores are computed as `(Q, K)` and value reconstruction is `(K, d)`, then `(Q, K) @ (K, d)`. There is no head-broadcasting. For Qwen3.6-27B (which has GQA: 4 KV heads, ~24 attention heads), attention queries from 6 different attention heads must share the same KV head. The current implementation assumes 1 KV head per query head. This will produce wrong attention scores under GQA.

### HIGH-4: vllm_integration `TurboQuantAttentionBackend.__init__` allocates 80+ Lloyd-Max codebooks at construction

For an 80-layer model with `bits=3`, the backend creates 80 `TurboQuantEstimator` + 80 `PolarQuant` instances. Each codebook runs `scipy.integrate.quad` Lloyd-Max iterations on CPU at construction (`codebook.py` precomputes once per (d,bits) pair, but the cache key may vary). This blocks the constructor for several seconds. Should use a shared codebook table.

### HIGH-5: Mutable seeding scheme guarantees collisions for >= 1000 layers

`vllm_integration.py:175`: `seed=layer_idx * 1000` for keys, `seed=layer_idx * 1000 + 500` for values. Internal `TurboQuantEstimator` uses `seed+1` for QJL. So layer 0 keys: seeds 0 (rotation) and 1 (QJL). Layer 0 values: seed 500. Layer 1 keys: 1000, 1001. With `mse_only=True` path elsewhere, the value seed (500) and the QJL seed of layer 0 (1) won't collide. But the gap of 1000 means model layers >= 1000 (no current model has this many) would alias the layer-0 QJL seed. Trivial issue, mention only because the comment claims seeds are spaced "to avoid accidental correlation" — they are not in any rigorous sense.

### HIGH-6: `__init__.py` lists private name in `__all__`

`__init__.py:310`: `"_compute_attention_entropy"`. Public API should not have leading-underscore names. Either rename to `compute_attention_entropy` or drop from `__all__`.

### HIGH-7: Two duplicate package files (setup.py + pyproject.toml)

Both files declare the same metadata and extras with subtle drift. Modern packaging is pyproject-only. Risk: extras_require differs, so `pip install turboquantdc[dev]` behaves differently from a build using setup.py.

---

## MEDIUM

- **MED-1**: `__init__.py:1-14` module docstring lists modules `kv_cache`, `vllm_integration` but omits `e8_lattice`, `residual_quant`, `generation_core`, `hf_integration` — the actual production modules. Misleading.
- **MED-2**: `__init__.py:217-240` `run_model` defaults to `meta-llama/Llama-3.3-70B-Instruct` which is gated. A user calling `run_model()` without a HF token gets a permissions error. Default should be a non-gated model (e.g., `Qwen/Qwen2.5-7B-Instruct`).
- **MED-3**: `vllm_integration.py:194-199` `_compressed_key_store` and `_compressed_value_store` are instantiated as `[[] for _ in range(num_layers)]` and never populated by `compress_kv` (which returns the result instead). `memory_usage()` (line 331) reports based on these stores — it always returns 0. Misleading API.
- **MED-4**: `hf_integration.py:541-555` `batch_repeat_interleave` mutates entries in-place via `entry[key] = entry[key].repeat_interleave(...)` while iterating `for key in entry` — works in Python 3 (dict view), but is unsafe if dict keys change. The simpler `for key in list(entry)` is safer.
- **MED-5**: `setup.py:46` `url="https://github.com/turboquantdc/turboquantdc"` does not match the README link `dhawalc/turboQuantDC`. Mismatch will publish wrong URL to PyPI.
- **MED-6**: `pyproject.toml:34` classifies as "Operating System :: OS Independent". The CUDA kernel and triton paths are Linux-only / NVIDIA-only. Should be `Operating System :: POSIX :: Linux` and `Environment :: GPU :: NVIDIA CUDA :: 12`.
- **MED-7**: `requirements-dev.txt` does not include `ruff` (referenced by `pyproject.toml:[tool.ruff]`) or `mypy` (referenced by `[tool.mypy]`). Anyone running `ruff check .` after `pip install -r requirements-dev.txt` will fail.
- **MED-8**: `MANIFEST.in:7-11` `prune docs` removes the very `code_review/` dir users may want to read. Marginal.
- **MED-9**: `.github/workflows/pages.yml` is the ONLY CI file. There is no test workflow, no linter workflow, no build workflow. With 1,818 tests, a CI run on PR is the obvious move.
- **MED-10**: `hf_integration.py:457` `is_compileable = False` is a class attribute. `transformers` 5.x checks for this on instances; `Cache.is_compileable` is a `@property` in the base. Setting it as a class attribute works but inhibits subclasses overriding via property. Cosmetic.
- **MED-11**: `generation_core.py:166-170` `use_triton=_TRITON_AVAILABLE` evaluates `_TRITON_AVAILABLE` at import time. If the user installs triton AFTER importing turboquantdc, the default stays `False` for the session. Document or use a lazy default.

---

## LOW

- **LOW-1**: `__init__.py:148-156` silently swallows `ImportError` for retrieval modules. If `faiss-gpu` is partially installed with a broken native lib, the silent pass hides the real error. Log the failure once at debug level.
- **LOW-2**: `vllm_integration.py:597` `signs_01 = ((ck["qjl_signs"] + 1.0) * 0.5).to(torch.int8)`. The expression `(qjl_signs + 1.0) * 0.5` is doing the obvious mapping but allocates two intermediate float tensors. `((ck["qjl_signs"] > 0).to(torch.int8))` is one tensor and the same result.
- **LOW-3**: `__init__.py:70-71` exports `_compute_attention_entropy` from `generation_cache` (the small wrapper file). `generation_hybrid.py:30` is the canonical implementation. Two paths for the same name confuses `find_definition` IDE features.
- **LOW-4**: `setup.py:10` `_base` and pyproject's `[project.optional-dependencies].base` both repeat install_requires. `pip install turboquantdc[base]` does nothing. Hide it or document it.
- **LOW-5**: `LICENSE` is MIT but `pyproject.toml:10` uses `license = "MIT"` (string). Modern pyproject prefers `license = {text = "MIT"}` or SPDX `license = "MIT"` with `[tool.setuptools]` license_files. Trivial.
- **LOW-6**: README shows `pip install turboquantdc[all]` (line 125). The `all` extra includes `faiss-gpu` which fails on Python >= 3.12. Users who run the documented install command on a fresh Python 3.12+ environment get an installer error. Either gate `faiss-gpu` to py < 3.12 with environment markers, or drop it from `all`.

---

## Final integration recommendation (one screen)

For tonight's 6-hour goal "Qwen3.6-27B-AWQ-INT4 with 3-bit KV on 4090":

1. **Use HF transformers, not vLLM.** Path: `model.generate(past_key_values=GenerationCache(...))`. This works today.
2. Read `num_layers`, `num_kv_heads`, `head_dim` from `model.config` directly. Do NOT trust `get_turboquant_config`.
3. Use `GenerationCache(key_bits=3, val_bits=3, anchor_strategy="boundary", num_layers=N, fp16_window=128, use_residual_quant=True, center_before_quantize=True, quantizer_type="e8")`. This is the configuration documented in HANDOFF.md as PROVEN near-lossless.
4. Validate against a 256-token sanity prompt before running long context. Compare top-1 token vs FP16 baseline.
5. Fix the README quickstart code IMMEDIATELY (CRIT-1). It is the first thing every visitor sees and it raises TypeError on every kwarg.
6. Update CLAUDE.md or delete it. It actively misleads an agent reading the repo.
7. Consider deferring vllm_integration.py to v0.4.0. Do not advertise it as working in the README. It is research-grade scaffolding, not a vLLM plugin.

If vLLM is non-negotiable: ship a separate `vllm serve` process for the OpenAI API and run TurboQuant for the comparison/research path in a sidecar HF process. Two servers, no risky integration. Half a day to wire up.
