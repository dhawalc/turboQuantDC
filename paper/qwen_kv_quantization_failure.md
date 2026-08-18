# Diagnosing Catastrophic Low-Bit KV-Cache Quantization Failure in Qwen Models

**Author:** Dhawal Chheda
**Draft date:** 2026-08-17
**Status:** Working manuscript. Not submitted. Numbers are from previously-run,
committed experiments; Section 7 lists the ablations still outstanding.
**Repository:** https://github.com/dhawalc/turboQuantDC

---

## Abstract

Low-bit KV-cache quantization can substantially reduce memory requirements during
large-language-model inference, but its behavior varies across model architectures.

During experiments with an independent implementation of a TurboQuant-family
rotate-then-scalar-quantize KV-cache compressor, we observed catastrophic
perplexity degradation on Qwen2.5-7B-Instruct under a tested 3-bit configuration:
perplexity increased from 7.52 with an uncompressed KV cache to 9,410.5 on a
4,095-token wikitext-2 evaluation. The same configuration produced fluent-looking
degenerate output and 0/3 needle recall at 8K context.

We investigate the source of this failure. The tested pipeline normalizes each key
vector to unit length before rotation and scalar quantization, so its quantizer
budget is spent on key *direction*. We find that when a per-head key mean is large
relative to the per-token deviation around it, every normalized key points in
nearly the same direction, the rotated coordinates no longer resemble the
zero-mean distribution the Lloyd-Max codebook was designed for, and the
token-discriminative component of the key — the only component softmax responds to
— is quantized away.

Subtracting the per-head running key mean before quantization and restoring it
after dequantization reduced perplexity from 9,410.5 to 7.90 under the same
experimental configuration, and restored 3/3 needle recall.

Critically, the failure is **not uniform across the Qwen family**. In a separate
sweep across three model sizes, Qwen2.5-3B (2 KV heads) and Qwen2.5-7B (4 KV heads)
both failed catastrophically without mean removal, while Qwen2.5-14B (8 KV heads)
did not fail at all (PPL 5.54 vs. an FP16-KV baseline of 4.94). The failure in our
data tracks KV-head count under grouped-query attention rather than model family or
parameter count. We report this as an observed correlation over three models, not
an established architectural law.

This paper documents the failure mode, experimental methodology, proposed
mechanism, correction, and limitations, and provides reproducible artifacts for
independent evaluation.

---

## 1. Introduction

**Problem.** KV-cache memory increasingly constrains long-context LLM inference. At
long context the cache, not the weights, dominates accelerator memory, and low-bit
quantization of cached keys and values is one of the few interventions that
addresses this without changing the model.

**Observation.** Low-bit KV quantization does not fail uniformly across
architectures. A configuration that is near-lossless on one model can be
catastrophic on another with the same head dimension, the same bit-width, and the
same code path. The failure we report is not a graceful degradation: perplexity
rose by three orders of magnitude, and generated text collapsed into digit and
punctuation fragments.

**Research question.** Why does the tested 3-bit configuration degrade
catastrophically on Qwen2.5-7B-Instruct while apparently reasonable quantization
parameters are used, and while the same configuration is nearly lossless on
Qwen2.5-14B-Instruct?

**Contribution.** We:

1. document a reproducible catastrophic failure of a TurboQuant-family 3-bit KV
   compressor on Qwen2.5-7B and Qwen2.5-3B, with committed raw artifacts;
2. isolate the failure to the **key** quantization path (it persists when values
   are left uncompressed);
3. propose a mechanism — violation of the rotated-coordinate distributional
   assumption by a large shared per-head key component, under a pipeline that
   normalizes keys before quantizing direction;
4. evaluate a correction (per-head key mean removal with exact restoration) that
   recovers to within +0.38 PPL of the uncompressed-KV baseline; and
5. report a correlation between failure severity and KV-head count that
   constrains, but does not yet establish, the generality of the mechanism.

We deliberately restrict the claim. We do not claim that canonical TurboQuant fails
on Qwen; we claim that *this* TurboQuant-family configuration does, and we describe
the configuration precisely enough to be checked.

---

## 2. Background and Related Work

> **Reference-verification note.** Citation details in this draft were written
> without network access. Every entry in the References section must be verified
> against the published record before this manuscript is circulated externally.
> Entries whose venue or identifier we could not confirm offline are marked
> `[verify]`.

### 2.1 KV-cache quantization

During autoregressive decoding, transformer inference caches per-layer key and
value tensors for all previous positions. The cache grows linearly in sequence
length and, at long context, exceeds the memory footprint of the weights.
Quantizing the cache to 2–4 bits per coordinate is attractive because it is
training-free and applies at inference time only.

The central difficulty is that keys and values play different roles. Values are
averaged by attention weights, so value quantization error enters the output
additively and is relatively forgiving. Keys enter through inner products inside a
softmax, so key quantization error perturbs the *attention distribution* itself.
The two therefore warrant different treatment.

### 2.2 TurboQuant

TurboQuant is a two-stage vector quantization scheme for this setting. Stage 1
applies a random orthogonal rotation and a per-coordinate Lloyd-Max scalar
quantizer, exploiting the fact that after rotation the coordinates of a
unit-norm vector concentrate around a distribution well approximated by
`N(0, 1/d)` for large head dimension `d`. Stage 2 adds a 1-bit correction — in the
original formulation a QJL-style random projection of the residual, of which only
the signs are stored — that makes the resulting inner-product estimate unbiased.

The property that matters here is the **distributional assumption**: the Lloyd-Max
codebook used in Stage 1 is optimal for a specific, approximately zero-mean
coordinate distribution. If the rotated coordinates do not follow that
distribution, the codebook is not merely suboptimal, it can be
non-discriminative. Our proposed mechanism is a violation of exactly this
assumption.

### 2.3 Key centering / mean removal

Subtracting a shared offset from keys before quantization is not a new idea; it
appears in the low-bit KV literature under headings such as key centering, offset
handling, or zero-point calibration, and is closely related to per-channel
(rather than per-token) key quantization, which addresses the fact that key
outliers in many models are channel-aligned.

The justification is a shift invariance of softmax. For a query `q` and keys
`k_i = μ + δ_i` sharing a common component `μ`:

```
q · k_i = q · μ + q · δ_i
```

and since `q · μ` is identical across `i` for a given query,

```
softmax_i(q · k_i) = softmax_i(q · δ_i)
```

The common component contributes nothing to the attention distribution. What is
new in this report is not the technique but the **magnitude and character of the
failure it prevents** in a specific pipeline, and the observed dependence on
KV-head count.

### 2.4 Qwen attention architecture

The Qwen2.5 instruct models used here employ grouped-query attention (GQA): many
query heads share a smaller number of key/value heads. The three models in our
sweep differ substantially in that sharing ratio — Qwen2.5-3B, 7B, and 14B are
recorded by our benchmark harness (which reads `num_key_value_heads` from the
loaded model config) as having 2, 4, and 8 KV heads respectively. All three use a
head dimension of 128.

A KV head that is shared by more query heads must serve a broader set of
consumers. It is plausible, though not established here, that such heads adopt a
more "generic" or anchor-like representation with a larger shared component. Our
data are consistent with that and are reported in Section 6.3, but three models is
not enough to establish a law.

### 2.5 Prior low-bit KV-cache methods

Relevant prior work includes: per-channel and outlier-aware KV quantization
(KVQuant); tuning-free asymmetric per-channel key / per-token value quantization
(KIVI); the QJL 1-bit residual sign transform; and rotation-based quantization
methods that use Hadamard or learned orthogonal transforms to make activation
distributions more quantization-friendly (QuaRot, SpinQuant). The observation that
transformer activations contain a small number of very-large-magnitude,
persistent, channel-aligned components (massive activations; attention sinks) is
the closest existing explanation for why a large shared key component would exist
at all.

Our contribution is orthogonal to these methods: it is a diagnosis of a specific
failure mode and a report of a strong, previously undocumented dependence on
KV-head count.

---

## 3. Failure Observation

### 3.1 Headline result

Qwen2.5-7B-Instruct, wikitext-2 test, 4,095 evaluated tokens, sliding window 512 /
stride 256, 3-bit keys:

| Configuration | Perplexity | Δ vs baseline |
|---|---:|---:|
| Uncompressed KV cache ("FP16 baseline") | 7.5225 | — |
| 3-bit baseline (no mean removal) | 9410.4876 | +9402.97 |
| 3-bit + mean-removal | 7.9029 | +0.38 |

Source: [`benchmarks/results/ppl_for_tom.json`](../benchmarks/results/ppl_for_tom.json)

**Note on the baseline label.** The row labelled "FP16 baseline" is an *FP16 KV
cache*, not an FP16 model. All model weights in every row, including the baseline,
are quantized to 4-bit NF4 via bitsandbytes. The comparison is therefore
apples-to-apples with respect to the KV cache, which is the variable under study,
but the absolute perplexities are those of a 4-bit-weight model. We retain the
original label for traceability to the raw artifact and flag the discrepancy here
rather than silently renaming it.

### 3.2 The failure is not bit-starvation

Adding a bit does not rescue the configuration:

| Model | Bits | No mean-removal | + mean-removal | Baseline |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 3 | 9410.4876 | 7.9029 | 7.5225 |
| Qwen2.5-7B | 4 | 1048.9915 | 7.7583 | 7.5225 |
| Qwen2.5-3B | 3 | 60.1999 | 11.0165 | 10.7177 |
| Qwen2.5-3B | 4 | 13.2231 | 10.8273 | 10.7177 |

At 4 bits the 7B model is still catastrophic (PPL 1,049). A failure that survives a
33% increase in quantizer resolution is not primarily a resolution problem. This
observation is the main empirical reason we favour a distributional-mismatch
explanation over a "not enough levels" explanation.

### 3.3 The failure is behavioural, not only statistical

Needle-in-a-haystack, Qwen2.5-7B-Instruct, 8,022-token context, needle
`"The secret code is PINEAPPLE-77."`:

| Needle position | Uncompressed KV | 3-bit, no mean-removal | 3-bit + mean-removal |
|---|---|---|---|
| 10% | PASS | FAIL | PASS |
| 50% | PASS | FAIL | PASS |
| 90% | PASS | FAIL | PASS |

Failing generations were not merely wrong; they were degenerate. Verbatim samples
from the raw artifact:

```
0000., .  numberWith); .0..0.. I.. The0 the, .,.
001 9111 2 mathematics 9 of mathematics 0. 1.  (1 10 mathematics of
1 0 0000000 strugg0 mathematics. 00.52 1.0.0, .
```

Source: [`benchmarks/results/niah_for_tom.json`](../benchmarks/results/niah_for_tom.json)

The character of the failure — repeated digits and punctuation, fragments of the
filler text, no retrieval — is what one expects if the attention distribution has
collapsed and the model is decoding from a nearly context-free state.

### 3.4 Experimental provenance

| Item | Value |
|---|---|
| Model revisions | `Qwen/Qwen2.5-7B-Instruct`, `Qwen/Qwen2.5-3B-Instruct` (HF hub, revision not pinned — see §9) |
| Weight quantization | bitsandbytes NF4, `bnb_4bit_compute_dtype=float16` |
| Dataset | wikitext-2-raw-v1, `test` split |
| PPL eval length | 4,095 tokens (`MAX_EVAL_TOKENS=4096`) |
| Window / stride | 512 / 256 |
| NIAH context | 8,022 tokens, 3 needle positions |
| Seed | 42 |
| Hardware | NVIDIA GeForce RTX 4090, 24 GB |
| Software (as resolved on the host today) | Python 3.13.12, PyTorch 2.11.0+cu130, transformers 5.5.0, bitsandbytes 0.49.2, datasets 4.8.4 |
| Run date | 2026-04-09 |
| Code commit (PPL script + report) | `8f54a28e4a0adf139542408efb0f5972eeed1ba0` |
| Code commit (PPL JSON + NIAH) | `8f18faef27252910c0bc70ddd87ce427e41ef107` |

---

## 4. Hypothesis

### 4.1 The invariance argument

For a query `q` and keys `k_i` decomposed as a shared component `μ` plus a
token-specific deviation `δ_i`:

```
q · (μ + δ_i) = q · μ + q · δ_i
```

Because `q · μ` is identical across keys for the same query:

```
softmax(qK^T + c) = softmax(qK^T)
```

The common component therefore does not carry the representational importance
inside the quantized key representation that its numerical magnitude might
suggest. All of the information that the attention distribution can respond to
lives in `δ_i`.

Note the direction of this argument. It does **not** say `μ` may be discarded — our
correction stores `μ` and adds it back exactly. It says that *error* in `μ` is
common-mode and cancels in the softmax, whereas error in `δ_i` does not. The
quantizer's budget should therefore be spent on `δ_i`.

### 4.2 Why this pipeline is unusually sensitive

The tested pipeline quantizes a key as follows (per head, per position):

1. optionally subtract the per-head running mean: `k̃ = k − μ_h`
2. **split off the norm**: `n = ‖k̃‖`, `u = k̃ / n`, storing `n` separately
3. rotate: `r = W u`, with `W` an orthonormal Walsh–Hadamard transform
4. per-coordinate Lloyd-Max scalar quantization of `r` at `b` bits
5. a residual-sign correction stage
6. dequantize, unrotate, rescale by `n`, and add `μ_h` back

Step 2 is decisive. Because the norm is stored separately, the quantizer's entire
budget is spent describing the key's **direction**. The Lloyd-Max codebook in step
4 is built for the coordinate distribution of a rotated *unit* vector — roughly
zero-mean, concentrated, approximately `N(0, 1/d)`.

Now suppose centering is off and `‖μ_h‖ ≫ E‖k − μ_h‖`. Then for every token

```
u = k / ‖k‖ ≈ μ̂_h + (small perturbation)
```

Every normalized key points in nearly the same direction. After the *fixed*
(deterministic, data-independent) WHT rotation, the coordinates of `r` are
dominated by the constant vector `W μ̂_h` with a small spread around it — not a
zero-mean concentrated distribution. The scalar quantizer, whose decision
boundaries were placed for the assumed distribution, maps most or all tokens into
the same cells. The reconstructed directions become nearly identical across
tokens, `δ_i` is annihilated, and the surviving logit structure is dominated by the
common term that softmax ignores.

**Hypothesis.** A large per-head key mean violates the zero-mean, concentrated
coordinate distribution that the Lloyd-Max codebook assumes. Under a pipeline that
normalizes keys and quantizes direction, this consumes the usable quantization
range on a component that is invisible to attention, and the resulting error in the
components that *do* affect relative attention logits is large enough to collapse
the attention distribution.

### 4.3 Predictions this hypothesis makes

The hypothesis is worth stating because it is falsifiable. It predicts:

- **P1.** Increasing bit-width should *not* rescue the failure, because the
  mismatch is distributional rather than resolution-limited. — **Consistent with
  observation** (§3.2: 4-bit on 7B is still PPL 1,049).
- **P2.** The failure should be attributable to the key path specifically, and
  should persist with values left uncompressed. — **Consistent with observation**
  (§6.2: the April-15 sweep compressed keys only and still failed).
- **P3.** Failure severity should scale with the ratio `‖μ_h‖ / E‖k − μ_h‖`
  measured on real activations. — **NOT YET TESTED.** This is the single most
  important outstanding experiment; see §7.
- **P4.** A data-independent rotation (WHT) should be more vulnerable than a random
  orthogonal rotation, since a random rotation spreads a fixed offset across
  coordinates unpredictably rather than into a fixed pattern. — **NOT YET TESTED.**
- **P5.** Models whose per-head key mean is small should not exhibit the failure.
  — **Consistent with observation but confounded** (§6.3: 14B, with 8 KV heads,
  does not fail — but we have not measured its key mean).

P3 is the load-bearing untested prediction. Until it is measured, the mechanism in
§4.2 remains a hypothesis that is *consistent with* the evidence rather than
demonstrated by it.

---

## 5. Experimental Method

### 5.1 The tested pipeline, stated precisely

The compressor under test is `GenerationCache` in this repository, a HuggingFace
`Cache`-protocol implementation. It is a TurboQuant-*family* method, and it differs
from canonical TurboQuant in ways that matter for interpreting the result:

| Aspect | Canonical TurboQuant | This implementation |
|---|---|---|
| Stage-1 rotation | random orthogonal (QR of Gaussian) | Walsh–Hadamard (deterministic) for power-of-two `d`; QR otherwise |
| Stage-2 correction | QJL: random projection, store signs | ResidualQuant: signs of the residual directly in rotated space |
| Norm handling | — | per-vector norm split off and stored; plus a norm-correction ratio |
| Values | MSE stage only | 3-bit (April-9 run) or uncompressed (April-15 run) |

Because of these differences, **no result in this paper should be read as a
statement about canonical TurboQuant.** The failure we document is a failure of the
configuration described above.

Fixed settings for all compressed rows: `fp16_window=0` (no uncompressed recent
window), `anchor_interval=0` (no FP16 anchor layers), `use_norm_correction=True`,
`use_residual_quant=True`, `seed=42`. The two mitigations that would mask the
failure — an FP16 recent window and FP16 anchor layers — were deliberately
disabled, so what is measured is pure compression.

### 5.2 Baseline

The baseline runs the identical model and identical evaluation loop with an
unmodified HuggingFace cache, i.e. FP16 keys and values. Weights are NF4 in both
arms. Perplexity is computed with a sliding window (context 512, stride 256), with
the overlap region masked to `-100` so no token is scored twice, and with a
**fresh cache constructed per window** so no compression state leaks across
windows.

### 5.3 Mean-removal intervention

When `center_before_quantize=True`, the layer maintains a running per-head mean
over the sequence dimension, shaped `(batch, heads, 1, head_dim)`:

```
mean_new = (mean_old · n_old + Σ_new) / (n_old + n_new)
```

Keys are centered against the current running mean before quantization; the mean
snapshot actually used for each chunk is retained so that dequantization restores
`k̂ = n·û + μ_chunk` consistently even though the running mean advances. The
intervention applies to **keys only** — values are never centered, which is
consistent with the invariance argument in §4.1, since averaging over values is not
shift-invariant.

### 5.4 Evaluation

- **Perplexity**: wikitext-2-raw-v1 test split, non-empty lines joined, truncated
  to 4,096 tokens, sliding window 512 / stride 256.
- **Needle-in-a-haystack**: 8,022-token synthetic context built from a repeated
  mathematics-history filler paragraph, a unique needle inserted at 10% / 50% /
  90% depth, exact-substring match on the generated answer.
- **Attention-level metrics**: cosine similarity of attention score vectors and
  top-1 / top-5 agreement against the uncompressed reference, on real cached KV
  tensors.

### 5.5 Reproducibility

Both headline experiments are single scripts with no arguments:

```bash
python benchmarks/ppl_for_tom.py     # Table in §3.1, §3.2
python benchmarks/niah_for_tom.py    # Table in §3.3
```

Both write their raw output to `benchmarks/results/`. The scripts hard-code an
`HF_HOME` of `/media/dhawal/Beast/cache/hub`; on a different host this must be
changed or the environment variable overridden.

---

## 6. Results

### 6.1 Primary result (April 9, 2026)

Full matrix, both models, both bit-widths, from
[`benchmarks/results/ppl_for_tom.json`](../benchmarks/results/ppl_for_tom.json):

**Qwen2.5-7B-Instruct** — baseline PPL 7.5225

| Config | PPL | Δ | Wall time |
|---|---:|---:|---:|
| Uncompressed KV | 7.5225 | — | 3.2 s |
| 3-bit | 9410.4876 | +9402.97 | 20.2 s |
| 3-bit + mean-removal | 7.9029 | +0.38 | 23.0 s |
| 4-bit | 1048.9915 | +1041.47 | 32.5 s |
| 4-bit + mean-removal | 7.7583 | +0.24 | 30.5 s |

**Qwen2.5-3B-Instruct** — baseline PPL 10.7177

| Config | PPL | Δ | Wall time |
|---|---:|---:|---:|
| Uncompressed KV | 10.7177 | — | 2.5 s |
| 3-bit | 60.1999 | +49.48 | 25.6 s |
| 3-bit + mean-removal | 11.0165 | +0.30 | 23.6 s |
| 4-bit | 13.2231 | +2.51 | 36.1 s |
| 4-bit + mean-removal | 10.8273 | +0.11 | 35.5 s |

Two features are worth noting. First, the effect size is wildly different between
the two models (9,403 vs. 49 PPL points of damage) despite identical head dimension
and identical code path. Second, mean removal lands both models within +0.11 to
+0.38 PPL of baseline at both bit-widths, i.e. the correction is not
model-specific even though the failure is.

### 6.2 Independent replication with values uncompressed (April 15, 2026)

A later, larger sweep re-ran the comparison with a different harness — it patches
`DynamicCache.update` to quantize **keys only, leaving values in FP16** — over
8,191 evaluated tokens (cap 8,192) and three model sizes:

| Model | KV heads | Baseline PPL | 3-bit, no mean-removal | 3-bit + mean-removal |
|---|---:|---:|---:|---:|
| Qwen2.5-3B | 2 | 11.4352 | 2339.7309 | 11.8743 |
| Qwen2.5-7B | 4 | 8.4293 | 13224.9449 | 9.0589 |
| Qwen2.5-14B | 8 | 4.9431 | **5.5409** | 5.5826 |

Sources: report
[`benchmarks/results/rotorquant_comprehensive.md`](../benchmarks/results/rotorquant_comprehensive.md);
raw per-model JSON in
[`rotorquant_comprehensive_20260415_0840.json`](../benchmarks/results/rotorquant_comprehensive_20260415_0840.json)
(3B, 7B) and
[`rotorquant_comprehensive_20260415_0907.json`](../benchmarks/results/rotorquant_comprehensive_20260415_0907.json)
(14B). Every figure in the table above was read back from those JSON files.

This is the most informative result in the paper, for three reasons.

1. **It isolates the failure to the key path.** Values were not compressed at all,
   and the collapse still occurred. The April-9 run compressed values at 3 bits, so
   on its own it could not exclude the value path.
2. **It replicates under different conditions** — different evaluation length
   (8,191 vs 4,095 tokens), different cache integration, different date. Absolute
   perplexities shift with evaluation length (7B baseline 8.4293 here vs 7.5225 in
   §6.1), as expected; the qualitative result does not.
3. **It falsifies the naive framing.** Qwen2.5-14B does not fail. A paper titled
   "…in Qwen Models" must confront this directly, and §8 does.

### 6.3 The KV-head correlation

Ordering the three models by KV-head count rather than parameter count:

| KV heads | Model | Damage without mean-removal (PPL ratio to baseline) |
|---:|---|---:|
| 2 | Qwen2.5-3B | ~205× |
| 4 | Qwen2.5-7B | ~1,569× |
| 8 | Qwen2.5-14B | ~1.12× (no failure) |

The failure is severe at 2 and 4 KV heads and absent at 8. It is not monotone in
KV-head count over these three points (7B is worse than 3B), so the honest summary
is a **threshold-like effect around low KV-head counts**, not a smooth trend. With
`n = 3` models, KV-head count is also perfectly confounded with parameter count and
with every other way Qwen scaled these three configurations. We flag this
correlation as the most promising lead for the next round of experiments, not as a
finding.

### 6.4 Robustness of the correction across prompt domains

An adversarial validation run on Qwen2.5-3B compared attention cosine similarity
with and without centering across five prompt types:

| Prompt domain | With centering | Without | Δ |
|---|---:|---:|---:|
| code | 0.9863 | 0.9756 | +0.0107 |
| math | 0.9800 | 0.9519 | +0.0281 |
| creative | 0.9895 | 0.9802 | +0.0093 |
| factual | 0.9893 | 0.9845 | +0.0048 |
| adversarial | 0.9845 | 0.9720 | +0.0125 |

Source: [`benchmarks/results/adversarial_validation.md`](../benchmarks/results/adversarial_validation.md)
(commit `517264696def516f5188e62c99a7ab3c3d49002d`)

Centering never hurt on this axis. Note the tension with §6.1, though: the
"without" column sits at 0.952–0.985 — that is the *catastrophic* configuration,
yet it looks nearly perfect by this metric. Attention-level cosine similarity is
evidently a poor proxy for downstream quality here.

We therefore treat §6.4 as weak supporting evidence only. It should be read
alongside **Limitation 9**, which documents three mutually inconsistent
attention-cosine measurements across this repository's runs. No conclusion in
this paper depends on any of them.

### 6.5 Third-party corroboration

An independent collaborator, working from a separate implementation of TurboQuant,
reported a +62.95 PPL degradation at 3 bits on Qwen2.5-3B-Instruct. Our
corresponding measurement is +49.48 (§6.1). We record this as informal
corroboration of the same-magnitude effect from code we did not write; it is not a
controlled replication, since neither the harness nor the evaluation protocol was
matched.

---

## 7. Ablations

Status of the ablation programme. Three of the six requested axes already have
data; three do not.

| # | Ablation | Status | Evidence |
|---|---|---|---|
| 1 | Mean removal ON/OFF | **Done** | §6.1, §6.2, §6.4 |
| 2 | Multiple bit widths (3, 4) | **Done** | §3.2, §6.1 |
| 3 | Multiple Qwen sizes (3B, 7B, 14B) | **Done** | §6.2 |
| 4 | At least one non-Qwen architecture | **Not done** | — |
| 5 | Different context lengths | **Partial** | 4,095 vs 8,192 tokens (§6.1 vs §6.2), but confounded with harness changes |
| 6 | Different evaluation samples / seeds | **Not done** | single seed (42), single dataset, single run per cell |

### 7.1 Highest-priority next experiment

**Measure the per-head key mean magnitude directly.** This tests prediction P3 and
is the difference between a hypothesis and a mechanism. The quantity to report,
per layer and per KV head, is the ratio

```
ρ_h = ‖μ_h‖ / E_t‖k_{h,t} − μ_h‖
```

on real activations for Qwen2.5-3B, 7B, and 14B over a fixed wikitext-2 passage.
The hypothesis predicts `ρ_h` is large for 3B and 7B and small for 14B. If `ρ_h` is
similar across all three, the mechanism in §4.2 is wrong and the KV-head
correlation has some other cause.

A ready-to-run script is provided at
[`paper/experiments/measure_key_mean.py`](experiments/measure_key_mean.py).
It requires the Qwen2.5 checkpoints, which are **not** currently present in the
local HF cache and must be re-downloaded.

### 7.2 Remaining programme, in priority order

1. **P3 measurement** (above) — turns the hypothesis into a mechanism.
2. **Non-Qwen control** — Llama-3.1-8B (8 KV heads) and Mistral-7B (8 KV heads).
   The hypothesis predicts neither fails. A non-Qwen model with ≤4 KV heads would
   be a much stronger test, and finding one should be treated as part of the task.
3. **Rotation ablation (P4)** — WHT vs. random-orthogonal (QR) rotation with
   centering off. If a random rotation substantially mitigates the failure, the
   deterministic-rotation component of the mechanism is confirmed.
4. **Seeds and samples** — ≥3 seeds and ≥3 disjoint wikitext-2 passages per cell,
   reporting variance. Currently every number in this paper is `n = 1`.
5. **Context length, decoupled from harness** — 512 / 2K / 8K / 32K on a single
   fixed harness.
6. **Synthetic control** — inject a synthetic offset of controlled magnitude into
   the keys of a model that does *not* fail, and verify that failure appears at the
   predicted `ρ_h`. This is the cleanest causal test available and does not depend
   on finding a naturally-occurring low-KV-head non-Qwen model.

---

## 8. Discussion

### 8.1 What the experiments demonstrate

- Under the configuration in §5.1, 3-bit key quantization of Qwen2.5-7B-Instruct
  produces perplexity of ~9,410 against a ~7.52 uncompressed-KV baseline, and 0/3
  needle recall at 8K context. This is reproducible from committed code and raw
  artifacts.
- Per-head key mean removal with exact restoration recovers perplexity to ~7.90
  and 3/3 needle recall, under the same configuration.
- The failure persists when values are left entirely uncompressed, so it is a
  **key**-quantization failure.
- The failure persists at 4 bits, so it is not simply insufficient resolution.
- The correction helps at both 3 and 4 bits, on both 3B and 7B, and across five
  prompt domains at the attention level.

### 8.2 What remains a hypothesis

The **mechanism** is a hypothesis. We have not measured the per-head key mean on
any of these models. Every statement in §4.2 about `‖μ_h‖` being large is an
inference from the effectiveness of the correction, which is weak evidence:
mean removal changes the input distribution to the quantizer in several ways at
once, and "the intervention that removes the mean helps" does not by itself
establish "the mean was the problem."

The **KV-head dependence** is an observed correlation over three models in a single
family, with parameter count fully confounded. It is a lead, not a result.

### 8.3 What may generalize

If the mechanism is confirmed, the generalizable statement is not about Qwen. It is
about a *class* of quantizers: any KV compressor that (a) normalizes keys and
spends its budget on direction, and (b) uses a fixed codebook designed for a
zero-mean coordinate distribution, is vulnerable to any model whose keys carry a
large shared per-head component. The correct engineering response is to make
centering unconditional for such quantizers, since it is cheap and, in our data,
never harmful for the WHT pipeline.

There is a corollary for benchmarking practice. A method evaluated only on models
with 8+ KV heads — which describes much of the Llama-centric low-bit KV literature
— would not have surfaced this failure at all. Architecture coverage in KV-cache
quantization benchmarks appears to be systematically too narrow.

### 8.4 What cannot yet be claimed

We cannot claim that:

- canonical TurboQuant fails on Qwen (we tested a variant, §5.1);
- Qwen models in general are vulnerable (14B is not, §6.2);
- KV-head count is causal (confounded, `n = 3`);
- the per-head key mean is large in these models (unmeasured);
- the effect holds outside wikitext-2, seed 42, or these context lengths.

### 8.5 A note on metric choice

§6.4 shows attention cosine similarity above 0.97 in configurations whose
end-to-end perplexity is catastrophic. Whatever the resolution of the measurement
inconsistency noted in §9, the practical lesson stands: **KV-cache compression
methods should not be validated on attention-level reconstruction metrics alone.**
A method can preserve attention-score cosine similarity and still destroy the
relative logit structure that generation depends on.

---

## 9. Limitations

1. **Not canonical TurboQuant.** The pipeline uses a deterministic Walsh–Hadamard
   rotation instead of a random orthogonal one, and direct residual signs instead
   of QJL. Conclusions do not transfer to the original algorithm without retesting.
2. **Model weights are 4-bit NF4, not FP16, in every arm** including the baseline.
   Weight quantization error may interact with KV quantization error. An FP16-weight
   replication is required before any absolute perplexity is quoted externally.
3. **No non-Qwen model tested** with this ablation. The title's scope is not yet
   earned; on current evidence the claim is about low-KV-head Qwen2.5 instruct
   models specifically.
4. **The failure is not uniform within Qwen.** Qwen2.5-14B does not exhibit it.
5. **`n = 1` everywhere.** One seed, one dataset, one passage, one run per cell. No
   variance estimates. The effect sizes are large enough that run-to-run noise is
   unlikely to explain the 9,410 vs 7.90 contrast, but this is an argument, not a
   measurement.
6. **Model revisions not pinned.** The scripts request HF model names without a
   revision hash. Upstream re-uploads would not be detected.
7. **Environment not pinned to the run.** The software versions in §3.4 are those
   resolved on the host today; they match the versions recorded in a contemporaneous
   2026-04-04 results file in this repository, but the April-9 run itself did not
   emit a lockfile.
8. **Storage overhead of the correction is not accounted for.** Information
   -theoretically the stored mean is `d` values per head. As implemented, the
   per-chunk mean is materialized to full sequence length
   (`chunk_mean = mean.expand(...).clone()`) and appended to a per-chunk list, so
   the mean-removal arm carries an FP16 tensor the size of the uncompressed key
   tensor. **The perplexity comparisons in this paper are therefore quality
   comparisons at equal quantizer bit-width, not at equal memory.** No compression
   ratio should be quoted from these runs. Fixing the storage to `O(d)` per head is
   straightforward and is a prerequisite for any memory claim.
9. **Unresolved measurement inconsistency in the attention-cosine metric.** Three
   runs in this repository report mutually irreconcilable attention cosine
   similarities for nominally comparable 3-bit configurations:

   | Source | Model | No mean-removal | + mean-removal |
   |---|---|---:|---:|
   | April-4 integration | 3B | 0.6642 | 0.9939 |
   | April-9 adversarial validation | 3B | 0.9519–0.9845 | 0.9800–0.9895 |
   | April-15 sweep | 7B | 0.1290 | 0.0888 |

   In the April-15 sweep the reported cosine is ≈0.09–0.13 for *every* method,
   including `PolarQuant-WHT+Mean`, whose perplexity (9.06) is the best in that
   table — while its `vec_cos` for the same method is 0.867 and for plain
   `PolarQuant-WHT` is 0.984. At least one of these metric paths is computing
   something other than what its name implies. **No claim in this paper rests on
   any attention-cosine number**; §6.4 is reported as weak supporting evidence
   only, and the April-15 cosine column is excluded from the results entirely. The
   discrepancy must be resolved before any attention-level metric from this
   repository is published anywhere.
10. **Context length is confounded with harness.** The 4,095- and 8,192-token
    results also differ in cache integration and in whether values were compressed,
    so §6.2 is a replication under changed conditions rather than a context-length
    ablation.

---

## 10. Reproducibility

**Repository:** https://github.com/dhawalc/turboQuantDC (public)

**Immutable commit permalinks for the primary artifacts:**

| Artifact | Permalink |
|---|---|
| PPL script | https://github.com/dhawalc/turboQuantDC/blob/8f54a28e4a0adf139542408efb0f5972eeed1ba0/benchmarks/ppl_for_tom.py |
| PPL report (markdown) | https://github.com/dhawalc/turboQuantDC/blob/8f54a28e4a0adf139542408efb0f5972eeed1ba0/benchmarks/results/ppl_for_tom.md |
| PPL raw results (JSON) | https://github.com/dhawalc/turboQuantDC/blob/8f18faef27252910c0bc70ddd87ce427e41ef107/benchmarks/results/ppl_for_tom.json |
| NIAH script | https://github.com/dhawalc/turboQuantDC/blob/8f18faef27252910c0bc70ddd87ce427e41ef107/benchmarks/niah_for_tom.py |
| NIAH raw results (JSON) | https://github.com/dhawalc/turboQuantDC/blob/8f18faef27252910c0bc70ddd87ce427e41ef107/benchmarks/results/niah_for_tom.json |
| Adversarial validation | https://github.com/dhawalc/turboQuantDC/blob/517264696def516f5188e62c99a7ab3c3d49002d/benchmarks/results/adversarial_validation.md |

The following artifacts were previously excluded from version control by a
`results/` ignore rule in `.gitignore`. They are force-added in the **same commit
as this manuscript**, so that every number cited here resolves to a tracked file;
their permalink is therefore the commit that introduces this document.

| Artifact | Cited in |
|---|---|
| `benchmarks/results/rotorquant_comprehensive.md` | §6.2, §6.3 |
| `benchmarks/results/rotorquant_comprehensive_20260415_0835.json` (3B) | §6.2 |
| `benchmarks/results/rotorquant_comprehensive_20260415_0840.json` (3B, 7B) | §6.2 |
| `benchmarks/results/rotorquant_comprehensive_20260415_0907.json` (14B) | §6.2 |
| `benchmarks/results/mean_removal_integration_results.json` | §9 (Limitation 9) |

Every figure in the §6.2 table was independently read back out of the raw JSON
files, not transcribed from the summary markdown.

**Outstanding-experiment tooling.** The measurement that would convert §4's
hypothesis into a mechanism is implemented and unit-checked against analytically
known inputs, but has not been run against real models (the Qwen2.5 checkpoints
are not currently in the local HF cache):

```bash
python paper/experiments/measure_key_mean.py                    # all three sizes
python paper/experiments/measure_key_mean.py --models 7B        # single model
```

**Implementation entry points:**

| Component | Location |
|---|---|
| Cache object | `turboquantdc/generation_core.py` — `GenerationCache` |
| Centering + quantization | `turboquantdc/generation_layers.py` — `_CompressedLayer.update`, `_quantize_vectors_python` |
| Residual-sign estimator | `turboquantdc/residual_quant.py` |

**Environment:**

```
GPU        NVIDIA GeForce RTX 4090, 24 GB
Python     3.13.12
PyTorch    2.11.0+cu130
transformers   5.5.0
bitsandbytes   0.49.2
datasets       4.8.4
```

**To reproduce:**

```bash
git clone https://github.com/dhawalc/turboQuantDC && cd turboQuantDC
pip install torch transformers accelerate bitsandbytes datasets scipy
# Edit HF_CACHE_DIR in the scripts, or export HF_HOME, before running.
python benchmarks/ppl_for_tom.py
python benchmarks/niah_for_tom.py
```

Expected wall time is a few minutes per model on a 24 GB GPU once weights are
cached; the checkpoints themselves are roughly 15 GB and 6 GB for 7B and 3B.

---

## References

> All entries require verification against the published record before external
> circulation. `[verify]` marks entries whose venue, year, or identifier could not
> be confirmed offline.

1. Zandieh, A., et al. *TurboQuant: Online Vector Quantization with Near-optimal
   Distortion Rate.* arXiv:2504.19874, 2025. `[verify — venue listed elsewhere in
   this repository as ICLR 2026]`
2. Zandieh, A., Daliri, M., Han, I. *QJL: 1-Bit Quantized JL Transform for KV Cache
   Quantization with Zero Overhead.* arXiv:2406.03482, 2024. `[verify]`
3. Liu, Z., et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.*
   ICML, 2024. arXiv:2402.02750. `[verify]`
4. Hooper, C., et al. *KVQuant: Towards 10 Million Context Length LLM Inference with
   KV Cache Quantization.* NeurIPS, 2024. arXiv:2401.18079. `[verify]`
5. Ashkboos, S., et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.*
   arXiv:2404.00456, 2024. `[verify]`
6. Liu, Z., et al. *SpinQuant: LLM Quantization with Learned Rotations.*
   arXiv:2405.16406, 2024. `[verify]`
7. Sun, M., Chen, X., Kolter, J. Z., Liu, Z. *Massive Activations in Large Language
   Models.* arXiv:2402.17762, 2024. `[verify]`
8. Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. *Efficient Streaming Language
   Models with Attention Sinks.* ICLR, 2024. arXiv:2309.17453. `[verify]`
9. Ainslie, J., et al. *GQA: Training Generalized Multi-Query Transformer Models
   from Multi-Head Checkpoints.* EMNLP, 2023. arXiv:2305.13245. `[verify]`
10. Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115, 2024. `[verify]`
11. Merity, S., Xiong, C., Bradbury, J., Socher, R. *Pointer Sentinel Mixture
    Models.* arXiv:1609.07843, 2016. `[verify]`
12. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. *QLoRA: Efficient
    Finetuning of Quantized LLMs.* NeurIPS, 2023. arXiv:2305.14314. `[verify]`
13. Lloyd, S. P. *Least Squares Quantization in PCM.* IEEE Transactions on
    Information Theory, 28(2):129–137, 1982. `[verify]`
14. Max, J. *Quantizing for Minimum Distortion.* IRE Transactions on Information
    Theory, 6(1):7–12, 1960. `[verify]`
