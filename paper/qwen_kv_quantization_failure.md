# What Determines KV-Cache Quantization Damage: A Diagnosis in Qwen and a Predictive Law

**Author:** Dhawal Chheda
**Draft date:** 2026-08-17 (revised 2026-08-18)
**Status:** Working manuscript. Not submitted.

The paper has two halves, and the second grew out of the first. §§3–6.12
diagnose a catastrophic low-bit KV-cache failure in Qwen2/Qwen2.5 — mechanism,
correction, causal test, and its birth and disappearance across five model
generations. §§6.13–6.22 pursue the question that diagnosis raised: *what
actually determines how much a compressed KV cache costs?* — establishing that
the reconstruction metrics the field uses do not predict damage, that our own
replacement is better but insufficient, and finally that damage is a
model-specific function of one cheap scalar which can be calibrated without
running the quantizer at all (§6.22, the paper's strongest result).

Sections 3–6.5 report previously-run committed experiments; §§6.6–6.22 report
measurements made on 2026-08-18, including several that refute earlier claims
of this same manuscript. Section 7 lists the ablations still
outstanding, and §9 the limitations, including hypotheses this project formed
and then refuted with its own data.
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

Direct measurement on the failing model's activations confirms the mechanism's
central prediction: **53% of the average key's squared magnitude is the shared
per-head mean**, and the mean cosine between a key and that mean is 0.715. Running
the compressor on those real keys reproduces the failure at the level that matters:
at layer 0, reconstructed keys retain a vector cosine similarity of 0.9950 both
with and without centering, while the correlation of the *attention logits* they
produce is 0.5488 without centering and 0.9934 with it. A natural sharper
hypothesis — that Qwen2.5's learned key-projection bias, which no newer model we
scanned retains, is the source of the shared component — was tested and refuted:
zeroing it removes the extreme tail of mean-dominated heads but leaves the bulk
shared component unchanged.

Measuring three generations under one protocol shows the failure is being
engineered out over time. Qwen3-8B, bias-free with QK-Norm, still carries the same
bulk shared component as Qwen2.5-7B (51.0% of key energy against 52.7%) with a
much smaller tail (worst-head ratio 7.0 against 59.1). Qwen3.5-4B — which declares
the same model class, KV-head count and head dimension as the newly released
Qwen3.8-27B — is qualitatively different: 29.5% shared energy, **no head anywhere
in the model whose mean exceeds its deviation**, no first-layer outlier, and
uncentered logit correlation of 0.9905 against Qwen2.5-7B's 0.9619. Centering
still helps there but is no longer the difference between working and failing. The
severe failure documented in this paper is therefore specific to the older
generation, while the diagnostic method and the metric warning generalise.

Critically, the failure is **not uniform across the Qwen family**, and it is not
what it first appeared. Within Qwen2.5 the severity tracked KV-head count, and we
initially reported that. A 14-model atlas refutes it: Qwen3.5-0.8B has 2 KV heads,
the same as the Qwen2.5 models that degrade by three orders of magnitude, and is
completely unaffected; Gemma 2, Gemma 3 and Falcon 3 all share Qwen2.5-7B's 4 KV
heads and are likewise unaffected. Every catastrophic configuration we found, at
any bit-width, belongs to Qwen2.5. Qwen3 is intermediate and Qwen3.5 — the
architecture of the newly released Qwen3.8 — is immune.

We then ran the compressor end-to-end across 150 configurations spanning 26
models and thirteen architecture lineages, recording true perplexity alongside
the cheap reconstruction metrics this field validates compressors on.
Per-vector cosine similarity is *uncorrelated* with real damage (Pearson
−0.003 over 176 cells): a quarter of broken configurations pass a 0.995
criterion, including ones at ×376 and ×2,929 the baseline perplexity, while
configurations costing 7% are rejected. The worst-layer correlation of the
attention logits the reconstructed keys produce is strictly better — it catches
every catastrophic cell and transfers to unseen architectures — but it is not
sufficient either, and we show why by manufacturing failures that no
reconstruction-side statistic can see.

That failure has a structural cause and a remedy. Damage factorizes into the
score-space noise a compressor injects and the model's sensitivity to it;
every reconstruction metric measures only the first factor. Measuring the
second directly — perturbing cached keys with mean-free isotropic Gaussian
noise at several magnitudes, no quantizer involved, five to eleven forward
passes per model — yields a per-model **noise-response curve**. Evaluating a
model's own curve at a quantizer configuration's measured score-space noise
predicts that configuration's perplexity damage with **R² = 0.969 across 96
configurations and 13 models, over a damage range from ×0.99 to ×1,348**, with
median error ×1.01 and **zero dangerous misses at a ×5 catastrophe gate**. A
held-out test at bit-widths absent from the study (5, 6 and 8 bits) predicts
within ×1.5. The curve is calibrated on unstructured Gaussian noise and tested
on structured quantizer error, so the amount of score-space noise — not its
structure — is what determines damage. This explains why no universal proxy
threshold can exist (the map from proxy to damage is a model property), and it
converts compressor certification into a cheap procedure. The instrument is
adapted from weight-quantization work that calibrates an error-to-perplexity
coefficient by noise insertion (the Linearity Theorem / HIGGS); what is new
here is the score-space coordinate, the KV-cache setting, and validity in the
catastrophic regime where that work's quadratic error model is stated not to
hold. Its limits are
stated: it cannot resolve a 2% tax from an 8% one, one-bit quantization is
systematically under-predicted, and the curve is not predictable from the
key-mean structure, so both factors must be measured.

We also quantify the correction's value: 2-bit centered beats 6-bit
uncentered, so mean removal is worth roughly four bits of precision.

This paper documents the failure mode, experimental methodology, proposed
mechanism, correction, metric validation, and limitations, and provides
reproducible artifacts for independent evaluation.

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
   recovers to within +0.38 PPL of the uncompressed-KV baseline;
5. initially report a correlation between failure severity and KV-head count —
   then **refute it ourselves** with a 26-model atlas (§6.14) and locate the
   pathology's true boundary instead: it was born with the Qwen2 generation and
   engineered out by Qwen3.5, with the mechanism's quantity ρ tripling exactly
   at the discontinuity while shared key *energy* stays constant (§6.20);
6. *(added 2026-08-18)* show across 150+ configurations and 26 models that
   **per-vector reconstruction metrics do not predict KV-compression damage**,
   validate a strictly better cheap statistic — worst-layer attention-logit
   correlation, which catches every catastrophic cell and transfers to unseen
   architectures (§6.15) — and then map its scope boundary by manufacturing
   failures that no reconstruction-side proxy can see, explaining *why*
   structurally: such proxies measure the noise a quantizer injects, never the
   model's sensitivity to it (§6.19);
7. quantify the correction's worth in bits: 2-bit centered outperforms 6-bit
   uncentered, i.e. centering substitutes for roughly four bits of precision
   (§6.16);
8. *(added 2026-08-18)* establish the mechanism **causally**: injecting a
   synthetic shared key component into an immune model reproduces the entire
   failure dose-dependently, and centering neutralizes every dose (§6.18); and
9. *(added 2026-08-18)* **establish a damage law**: quantization damage is a
   model-specific function of one cheap scalar, calibrated by injecting
   Gaussian noise into cached keys with no quantizer involved, predicting
   96 configurations across 13 models with R² 0.969 over a ×1,348 damage
   range and zero dangerous misses at a catastrophe gate — which explains
   why contributions 6's proxy could not have sufficed alone, and yields a
   cheap certification procedure with explicitly measured limits (§6.22); and
10. *(added 2026-08-18)* document independent concurrent observations of the
   same failure, family-specificity, and metric blindness by unrelated parties
   in different codebases (§2.6), and position the work against prior art found
   in an online search: centering-before-VQ exists (NSNQuant, classical
   mean-removed VQ), score-space evaluation exists in embryo (KIVI, the
   withdrawn HeadQ) — the diagnosis, causal test, bit-quantification, and
   paired falsification at scale are what this paper adds.

We deliberately restrict the claim. We do not claim that canonical TurboQuant fails
on Qwen; we claim that *this* TurboQuant-family configuration does, and we describe
the configuration precisely enough to be checked.

---

## 2. Background and Related Work

> **Reference-verification note.** Every arXiv identifier in the References
> section was checked against the arXiv API on 2026-08-18 and resolves with
> matching title and first author. `[verify]` now marks only what that check
> cannot confirm — conference-venue attributions and the print-era references.
> See the note at the head of the References section.

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
failure it prevents** in a specific pipeline, and the diagnosis of *which models*
need it and why.

Concrete precedents, verified online 2026-08-18: mean-removed vector quantization
is classical (Gersho & Gray's textbook treats it as a standard VQ variant), and
**NSNQuant** (arXiv:2505.18231, May 2025) applies channel-wise centering ("Shift")
between two token-wise normalizations followed by a Hadamard transform,
specifically so that KV vectors match a standard-normal codebook without
calibration. NSNQuant is prior art for centering-before-VQ as a technique. What it
does not contain is this paper's diagnostic content: it does not identify
mean-dominance as a family-specific pathology, does not measure ρ_h on any model,
and does not report that the intervention is the difference between ×1,416 and
×1.03 on one family while being nearly free on others.

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
(KIVI); the QJL 1-bit residual sign transform; rotation-based quantization
methods that use Hadamard or learned orthogonal transforms to make activation
distributions more quantization-friendly (QuaRot, SpinQuant, RotateKV); and
layer-wise sensitivity-aware bit allocation (KVTuner). The observation that
transformer activations contain a small number of very-large-magnitude,
persistent, channel-aligned components (massive activations; attention sinks) is
the closest existing explanation for why a large shared key component would exist
at all.

**Score-space evaluation is also not new as a concept.** KIVI already reported
"attention score error" alongside reconstruction error when comparing quantization
axes; HeadQ (arXiv:2605.03562, since withdrawn by its author) proposed
Fisher/score-space error explicitly because it "predicts attention KL far better
than raw key MSE"; and a June-2026 study of alignment under KV quantization
(arXiv:2606.09864) reports model-specific failures "invisible to standard
metrics". What we add to this thread is not the idea that score space is the right
place to look, but the **paired, end-to-end falsification at scale**: 100+
(model, bit-width, centering) cells in which the same forward pass records both
the proxy metrics and the true perplexity, showing that the cosine criterion this
project itself started with certifies configurations that raise perplexity 376×
and rejects configurations that cost 7%, while a worst-layer logit-correlation
threshold transfers across families — together with a measured scope boundary
where every tested proxy fails (§6.19).

Our contribution relative to these methods is therefore diagnostic rather than
algorithmic: a mechanism for *why* one family collapses (measured, then verified
causally by injection), a quantification of what the classical fix is worth (about
four bits), and a validated-and-bounded cheap predictor of real damage.

### 2.6 Independent observations of the same phenomenon

*(added 2026-08-18 after an online prior-art search; none of the following
parties has any connection to this project)*

The core observations of this paper have been independently reproduced in public,
with different quantizers, before we found them ourselves or concurrently with us:

- **llama.cpp issue #21385** (user SCJedi, 2026-04-03) reports q4_0 KV cache
  "completely lossless" on hybrid-attention Qwen3.5 (BLEU 1.000) — consistent
  with our finding that the Qwen3.5 generation is immune — and proposes per-head
  entropy-adaptive bit allocation for standard models.
- **In the same thread** (user jagmarques, 2026-05-05): on Qwen2.5-7B, 3-bit keys
  / 2-bit values with first- and last-layer FP16 protection gives +0.84% PPL, but
  *removing the layer protection blows perplexity from 6.12 to ~3,300* (×540) —
  an independent measurement of the same catastrophic key-quantization failure,
  in a different codebase, with a different quantizer. The same commenter reports
  Mistral-7B at +0.31% under the identical protection-off setting — an independent
  observation of the family-specificity. Notably, their choice of *which* layers
  to protect — first and last — is exactly what our per-layer profile predicts:
  on Qwen2.5-7B at 3 bits the two worst layers by attention-logit correlation
  are layer 0 (0.54) and layer 27 (0.80), with every layer in between above 0.91
  (next-worst is layer 1 at 0.92; Figure 1). Two codebases, two quantizers, the same two layers.
- **In the same thread** (user sztlink, 2026-05-06): a KLD-based check of the
  q4_0 claim scores "close" (98.81) while a trajectory-preservation harness rates
  the same configuration degraded — an independent sighting of the central metric
  theme of this paper, that fidelity measures of different granularity disagree
  about KV-cache damage.
- **AXELRAM** (arXiv:2604.02638, April 2026) reports "catastrophic spikes" on
  Qwen2.5-3B under sign-pattern perturbations of the KV cache while LLaMA-3.1-8B
  is "fully stable" under the same treatment, correlating the fragility with
  layer-wise norm heterogeneity.

We record these because convergent evidence from independent implementations is
worth more than any additional experiment we could run ourselves. Permalink:
[ggml-org/llama.cpp#21385](https://github.com/ggml-org/llama.cpp/issues/21385).

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

- **P1.** *(status revised 2026-08-18 — see §6.16.)* Increasing bit-width should *not* rescue the failure, because the
  mismatch is distributional rather than resolution-limited. — **Consistent with
  observation** (§3.2: 4-bit on 7B is still PPL 1,049).
- **P2.** The failure should be attributable to the key path specifically, and
  should persist with values left uncompressed. — **Consistent with observation**
  (§6.2: the April-15 sweep compressed keys only and still failed).
- **P3.** The ratio `‖μ_h‖ / E‖k − μ_h‖` should be large on a model that fails.
  — **CONFIRMED** (§6.7): on Qwen2.5-7B, 53% of the average key's energy is the
  shared per-head mean, and the mean cosine of a key to that mean is 0.715.
- **P4.** A data-independent rotation (WHT) should be more vulnerable than a random
  orthogonal rotation, since a random rotation spreads a fixed offset across
  coordinates unpredictably rather than into a fixed pattern. — **NOT YET TESTED.**
- **P5.** Models whose per-head key mean is small should not exhibit the failure.
  — **Consistent with observation but confounded** (§6.3: 14B, with 8 KV heads,
  does not fail — but we have not measured its key mean).
- **P6.** *(added 2026-08-18)* The damage should appear in relative attention
  logits while vector-level reconstruction metrics still look healthy.
  — **CONFIRMED** (§6.9): at layer 0, vector cosine is 0.9950 both with and
  without centering, while the attention-logit correlation is 0.5488 vs 0.9934.

As of 2026-08-18 the mechanism is no longer purely inferential: P3 and P6 are
measured directly (§6.7, §6.9). The *generality* concern — `ρ_h` originally
measured on exactly one model, and only on a model that fails — is resolved by
the end of the campaign: ρ is now measured on seven models spanning both sides
of the failure boundary (§6.12, §6.20), and it separates them cleanly
(mean ρ ≈ 1.0 on the immune Qwen1.5 against 2.2–2.8 on the catastrophic
Qwen2/2.5, with shared *energy* roughly constant at ~50% across all of them).
A natural sharper hypothesis, that Qwen2.5's `k_proj` bias is the source of
the shared component, was tested and **refuted** (§6.8), and re-refuted from
the other direction by the lineage (Qwen1.5 carries the same bias and is
immune, §6.20).

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

### 6.6 Architectural scan: which models carry a shared key component by construction

*(added 2026-08-18)*

We scanned every language model available locally as a GGUF blob, reading the
tensor index and metadata directly. Vision-tower tensors (`v.blk.*`) are excluded:
ViTs conventionally carry a QKV bias, and counting it would be a false positive
for the language model.

| Model | Arch | Attn layers | KV heads | head_dim | `k_proj` bias | QK-Norm | mean ‖b_h‖ | max ‖b_h‖ |
|---|---|---:|---:|---:|:--:|:--:|---:|---:|
| qwen2.5:7b | qwen2 | 28/28 | 4 | 128 | **yes** | no | 31.98 | **920.34** |
| qwen2.5-coder:7b | qwen2 | 28/28 | 4 | 128 | **yes** | no | 36.04 | **998.71** |
| qwen3:14b | qwen3 | 40/40 | 8 | 128 | no | yes | – | – |
| qwen3.6:27b | qwen35 | 16/64 | 4 | 256 | no | yes | – | – |
| gemma3:12b | gemma3 | 48/48 | 8 | 256 | no | yes | – | – |
| gemma3:27b | gemma3 | 62/62 | 16 | 128 | no | yes | – | – |
| gemma4:latest | gemma4 | 42/42 | 2 | 512 | no | yes | – | – |

Source: [`paper/experiments/results/architecture_scan.json`](experiments/results/architecture_scan.json)

Qwen2.5 is the **only** family in this set that adds a learned bias to the key
projection. A `k_proj` bias is, by construction, a component shared by every key
at every position before RoPE, and its per-head norm reaches 920–999. Qwen3
removed the QKV bias that Qwen2 used and introduced QK-Norm in its place
(RMSNorm applied to the query and key head vectors), and every newer architecture
here — Qwen3, Qwen3.5/3.6, Gemma 3, Gemma 4 — follows the same pattern.

Two entries deserve note. **gemma4 has only 2 KV heads** — fewer than the
Qwen2.5-7B that failed — yet carries no key bias; it is therefore the single best
available test for separating the KV-head-count correlation (§6.3) from the
shared-key-component mechanism (§4.2). **qwen3.6:27b is a hybrid stack**: only 16
of its 64 layers are attention layers (4 KV heads, head_dim 256); the rest use a
non-attention token mixer and hold no KV cache.

### 6.7 Direct measurement of the per-head key mean — prediction P3

*(added 2026-08-18)*

We measured `ρ_h = ‖μ_h‖ / E_t‖k_t − μ_h‖` on **real post-RoPE keys** read out of
the KV cache of Qwen2.5-7B-Instruct — the exact model of §3.1 — loaded from the
local Q4_K_M GGUF blob, over 1,024 tokens:

| Statistic | Value |
|---|---:|
| ρ mean | 2.456 |
| ρ median | 0.978 |
| ρ p90 | 2.113 |
| ρ max | 59.116 |
| **mean cos(k_t, μ_h)** | **0.7150** |
| **shared-mean energy fraction ‖μ_h‖²/E‖k_t‖²** | **0.5267** |
| heads with ρ > 1 | 43.8% |
| heads with ρ > 3 | 8.9% |

Source: [`paper/experiments/results/key_mean_rho.json`](experiments/results/key_mean_rho.json)

**P3 is confirmed.** On the model that fails, an average key makes a 44° angle
with its own head's mean, and **53% of the average key's squared magnitude is a
component that softmax cannot see**. The distribution is heavy-tailed: the median
head is near ρ ≈ 1 while the worst is ρ ≈ 59. This is the measurement §4.2's
mechanism needed and had never had.

### 6.8 Causal attribution: the bias is *not* the main cause

*(added 2026-08-18)*

§6.6 makes an obvious hypothesis available — that Qwen2.5's `k_proj` bias *is* the
shared component. We tested it directly by loading the model once and measuring
both arms, identical in every other respect:

| Statistic | A: as shipped | B: `k_proj.bias := 0` |
|---|---:|---:|
| ρ mean | 2.456 | 1.117 |
| ρ median | 0.978 | 1.148 |
| ρ max | **59.116** | **1.742** |
| heads with ρ > 3 | **8.9%** | **0.0%** |
| mean cos(k_t, μ_h) | 0.7150 | 0.7244 |
| shared-mean energy fraction | 0.5267 | 0.5351 |

Source: [`paper/experiments/results/bias_ablation.json`](experiments/results/bias_ablation.json)

**The obvious hypothesis is wrong, and we report it as such.** Removing the bias
eliminates the extreme tail entirely — the worst head falls from ρ ≈ 59 to ρ ≈ 1.7,
and no head remains above ρ = 3 — but it leaves the *bulk* shared component
completely untouched: mean cosine to the head mean and shared-energy fraction both
move by under 0.01, in the wrong direction.

The correct statement is therefore twofold:

1. The large shared key component in Qwen2.5-7B is **not** produced by the key
   bias. It is produced by `W_k · E[x]` — the projection of the residual stream's
   own large, persistent mean, which connects this failure to the massive-activation
   and attention-sink literature (§2.5) rather than to a Qwen-specific weight.
2. The key bias **does** create a small population of extremely mean-dominated
   heads (ρ up to 59) that exist in no other model we scanned.

This weakens any claim that newer bias-free architectures are automatically safe.
They lack the extreme tail; whether they also lack the bulk shared component is
**unmeasured** (§9, Limitation 12).

### 6.9 Closing the loop: what the shared component does to the quantizer

*(added 2026-08-18)*

The measurements above establish that the shared component exists. This experiment
establishes that it is what breaks the quantizer, using the repository's **own
production 3-bit path** (`_CompressedLayer`, WHT rotation + ResidualQuant) applied
to the real Qwen2.5-7B keys from §6.7, with centering on and off.

Because softmax responds only to the spread of `u·k_i` across positions for
whatever query direction `u` arises, we compare true and reconstructed logit
vectors over 256 random unit probe directions, after removing the per-probe mean
(the part softmax ignores). We report the correlation between them and the ratio
of their standard deviations.

| | Vector cosine | **Logit correlation** | Logit spread ratio |
|---|---:|---:|---:|
| **Layer 0**, no centering | 0.9950 | **0.5488** | 1.746 |
| **Layer 0**, centered | 0.9950 | **0.9934** | 1.001 |
| All layers, no centering | 0.9948 | 0.9619 | 1.0595 |
| All layers, centered | 0.9972 | 0.9931 | 1.0012 |

Source: [`paper/experiments/results/quantizer_loop.json`](experiments/results/quantizer_loop.json)

This is the mechanism, demonstrated rather than argued:

- **Vector-level reconstruction is excellent in both arms and cannot tell them
  apart.** Cosine similarity is 0.9950 with and without centering at layer 0 —
  comfortably past the >0.995 success criterion this project set for itself.
- **The quantity softmax actually responds to is destroyed.** At layer 0 the
  attention-logit correlation without centering is 0.5488. Centering restores it
  to 0.9934 at identical bit-width.
- **The error is not merely attenuation, it is injected noise.** The logit spread
  ratio of 1.746 means the uncentered reconstruction produces logit variation 75%
  larger than the truth — the quantizer is manufacturing attention structure that
  is not in the model.
- Damage is concentrated at the ends of the stack: layer 0 (0.5488) and layer 27
  (0.8239) are the worst; middle layers sit near 0.988. Layer 0 was independently
  flagged as anomalous in this repository's April adversarial validation, which
  found rotation choice mattered ~70× more there than elsewhere.

This directly explains §8.5 and Limitation 9: a project validating on vector or
attention cosine similarity would have graded the uncentered configuration as
near-lossless while it was in fact producing perplexity in the thousands.

**Scope.** This experiment measures single-layer key reconstruction, not
end-to-end perplexity. It shows how the discriminative signal is lost at each
layer; it does not by itself prove that this compounds into the PPL 9,410 of
§3.1. The probe directions are random rather than the model's real queries.

### 6.10 The contrast case: a modern bias-free model has the *same* shared component

*(added 2026-08-18)*

§6.6 and §6.8 together invite a comfortable conclusion — that the shared component
is a Qwen2.5 artefact and that current architectures, having dropped the key bias
for QK-Norm, are safe. **They are not.** We repeated §6.7 and §6.9 verbatim on
Qwen3-8B, which is bias-free and applies QK-Norm, using the identical corpus,
token count, probe set, seed and quantizer configuration:

| Statistic | Qwen2.5-7B (bias, no QK-Norm) | Qwen3-8B (no bias, QK-Norm) |
|---|---:|---:|
| **mean cos(k_t, μ_h)** | **0.7150** | **0.7019** |
| **shared-mean energy fraction** | **0.5267** | **0.5104** |
| ρ median | 0.978 | 0.992 |
| heads with ρ > 1 | 43.8% | 49.0% |
| ρ mean | 2.456 | 1.219 |
| **ρ max** | **59.116** | **7.011** |
| **heads with ρ > 3** | **8.9%** | **2.8%** |

Source: [`results/quantizer_loop_qwen3_8b.json`](experiments/results/quantizer_loop_qwen3_8b.json)

**The bulk shared component is not architecture-specific.** Half of the average
key's energy is a softmax-invisible shared component in the modern bias-free model
too — 51.0% against 52.7%, and mean cosine 0.702 against 0.715. Dropping the key
bias did not remove it, exactly as §6.8's within-model ablation predicted.

What the architecture change *did* remove is the **tail**: worst-head ρ falls from
59.1 to 7.0, and the fraction of severely mean-dominated heads from 8.9% to 2.8%.
This matches the within-model ablation almost exactly (ρ max 59.1 → 1.7 when the
bias is zeroed) and identifies the bias as a tail-generating mechanism, not the
source of the bulk.

Quantizer damage follows the tail rather than the bulk:

| | Qwen2.5-7B | Qwen3-8B |
|---|---:|---:|
| Layer 0, uncentered logit correlation | **0.5488** | **0.8669** |
| Layer 0, centered | 0.9934 | 0.9909 |
| Layer 0, uncentered vector cosine | 0.9950 | 0.9971 |
| Layer 0, uncentered spread ratio | 1.746 | 1.095 |
| All layers, uncentered logit correlation | 0.9619 | 0.9836 |
| All layers, centered | 0.9931 | 0.9932 |

Three things follow, and the first is the practically important one:

1. **Centering is still required on current models.** Qwen3-8B's layer 0 loses
   measurable logit fidelity uncentered (0.8669) and recovers to 0.9909 with
   centering, at identical bit-width. The correction is not a legacy patch for a
   superseded model generation.
2. **The metric dissociation persists.** Qwen3-8B's layer 0 scores 0.9971 vector
   cosine in the arm whose logit correlation is 0.8669. A reconstruction metric
   still cannot see the damage.
3. **Severity tracks the tail, not the bulk.** The two models have near-identical
   bulk shared components but 8× different worst-head ρ and correspondingly
   different damage. On `n = 2` models this is an inference, not a law, but it
   sharpens §4.2: the quantity that predicts catastrophe is the *extreme* of the
   ρ distribution, not its centre.

**Scope.** Qwen3-8B is a *proxy* for the current architecture, not the model of
§6.11: it is dense with 8 KV heads and head_dim 128, whereas Qwen3.8-27B is a
hybrid stack with 4 KV heads and head_dim 256. It is also still a Qwen, so the
non-Qwen ablation remains open. We have not measured Qwen3-8B's end-to-end
perplexity under 3-bit compression, so "less per-layer damage" must not be read as
"does not fail".

### 6.11 Qwen3.8-27B: structural verification without downloading it

*(added 2026-08-18)*

Qwen3.8-27B was released in early August 2026 and is the newest model in scope for
this question. It could not be run here — ollama 0.20.0 refuses the manifest as too
new, the transformers GGUF loader does not support the architecture, and 27B
exceeds this host's RAM ceiling regardless (§9, Limitation 11). We instead read its
published weights directly over HTTP range requests, fetching the safetensors
shard headers and then only the exact byte ranges of the tensors of interest — a
few kilobytes in total rather than ~54 GB.

| Property | Qwen2.5-7B | Qwen3.8-27B |
|---|---|---|
| `self_attn.k_proj.bias` | **present** (‖b_h‖ to 920) | **absent** |
| `q_norm` / `k_norm` (QK-Norm) | absent | **present**, 17 attention layers |
| Attention layers | 28 of 28 | 17 of 64 (indices 0, 3, 7, …, 63) |
| Q / KV heads | 28 / 4 | 24 / 4 |
| head_dim | 128 | 256 |

Two verification notes matter here. The repository index lists 166 `.bias`
tensors, but **every one belongs to the vision tower** (`model.visual.*`); vision
transformers conventionally carry a QKV bias. The language model has none. This is
the identical false positive the local GGUF scan produced in §6.6, caught the same
way. Separately, an LLM-summarised read of the config reported "no QK-Norm",
which the tensor listing contradicts — the norm is structural (`q_norm`/`k_norm`
weight tensors) rather than a config flag. Both claims above come from the tensor
manifest, not from a summary of it.

We also read the QK-Norm gain vectors themselves, since a learned per-channel gain
applied to every key is a candidate for reintroducing a shared direction. It does
not: the gains are diffuse, with a participation ratio of 150–221 effective
channels out of 256 (0.59–0.86 of the head dimension), rising through depth.

| Tensor | mean γ | std | min | max | effective channels / 256 |
|---|---:|---:|---:|---:|---:|
| `k_norm` L3 | 0.220 | 0.135 | −0.574 | 0.754 | 150.4 (0.587) |
| `k_norm` L11 | 0.350 | 0.129 | −0.262 | 0.711 | 220.3 (0.861) |
| `k_norm` L23 | 0.436 | 0.179 | −0.965 | 1.203 | 188.8 (0.737) |
| `q_norm` L11 | 0.361 | 0.065 | −0.135 | 0.527 | 238.8 (0.933) |

Source: [`results/qwen38_qknorm_gains.json`](experiments/results/qwen38_qknorm_gains.json),
read via [`experiments/remote_safetensors.py`](experiments/remote_safetensors.py)

Qwen3.8-27B therefore shares the architectural properties of the Qwen3-8B proxy
measured in §6.10 — no key bias, QK-Norm present — and adds no new
shared-direction structure through its norm gains. On that basis we predicted that
it also carries a large bulk shared component and also requires centering.

**§6.12 tested that prediction against a closer proxy and it is wrong.**

### 6.12 The generational trend: Qwen3.5/3.8 largely eliminates the pathology

*(added 2026-08-18; supersedes the prediction at the end of §6.11)*

Qwen3-8B was a weak proxy for Qwen3.8-27B: dense, 8 KV heads, head_dim 128. A far
closer one exists. **Qwen3.5-4B declares the same `model_type` (`qwen3_5`) and the
same class (`Qwen3_5ForConditionalGeneration`) as Qwen3.8-27B, with the same 4 KV
heads, the same head_dim of 256, and the same 3-linear-attention + 1-full-attention
hybrid pattern.** It differs only in depth (32 vs 64 layers) and query-head count
(16 vs 24). It is also small enough to run here, which Qwen3.8-27B is not.

Its 4 KV heads additionally match Qwen2.5-7B exactly, which controls the KV-head
correlation of §6.3 that no earlier comparison could.

Same corpus, tokens, probe set, seed and quantizer configuration throughout:

| Model | Key bias | QK-Norm | KV heads | head_dim | **Shared energy** | **cos to mean** | **ρ max** | **heads ρ>1** |
|---|:--:|:--:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-7B | yes | no | 4 | 128 | **52.7%** | 0.7150 | **59.116** | 43.8% |
| Qwen3-8B | no | yes | 8 | 128 | **51.0%** | 0.7019 | 7.011 | 49.0% |
| **Qwen3.5-4B** | no | yes | 4 | 256 | **29.5%** | **0.5414** | **0.782** | **0.0%** |

And the resulting quantizer damage at 3 bits:

| Model | Uncentered logit corr. | Centered | Uncentered spread | Worst layer (uncentered) |
|---|---:|---:|---:|---|
| Qwen2.5-7B | 0.9619 | 0.9931 | 1.0595 | **L0: 0.5488** |
| Qwen3-8B | 0.9836 | 0.9932 | 1.0082 | L0: 0.8669 |
| **Qwen3.5-4B** | **0.9905** | 0.9929 | 1.0038 | L23: 0.9899 |

Source: [`results/quantizer_loop_qwen35_4b.json`](experiments/results/quantizer_loop_qwen35_4b.json)

The trend is monotonic across three generations and the newest is qualitatively
different, not merely better:

1. **The shared component is roughly halved** — 29.5% of key energy against 51–53%,
   and mean cosine 0.541 against 0.70–0.715.
2. **No mean-dominated heads exist at all.** Not one head in Qwen3.5-4B has
   ρ > 1; the maximum over every head and layer is 0.782. In Qwen2.5-7B the
   maximum is 59.1 and 43.8% of heads exceed 1.
3. **The layer-0 outlier disappears.** Qwen3.5-4B's uncentered logit correlation is
   flat across all eight attention layers (0.9899–0.9911). The catastrophic
   first-layer behaviour that characterises Qwen2.5-7B (0.5488) and is still
   visible in Qwen3-8B (0.8669) is simply absent.
4. **Centering still helps, but it is no longer decisive** — 0.9905 → 0.9929. On
   Qwen2.5-7B layer 0 the same intervention moves 0.5488 → 0.9934.

**This corrects §6.10's headline.** The claim that "the bulk shared component
survives the architecture change" holds for Qwen3 and is false for Qwen3.5/3.8.
The accurate statement is a generational one: Qwen2.5 is severe, Qwen3 is
intermediate, and the Qwen3.5/3.8 family has largely engineered the problem away.

**Confounds, and they are serious.** Four things change at once between Qwen3-8B
and Qwen3.5-4B — head dimension (128 → 256), stack type (dense → hybrid linear
attention), parameter count (8B → 4B), and training recipe. With one model per
generation we cannot attribute the improvement to any of them. The doubled head
dimension is the most mechanically plausible candidate, since it gives the
token-specific component twice the space to occupy relative to a shared direction,
but that is a conjecture. We also measured Qwen3.5-4B, **not** Qwen3.8-27B, and no
end-to-end perplexity was run on any of the three newer models.

### 6.13 End-to-end perplexity, and validation of the harness

*(added 2026-08-18)*

Every measurement to this point had been a per-layer proxy. We built a harness
that patches the repository's production quantizer into `DynamicCache` and
computes wikitext-2 sliding-window perplexity **and** the cheap proxy metrics from
the same forward pass, so each configuration yields a paired
(proxy, ground-truth-damage) observation.

It reproduces the April-9 result closely enough to trust:

| Quantity | April-9 run (`ppl_for_tom.py`) | This harness |
|---|---:|---:|
| Qwen2.5-7B baseline PPL | 7.5225 | **7.5225** |
| 3-bit, no centering | 9410.49 | 10655.23 |
| 3-bit, centered | 7.9029 | 7.7235 |
| 4-bit, no centering | 1048.99 | 938.46 |

The baseline agrees to four decimal places. The compressed arms differ modestly
because this harness compresses **keys only** while the April-9 run also
compressed values at 3 bits; the failure is unchanged in character and magnitude.

Source: [`experiments/ppl_harness.py`](experiments/ppl_harness.py),
[`results/ppl_qwen2.5-7b.json`](experiments/results/ppl_qwen2.5-7b.json)

### 6.14 A cross-architecture atlas

*(added 2026-08-18; final state of the atlas after the full campaign)*

We ran the harness over every model we could obtain, at 2, 3 and 4 bits with
centering on and off — in its final state, **150 keys-only configurations
across 26 models spanning five Qwen generations and twelve non-Qwen lineages**
(Llama 3.2, Gemma 2, Gemma 3, Phi-4, SmolLM2, OLMo 2, Granite 3.3, Falcon 3,
Ministral, OPT, Pythia, Yi 1.5), plus the 18 one-bit stress cells of §6.19.
Worst-case damage without centering, as a multiple of each model's own
uncompressed-KV baseline (table generated from the result files by
[`experiments/gen_atlas_table.py`](experiments/gen_atlas_table.py)):

| Model | Family | KV heads | baseline PPL | 2-bit | 3-bit | 4-bit |
|---|---|---:|---:|---:|---:|---:|
| qwen2.5-7b | Qwen2.5 | 4 | 7.52 | — | **×1,416** | **×125** |
| qwen2.5-1.5b | Qwen2.5 | 2 | 11.18 | **×1,348** | **×580** | **×376** |
| qwen2-1.5b | Qwen2 | 2 | 11.51 | **×1,085** | **×594** | **×138** |
| qwen2-7b | Qwen2 | 4 | 9.19 | **×916** | **×1,048** | **×53** |
| qwen2.5-3b | Qwen2.5 | 2 | 9.71 | **×41** | **×4.78** | ×1.20 |
| qwen3-1.7b | Qwen3 | 8 | 19.28 | **×21** | ×1.95 | ×1.05 |
| pythia-2.8b | Pythia | 32 (MHA) | 12.77 | **×2.35** | ×1.63 | ×1.43 |
| llama3.2-1b | Llama 3.2 | 8 | 16.17 | ×1.26 | ×1.07 | ×1.01 |
| qwen3-4b | Qwen3 | 8 | 16.27 | ×1.21 | ×1.03 | ×1.03 |
| qwen2.5-14b | Qwen2.5 | 8 | 4.26 | ×1.19 | ×1.04 | ×1.00 |
| smollm2-1.7b | SmolLM2 | 32 (MHA) | 10.26 | ×1.13 | ×1.03 | ×1.01 |
| llama3.2-3b | Llama 3.2 | 8 | 13.99 | ×1.11 | ×1.04 | ×1.01 |
| qwen3-14b | Qwen3 | 8 | 10.00 | ×1.07 | ×1.03 | ×1.00 |
| phi4-mini | Phi-4 | 8 | 11.13 | ×1.07 | ×1.02 | ×1.00 |
| ministral-8b | Ministral | 8 | 8.69 | ×1.07 | ×1.02 | ×1.01 |
| granite3.3-2b | Granite 3.3 | 8 | 8.94 | ×1.06 | ×1.02 | ×1.00 |
| opt-2.7b | OPT | 32 (MHA) | 15.47 | ×1.06 | ×1.02 | ×1.00 |
| yi1.5-6b | Yi 1.5 | 4 | 8.40 | ×1.04 | ×0.99 | ×1.00 |
| qwen1.5-1.8b | Qwen1.5 | 16 (MHA) | 15.96 | ×1.03 | ×1.01 | ×1.00 |
| gemma2-2b | Gemma 2 | 4 | 15.38 | ×1.03 | ×1.01 | ×1.00 |
| falcon3-1b | Falcon 3 | 4 | 11.72 | ×1.02 | ×1.01 | ×1.00 |
| olmo2-1b | OLMo 2 | 16 (MHA) | 15.88 | ×1.01 | ×1.00 | ×1.00 |
| gemma3-4b | Gemma 3 | 4 | 28.77 | ×1.01 | ×0.96 | ×0.94 |
| qwen3.5-0.8b | Qwen3.5 | 2 | 20.37 | ×1.01 | ×1.00 | ×1.00 |
| qwen3.5-4b | Qwen3.5 | 4 | 10.77 | — | ×1.00 | — |
| qwen3.5-9b | Qwen3.5 | 4 | 10.39 | ×1.00 | ×1.00 | ×1.00 |

Source: [`results/ppl_*.json`](experiments/results/), aggregated by
[`experiments/metric_analysis.py`](experiments/metric_analysis.py)

This supplies the non-Qwen control the paper had been missing since §7, and
the result is precise. **Every cell above ×5 belongs to Qwen2 or Qwen2.5** —
at any tested bit-width, both model sizes per generation. The only other
natural failures in the atlas are Qwen3-1.7B at 2 bits (×21) and Pythia-2.8B
at 2 bits (×2.35); both show the concentrated worst-layer collapse geometry
(lr_min 0.81 and 0.31 respectively), and both are fixed by centering
(×1.41 and ×1.08). Twelve of fourteen non-Qwen lineages are essentially
untouched at every bit-width tested.

**The atlas definitively refutes the KV-head-count hypothesis of §6.3.** That
correlation was formed by looking only within Qwen2.5, where head count
happened to track model size. The atlas breaks the confound in every
direction:

- **Qwen3.5-0.8B has 2 KV heads** — the same as Qwen2.5-1.5B (×1,348) and
  Qwen2-1.5B (×1,085) — and is completely immune (×1.01 at 2 bits).
- **Gemma-2-2B, Gemma-3-4B, Falcon3-1B and Yi-1.5-6B all have 4 KV heads** —
  the same as Qwen2.5-7B (×1,416) and Qwen2-7B (×916) — and are all immune.
- **MHA is on both sides**: Qwen1.5-1.8B, SmolLM2, OLMo-2 (no GQA) are
  immune, while MHA Pythia-2.8B is the one non-Qwen model with a genuine
  mean-dominance failure.

Low KV-head count is therefore neither sufficient nor necessary for the
failure. §6.3 should be read as a within-family artefact, and every statement
in this paper conditioned on KV-head count is superseded by this table. The
generational severity ordering of §6.12 holds end-to-end, on perplexity, and
§6.20 extends it backward to the birth of the pathology.

One planned cell is absent: Qwen2.5-32B. Its first run failed on GPU
contention; the retry was deliberately aborted mid-download during the final
campaign because its 65 GB of bf16 weights cannot fit a 24 GB GPU alongside
the resident ollama server and the CPU-spilled run would have blocked the
remaining experiments. Its absence is recorded, not hidden, and is not
evidence about the model either way.

### 6.15 The main result: reconstruction metrics do not predict damage

*(added 2026-08-18; numbers reflect the final 176-cell state of the campaign)*

Because §6.14 records the proxy metrics and the true perplexity for every
cell, we can ask directly whether the metrics this field validates compressors
on predict anything. Over all 176 cells — the 150-cell atlas, the 18 one-bit
stress cells of §6.19, and the 8 seed/corpus robustness cells of §6.13 —
correlated against log₁₀(PPL ratio):

| Proxy metric | Spearman | Pearson | best-threshold errors | held-out errors* |
|---|---:|---:|---:|---:|
| **per-vector cosine similarity** | −0.551 | **−0.003** | 25/176 | **43/146** |
| mean attention-logit correlation | −0.757 | −0.210 | 23/176 | 35/146 |
| **worst-layer logit correlation** | **−0.793** | **−0.836** | **9/176** | **11/146** |
| per-layer product | −0.808 | −0.422 | 23/176 | 25/146 |
| logit spread ratio | 0.789 | 0.207 | 25/176 | 125/146 |

\*threshold fitted on the Qwen2.5 cells only, evaluated on 146 cells from 31
unseen model configurations, 13 of them truly broken. Figure 2
([`figures/fig2_metric_scatter.png`](figures/fig2_metric_scatter.png)) plots
every cell against both headline metrics.

Per-vector cosine similarity — the criterion this project itself started with —
is *uncorrelated* with true damage (Pearson −0.003). The individual cells are
more damning than the aggregate: of the 24 broken cells, **6 pass a 0.995
cosine criterion and 12 score ≥0.9947**, including configurations at ×376,
×1,416 and ×2,929 the baseline perplexity, while working configurations are
rejected at 0.9904. The criterion is wrong in both directions at once.

Worst-layer logit correlation is not a universal damage meter either — §6.19
maps its boundary — but its errors are structured, not random:

- **It catches every catastrophic cell.** All 17 cells with damage above ×5
  score below 0.81, across two Qwen generations, two seeds, two corpora and
  Pythia; no cell at or above 0.81 is catastrophic. The ranking is
  cutoff-robust: it beats cosine at every damage cutoff from ×1.5 to ×10
  (9 vs 29, 9 vs 25, 6 vs 18 errors).
- **Its false passes are the moderate uniform-starvation cells** of §6.19
  (1-bit Llama/SmolLM2), which no reconstruction-side statistic can see.
- **Its false alarms are mechanistically real.** The five healthy cells below
  0.81 — OPT-2.7B at 2–3 bits, Pythia-2.8B at 3–4 bits, Granite-3.3-2B at
  1 bit — are models whose score structure genuinely is damaged (OPT's keys are mean-dominated,
  ρ = 2.15, §6.20) but which happen to tolerate it. The "false alarm" is a
  true positive about the quantizer and a false positive about the model —
  the distinction §6.22 formalizes.

**Why this is the durable result.** Computing the statistic requires one
forward pass over a short calibration text and 128 random probe
directions — no labels, no perplexity run. It is strictly cheaper than the
measurement it predicts, and for the catastrophic failure mode this paper
documents, it is the difference between shipping and not shipping a broken
cache. [`experiments/kvcheck.py`](experiments/kvcheck.py) packages it, with
the scope caveats of §6.19 printed in the output.

### 6.16 How much is centering worth? About four bits

*(added 2026-08-18)*

Sweeping bit-width on Qwen2.5-1.5B, keys only, uncentered versus centered:

| Bits | Uncentered PPL ratio | vector cosine | worst-layer logit r | Centered PPL ratio |
|---:|---:|---:|---:|---:|
| 2 | ×1347.8 | 0.9818 | 0.2584 | **×1.07** |
| 3 | ×580.2 | 0.9948 | 0.3397 | ×1.01 |
| 4 | ×376.4 | 0.9986 | 0.4949 | ×1.00 |
| 5 | ×7.67 | 0.9996 | 0.6751 | ×1.00 |
| 6 | ×1.17 | 0.9999 | 0.8402 | ×1.00 |
| 8 | ×1.00 | 1.0000 | 0.9820 | ×1.00 |

Source: [`results/bitsweep_qwen2.5-1.5b.json`](experiments/results/bitsweep_qwen2.5-1.5b.json)

Two readings, and the second is the practically important one.

1. **The failure is not purely distributional after all — but it is close.** Enough
   bits do eventually fix it, at 6. What §3.2 claimed from two data points (3 and
   4 bits) is true over the whole practical range: within 2–4 bits, adding
   precision buys almost nothing (×1348 → ×376 is still catastrophic), and cosine
   similarity climbs to 0.9986 while the model stays broken. Prediction P1 should
   be recorded as *partially* confirmed, and §4.2's mechanism as describing a
   severe rate penalty rather than an absolute barrier.
2. **Centering is worth roughly four bits.** 2-bit centered (×1.07) is *better*
   than 6-bit uncentered (×1.17). For equal quality, centering lets the cache run
   at one third the bit-width, and it costs `d` floats per head — amortized to
   nothing over a sequence. This is the single most actionable number in the
   paper.

### 6.17 Does anything beat mean-removal? Not for free

*(added 2026-08-18)*

Given that centering is worth four bits, it is worth asking whether a slightly
richer preconditioner buys more. We tested four alternatives on real Qwen3.5-4B
keys at 3 bits, scoring with the worst-layer logit correlation validated in §6.15:

| Preconditioner | Side information | logit correlation |
|---|---|---:|
| none | — | 0.9740 |
| **mean-removal** | `d` floats/head | **0.9812** |
| mean + per-channel whitening | `2d` floats/head | 0.9812 |
| mean + standardize rotated coords | `2d` floats/head | **0.8229** |
| project out top-1 principal direction | 1 float/**token** | 0.9829 |
| project out top-4 principal directions | 4 floats/**token** | 0.9860 |

Source: [`results/precondition_qwen3.5-4b_3bit.json`](experiments/results/precondition_qwen3.5-4b_3bit.json)

This is a negative result and we report it as one. Per-channel whitening is
**indistinguishable** from plain mean-removal (within 5e-5) — the anisotropy it corrects is
apparently not what the quantizer is losing. Standardizing each rotated coordinate
is actively harmful, costing more than centering gains, presumably because it
destroys the relative coordinate magnitudes that the inner product depends on.
Only the PCA variants improve on centering, and they do so by paying storage per
token rather than per head: at `d = 256` and 3 bits, four fp16 coefficients per
token is a 8.3% storage increase for a 0.005 gain in logit correlation, which
§6.16 shows is a far worse trade than simply spending those bits on precision.

**Mean-removal appears to be the right operating point**: it captures essentially
all of the freely available gain, and the obvious refinements either tie, hurt, or
cost more than they return.

### 6.18 Causal test: the failure can be induced in an immune model on demand

*(added 2026-08-18, discharging the mechanism half of Limitation 18)*

Every naturally-occurring failure in this paper belongs to Qwen2.5, which leaves
open whether the mechanism is real or whether Qwen2.5 is simply peculiar. We
tested it by **manufacturing the failure in a model that does not have it**.

A fixed random per-head direction `μ` is added to every key immediately before
quantization and subtracted immediately after dequantization. In exact arithmetic
this is a no-op — the model is untouched and sees its own keys. The only thing
that changes is that the quantizer must now represent a large shared component.
Sweeping its magnitude as a multiple of the mean key norm, on Llama-3.2-1B at
3 bits:

| Injected ‖μ‖ | Uncentered PPL | ×baseline | vector cosine | worst-layer logit r | **Centered PPL** |
|---:|---:|---:|---:|---:|---:|
| 0 (natural) | 17.26 | ×1.07 | 0.9950 | 0.9844 | ×1.06 |
| 1× | 18.60 | ×1.15 | 0.9886 | 0.9709 | ×1.06 |
| 3× | 84.45 | ×5.22 | 0.9446 | 0.8899 | ×1.06 |
| 10× | 10,449 | **×646** | 0.6799 | 0.5864 | ×1.07 |
| 30× | 29,603 | **×1,831** | 0.3022 | 0.3243 | **×1.09** |

Source: [`results/inject_llama3.2-1b.json`](experiments/results/inject_llama3.2-1b.json)

Three things follow, and the first is the one this paper most needed:

1. **The mechanism is causal and not Qwen-specific.** A shared key component is
   sufficient to reproduce the entire failure — ×1,831, exceeding the worst
   naturally-occurring Qwen2.5 wikitext cell (×1,416) and of the same order as
   its Gutenberg replicate (×2,929, §6.13) — in a Llama model that is otherwise immune.
   Qwen2.5 is not peculiar; it merely *has* a large shared component naturally.
2. **The damage is dose-dependent**, rising monotonically with ‖μ‖ across three
   orders of magnitude. §4.2 predicts exactly this.
3. **Centering neutralises it completely at every dose.** From ×1 to ×1,831 of
   uncentered damage, the centered arm never leaves ×1.06–×1.09.

**This does not extend the metric result of §6.15, and we are explicit about
that.** In the injected regime, vector cosine *does* track the damage (0.30 at the
worst dose) and would correctly flag these configurations. The metric's blindness
is specific to the *naturally occurring* regime, where the shared component is
large relative to the token-specific deviation but the reconstructed vector still
scores 0.9948 because the mean dominates the vector being reconstructed. So §6.18
validates the **mechanism**; §6.15's metric claim rests instead on the 17
natural catastrophic cells of §6.14 and §6.20 — two Qwen generations, two
seeds, two corpora, plus Pythia — and Limitation 18 is revised accordingly.

### 6.19 Stress test: 1-bit keys, and the scope boundary of every cheap proxy

*(added 2026-08-18. This section reports a negative result about our own §6.15
metric, found by deliberately trying to break it.)*

§6.15's validation had a weakness we set out to close: every broken cell was
Qwen-family, so the "detector" had only ever been tested against one damage
mechanism. To manufacture natural failures with a *different* mechanism — pure
resolution starvation, no shared-component pathology — we took the nine small
atlas models that show no catastrophic failure at 2–4 bits (all ≤ ×1.26) and
ran them at **1-bit keys** (plus the 1-bit residual signs the
pipeline always stores), centering on and off:

| Model | ctr off: PPL ratio | lr_min | ctr on: PPL ratio | lr_min | vec_cos (on) |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-0.8B | ×1.05 | 0.887 | ×1.04 | 0.931 | 0.960 |
| Falcon3-1B | ×1.07 | 0.862 | ×1.06 | 0.928 | 0.964 |
| OLMo-2-1B | ×1.15 | 0.879 | ×1.14 | 0.932 | 0.955 |
| Gemma-2-2B | ×1.17 | 0.881 | ×1.11 | 0.931 | 0.960 |
| Phi-4-mini | ×1.34 | 0.859 | ×1.37 | 0.931 | 0.965 |
| Granite-3.3-2B | ×1.71 | **0.761** | ×1.21 | 0.933 | 0.976 |
| Llama-3.2-3B | ×1.78 | 0.864 | **×2.54** | 0.932 | 0.968 |
| SmolLM2-1.7B | **×2.16** | 0.834 | **×2.02** | 0.928 | 0.973 |
| Llama-3.2-1B | **×4.91** | 0.856 | **×4.00** | 0.932 | 0.970 |

Source: [`results/ppl_*-k1.json`](experiments/results/)

Three results, in increasing order of importance:

1. **One-bit key indices are shippable on some models.** Qwen3.5-0.8B and
   Falcon3-1B take a 4–7% perplexity tax with 1-bit Lloyd-Max indices — about
   2.3 bits per coordinate all-in, counting the pipeline's always-present 1-bit
   residual signs and per-vector norms, i.e. ≈7× key compression. The
   robustness spread across otherwise "immune" families is itself large:
   Llama-3.2-1B pays ×4.9 under the identical quantizer.
2. **The desired non-Qwen natural failures exist** (Llama-3.2 and SmolLM2 above
   the ×2 damage line), and **every cheap proxy misses them**. The centered
   broken cells score lr_min 0.928–0.932 and cosine 0.968–0.973 — comfortably
   above every threshold that §6.15's atlas would set. This is not a defect
   specific to the logit-correlation statistic: per-vector cosine, mean logit
   correlation, and the spread ratio all false-pass the same cells. Adding these
   18 cells to the atlas (126 cells, 23 models), worst-layer logit correlation
   misclassifies 5 of 126 cells where cosine misclassifies 14, and 7 of 104
   held-out cells against cosine's 34 — still strictly dominant, no longer
   perfect.
3. **Why they miss is the interesting part.** Figure 1
   ([`figures/fig1_damage_geometries.png`](figures/fig1_damage_geometries.png))
   contrasts the two damage geometries per layer. Read down the centered columns:
   nine models, one quantizer setting, and the proxies are *constant* —
   lr_min 0.928–0.933, cosine 0.955–0.976 — while true damage spans ×1.04 to
   ×4.00. At 1 bit the noise the quantizer injects is essentially
   model-independent, so a statistic computed from (key, reconstructed-key)
   pairs sees the same thing everywhere. The damage variance lives entirely in
   the *model's sensitivity* to that noise, which no single-pass key-side
   statistic can observe. Schematically: damage ≈ injected noise × model
   sensitivity. §6.15's Qwen failures are detectable because the mean-dominance
   pathology makes the **noise term** explode (score-space collapse at layer 0);
   the 1-bit failures are invisible because only the **sensitivity term**
   varies.

**Restated scope of the §6.15 claim.** Worst-layer logit correlation is a
reliable detector of the concentrated score-space collapse that mean-dominated
keys cause: every atlas cell with damage above ×5 sits below 0.81, and no cell
at or above 0.81 is catastrophic. In the 1-bit regime the statistic errs in
both directions at the margins — it passes the moderate uniform failures
(above), and it under-rates one healthy model (Granite-3.3-2B uncentered,
lr_min 0.761 at a true cost of ×1.71), so below 2 bits a low reading means
"measure end-to-end", not "broken". It is not a general damage meter, and per
point 3 of this section no reconstruction-side statistic can be: certifying a
compressed cache for deployment requires at least one end-to-end measurement.
A cheap proxy can tell you *your quantizer is destroying score structure*; it
cannot tell you *your model happens to be fragile*. Nor is perplexity itself
the top of this ladder: concurrent work on alignment under KV quantization
[ref 20] reports Mistral-7B losing 15.2% of its safety refusals at ×1.03
perplexity — each metric in the hierarchy is blind to damage that lives below
its resolution.

The centering sign-flip on Llama-3.2-3B (×1.78 uncentered → ×2.54 centered) is
noted as an open observation: at 1 bit, spending the codebook on the deviation
around a small mean is evidently not always the right trade. We have not
investigated further.

### 6.20 Lineage: the pathology was born in Qwen2

*(added 2026-08-18)*

§6.12 established a generational arc forward from Qwen2.5. Running the same
harness *backward* through the lineage answers where the pathology began.
Worst-case uncentered damage and the corresponding worst-layer logit
correlation, keys only:

| Generation (release) | Model | KV heads | 2-bit | 3-bit | 4-bit | worst lr_min | centered (worst) |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen1.5 (Feb 2024) | 1.8B | 16 (MHA) | ×1.03 | ×1.01 | ×1.00 | 0.930 | ×1.04 |
| **Qwen2 (Jun 2024)** | 1.5B | 2 | **×1,085** | **×594** | **×138** | 0.221 | ×1.03 |
| **Qwen2 (Jun 2024)** | 7B | 4 | **×916** | **×1,048** | **×53** | 0.381 | ×1.04 |
| Qwen2.5 (Sep 2024) | 1.5B | 2 | ×1,348 | ×580 | ×376 | 0.258 | ×1.07 |
| Qwen2.5 (Sep 2024) | 7B | 4 | — | ×1,416 | ×125 | 0.537 | ×1.03 |
| Qwen3 (Apr 2025) | 1.7B | 8 | ×21.1 | ×1.95 | ×1.05 | 0.806 | ×1.41 |
| Qwen3.5 | 0.8B | 2 | ×1.01 | ×1.00 | ×1.00 | 0.961 | ×1.00 |

Source: [`results/ppl_qwen1.5-1.8b.json`](experiments/results/ppl_qwen1.5-1.8b.json),
[`results/ppl_qwen2-1.5b.json`](experiments/results/ppl_qwen2-1.5b.json),
[`results/ppl_qwen2-7b.json`](experiments/results/ppl_qwen2-7b.json)

Four conclusions:

1. **The pathology appears abruptly at the Qwen1.5 → Qwen2 transition** and at
   full severity immediately: Qwen2 is not an intermediate case, it is as broken
   as Qwen2.5 (×1,048 vs ×1,416 at 3 bits on the 7B models). The arc over five
   generations is immune → catastrophic → catastrophic → intermediate → immune.
2. **This independently re-refutes the k_proj-bias hypothesis (§6.8) from the
   other direction.** Qwen1.5 *carries the same QKV bias* that Qwen2 and Qwen2.5
   do — and is completely immune. The bias is present on both sides of the
   discontinuity; the pathology is on one side only.
3. **The KV-head refutation of §6.14 survives another confound check.** The
   Qwen1.5 → Qwen2 transition did introduce GQA, and within Qwen the pathology
   coincides exactly with that introduction — but Qwen3.5-0.8B (2 KV heads,
   GQA) and every non-Qwen GQA model in the atlas is immune to the
   mean-dominance collapse at 2–4 bits (the 1-bit Llama-3.2 failures of §6.19
   are a different, resolution-starvation mechanism), and Qwen3.5's immunity
   arrives *without abandoning GQA*. GQA is where the pathology
   appeared in this lineage, not what causes it. The remaining candidate is the
   Qwen2-era training recipe, which persisted through Qwen2.5 and was changed
   for Qwen3/3.5.
4. **The detector behaves correctly on all six new catastrophic cells**
   (lr_min 0.221–0.738, far below the 0.81 operating point,
   concentrated-collapse geometry), and on every immune lineage cell
   (≥0.918 over the 2–4-bit cells tabulated here; ≥0.886 including the 1-bit
   Qwen3.5-0.8B cell of §6.19).

Together with §6.14, every catastrophic natural failure observed in this
project's data is now precisely delimited: **Qwen2 and Qwen2.5 at any tested
bit-width, Qwen3 at 2 bits, and nothing else.**

**The mechanism's quantity tracks the discontinuity exactly.** Measuring ρ_h
(per-head mean-to-deviation ratio, post-RoPE, 2,048 wikitext tokens) on the
lineage — a prediction made before the measurement:

| Model | mean ρ | max ρ over heads | shared energy | worst uncentered PPL |
|---|---:|---:|---:|---:|
| Qwen1.5-1.8B | **0.995** | 5.9 | 49.0% | ×1.03 |
| Qwen2-1.5B | **2.841** | 64.3 | 50.7% | ×1,085 |
| Qwen2-7B | **2.228** | 51.9 | 55.7% | ×1,048 |
| Qwen2.5-1.5B | **2.723** | 46.6 | 53.6% | ×1,348 |

Source: [`results/rho_*.json`](experiments/results/), by
[`experiments/rho_lineage.py`](experiments/rho_lineage.py)

Mean ρ triples and the head-level tail explodes (5.9 → 64.3) at exactly the
generation where perplexity collapse appears. The sharpest observation is in
the third column: **shared energy is ~50% in all four models, including the
immune one.** The fraction of key energy carried by the mean — the quantity a
first look at the phenomenon naturally reaches for — does not distinguish
broken from healthy. What distinguishes them is whether the mean *dominates
the per-token deviation* (ρ > 1), i.e. whether normalized keys collapse onto
one direction. This sharpens §6.10's tail-not-centre conclusion into a clean
two-regime picture and further confirms the mechanism of §4.2.

### 6.21 Where the mean comes from: two sources, split by layer

*(added 2026-08-18, discharging Limitation 12's remaining measurement)*

Limitation 12 noted that the shared component's origin — `W_k · E[z]`, the
residual stream's persistent mean pushed through the key projection — was
inferred by elimination, never measured. We measured it on Qwen2.5-7B with
forward hooks: per layer, the exact linear decomposition
`μ_pre = W_k E[z] + b_k` (pre-RoPE; reconstruction error ≤0.3% even through
4-bit weights), the channel concentration of `E[z]`, and how much of the mean
survives RoPE.

| Layer | ‖μ_pre‖ | ‖W_k E[z]‖ | ‖b_k‖ | cos(W_k E[z], μ) | top-16 chan. share of E[z] |
|---:|---:|---:|---:|---:|---:|
| 0 | 604.4 | 10.9 | **604.7** | −0.01 | 0.77 |
| 1 | 159.3 | 12.0 | **158.3** | 0.13 | 0.92 |
| 2 | 65.1 | 12.6 | **64.4** | 0.15 | 0.69 |
| 4 | 30.4 | **21.6** | 11.4 | **0.96** | 0.69 |
| 25 | 32.9 | **18.6** | 17.1 | **0.93** | 0.61 |
| 27 | 921.7 | 20.0 | **920.5** | 0.02 | 0.58 |

Source: [`results/mu_origin_qwen2.5-7b.json`](experiments/results/mu_origin_qwen2.5-7b.json)
(RoPE survival ‖μ_post‖/‖μ_pre‖ is 0.76–1.00 everywhere, mean 0.86 — the mean
is not averaged away by position rotation.)

**The mean has two sources, and they partition the network by layer.**

1. **The boundary layers are pure bias.** At layers 0–3 and 27, μ_pre *is*
   `b_k`: at layer 0 the bias norm is 604.7 against a `W_k E[z]` contribution
   of 10.9 nearly orthogonal to μ; at layer 27 it is 920.5 against 20.0.
   These are precisely the two layers where the quantizer damage concentrates
   (0.54 and 0.80 logit correlation, Figure 1) — and precisely the two layers
   the independent llama.cpp mitigation of §2.6 protects.
2. **The middle layers are massive activations.** From layer 4 to 26,
   `cos(W_k E[z], μ) = 0.91–0.96` and `W_k E[z]` outweighs the bias — with 16
   of 3,584 residual channels (0.45%) carrying 58–82% of `E[z]`'s energy in these
   layers (up to 92% at layer 1).
   This is the massive-activations/attention-sink structure of the prior
   literature, imaged through the key projection.

**This resolves the apparent tension in §6.8.** Zeroing the bias removed "the
extreme tail of mean-dominated heads but left the bulk unchanged" — because
the bias *is* the extreme tail (boundary layers) and the residual-stream mean
*is* the bulk (middle layers). Both statements were correct; they were about
different layers. It also sharpens §6.20's lineage argument: Qwen1.5 "carries
the same QKV bias" *architecturally*, but what matters is bias *magnitude* at
the boundary layers, which is a property of the training run — measured next.

**Practical corollary.** Centering neutralizes both sources at once, which is
why it is uniformly sufficient. But the boundary-layer half of the problem
could equally be fixed at model-conversion time by folding the k_proj bias
into the cache layout (it is a per-layer constant), at zero runtime cost —
worth knowing for engines where a running mean is inconvenient.

**The bias source, traced through the lineage in checkpoint bytes.** Since
`b_k` is only kv_heads × head_dim floats per layer (256–2,048 in this lineage), it can be read from the published safetensors
by HTTP range request without downloading any model. Per-layer ‖b_k‖ across
the lineage:

| Model | L0 | L_last | median over all layers | worst uncentered PPL |
|---|---:|---:|---:|---:|
| Qwen1.5-1.8B | 54.4 | 20.0 | 14.8 | ×1.03 |
| Qwen2-1.5B | **1,111.7** | 15.7 | 12.7 | ×1,085 |
| Qwen2-7B | **580.1** | **891.0** | 22.4 | ×1,048 |
| Qwen2.5-1.5B | **1,102.7** | 14.9 | 11.5 | ×1,348 |
| Qwen2.5-7B | **604.7** | **920.5** | 22.0 | ×1,416 |

Source: [`results/kproj_bias_norms_lineage.json`](experiments/results/kproj_bias_norms_lineage.json)

Three observations close the loop:

1. **The Qwen2 recipe grew a ~20× boundary-layer key bias** (L0: 54 → 1,112)
   while the median layer's bias is unchanged across the entire lineage
   (~12–22). The pathology's birth (§6.20) is visible in the checkpoint bytes.
2. **Qwen2.5 inherited it nearly unchanged** (1,111.7 → 1,102.7 at 1.5B) —
   consistent with Qwen2.5 continuing the Qwen2 recipe, and explaining why
   both generations fail identically.
3. **The bias topology predicts the damage topology.** The 1.5B models have a
   monster bias only at L0; their damage profile dips only at L0 (0.34–0.37,
   next-worst layer ≥0.76). The 7B models have monsters at L0 *and* L27;
   their two worst layers are exactly L0 (0.54) and L27 (0.80), in both
   generations. A prediction made from weight bytes alone, confirmed in the
   activation measurements.

The likely function of these boundary biases is an attention-sink-like
mechanism (cf. massive-activations literature): a constant component that
every query can attend to. Softmax discards it (§2.3), which is precisely why
it is safe to remove for quantization and catastrophic to spend bits on.

### 6.22 The damage law: one cheap scalar and a per-model curve

*(added 2026-08-18. This section subsumes §6.15 and §6.19 and is the paper's
strongest result. It also refutes a hypothesis we formed while designing it.)*

§6.15 showed that no reconstruction metric predicts damage. §6.19 showed that
our own replacement has a scope boundary: at 1 bit, nine models read almost
identically on every proxy (worst-layer logit correlation 0.928–0.933) while
their true damage spans ×1.04 to ×4.00. We diagnosed that as
*damage = injected noise × model sensitivity*, with every proxy measuring only
the first factor. This section measures the second, and finds that the two
together determine damage almost completely.

**The probe.** For each model we perturb cached keys with mean-free isotropic
Gaussian noise — **no quantizer at all** —

    k̂ = k + σ · (‖k‖ / √d) · ε ,   ε ~ N(0, I)

at σ ∈ {0.05, 0.1, 0.2, 0.4, 0.8}, recording perplexity and the same proxy
statistics the atlas records. Five forward passes, once per model, no labels.
This yields the model's **noise-response curve**: score-space noise
(1 − worst-layer logit correlation) → log damage.

**The test.** For every quantizer configuration of that model — different
bit-widths, centering on and off, including the 1-bit stress cells — we predict
its damage by evaluating *its own model's curve* at its measured score-space
noise, and compare with the measured perplexity. The curve is calibrated on
unstructured Gaussian noise and tested on structured quantization error, so
nothing about the quantizer enters the prediction.

| | value |
|---|---:|
| cells predicted | **96** (13 models) |
| **R² on log₁₀ damage** | **0.969** |
| median absolute error | 0.004 log₁₀ (**×1.01**) |
| 90th-percentile error | 0.087 log₁₀ (×1.22) |
| worst error | 0.546 log₁₀ (×3.5) |
| true damage range covered | ×0.99 to **×1,348** |
| **dangerous misses at a ×5 catastrophe gate** | **0 / 96** |

Source: [`experiments/sensitivity_probe.py`](experiments/sensitivity_probe.py),
[`experiments/factorization.py`](experiments/factorization.py),
[`results/sensitivity_*.json`](experiments/results/),
Figure 3 ([`figures/fig3_factorization.png`](figures/fig3_factorization.png)).

**Damage is a model-specific function of one scalar.** Across three orders of
magnitude, a quantizer configuration's perplexity damage is determined by how
much score-space noise it injects, evaluated through a curve that is a property
of the model alone. The structure of the perturbation — Lloyd-Max codebook
error with sign residuals, versus isotropic Gaussian — does not matter beyond
the scalar amount it produces. That is the substantive physical claim, and it
is what makes the prediction cheap.

**This explains the two earlier negative results rather than replacing them.**

- *Why no universal threshold exists* (§6.15): the map from proxy to damage is
  model-specific, so any single cutoff must be wrong for some model. Fitting
  the best **model-agnostic** map from the same scalar (leave-one-model-out)
  gives a 90th-percentile error of ×3.5 against the per-model curve's ×1.22 —
  the per-model calibration is where the accuracy lives, and it is exactly
  what a universal threshold cannot have.
- *Which ingredient does which job.* These are separable and we separate them.
  Repeating the whole procedure with **per-vector cosine** as the abscissa,
  each model still getting its own curve, gives R² = 0.947 on the 14 cells
  with damage above ×1.5 — statistically indistinguishable from the
  score-space abscissa's 0.952. **Per-model calibration, not the choice of
  scalar, is what makes damaged-cell magnitudes predictable.** The scalar
  matters elsewhere: over all 96 cells cosine collapses to **R² = −0.004**
  against 0.971, because within a model the cosine-to-damage map is not
  monotone across the near-lossless mass, and as a ×5 catastrophe gate cosine
  produces five false alarms against one. So §6.15's Pearson −0.003 should not
  be read as "cosine contains no signal"; it contains signal that is
  unusable without per-model calibration and unreliable as a gate even with
  it.
- *Why the 1-bit failures were invisible* (§6.19): identical proxy readings on
  nine models, but the curves differ. Llama-3.2-1B's curve is steep and
  SmolLM2's is flat, so the same score-noise costs ×4.0 on one and ×2.0 on the
  other — predictable once the curve is known, undecidable from the proxy alone.

**We were wrong about the pathology, and the correction is more interesting.**
We predicted (H2, preregistered in the script) that mean-dominated
configurations would land *far above* their own curve, making deviation a
principled pathology detector. **They do not.** Qwen2.5-1.5B's uncentered cells
sit on its curve like everyone else's (median deviation 0.03 log₁₀). The reason
is visible in its curve: **Qwen2.5-1.5B loses ×348 perplexity at σ = 0.05**,
the smallest noise we injected, where Falcon3-1B loses 0.4%. Mean-dominated
models are not damaged by a special mechanism that evades the law — they are
models whose noise curves are catastrophically steep, because when the mean
carries most of a key's norm, a perturbation scaled to that norm swamps the
token-specific deviation that carries all the signal. §4.2's mechanism is
therefore a statement about the *shape of the curve*, and the pathology is
extreme sensitivity, not a separate failure mode.

**A certification protocol follows directly.** Five to eleven *perplexity*
evaluations under injected noise per model (one-time), then one quantized
forward pass per candidate configuration to read its score-space noise — that
pass is label-free and needs only a short calibration text. This predicts
end-to-end damage to within about ×1.2 at the 90th percentile without ever
running a perplexity evaluation *of the candidate configuration*, which is the
cost that scales with the number of configurations under consideration.

**What the curve does and does not add.** Within a single model the curve is a
monotone rescaling of the proxy, so it never changes the *ranking* of that
model's configurations — the proxy alone already orders them. What the curve
supplies is the **magnitude**, and with it cross-model comparability: it turns
"configuration A is noisier than B" into "configuration A costs ×4.0". Pooled
across models, converting the raw proxy into predicted damage raises Spearman
correlation with true damage from 0.793 to 0.880. §6.19's resolution should
therefore be read as *the magnitude becomes recoverable once the curve is
known*, not that the ordering was previously unknowable.

**Held-out validation at bit-widths the analysis never saw.** The bit-width
sweep of §6.16 additionally measured 5-, 6- and 8-bit keys on Qwen2.5-1.5B.
Those three precisions appear in no atlas cell and enter no fit here, so they
serve as held-out tests. Predicting them from the model's Gaussian-noise curve
alone:

| Configuration | predicted | actual | error (log₁₀) |
|---|---:|---:|---:|
| 5-bit | ×5.19 | ×7.67 | −0.170 |
| 6-bit | ×1.10 | ×1.17 | −0.026 |
| 8-bit | ×1.00 | ×1.00 | −0.000 |
| 5/6/8-bit + centering | ×1.00 | ×1.00 | ≤0.0004 |

Median error 0.0003 log₁₀, worst ×1.5. Source:
[`results/bitsweep_qwen2.5-1.5b.json`](experiments/results/bitsweep_qwen2.5-1.5b.json).

**Curve resolution is the dominant error source, and it is fixable.** With the
original five-point σ grid the same held-out test *over*-predicted by up to
×4 (6-bit: ×4.70 predicted against ×1.17 actual), because for a model this
steep the first measured point sits at score-noise 0.60 and everything below
it was linear interpolation from the origin. Adding six small-σ points
(σ ∈ [0.005, 0.04], 15 s of GPU time) reduced the worst held-out error from
0.604 to 0.170 log₁₀. Curves must be sampled where the configurations of
interest actually live; a fixed σ grid is not adequate for steep models.

**Honest residuals, and a hypothesis of ours that failed.** After refinement
the worst residual is Qwen2.5-1.5B at 4 bits uncentered; three of the next four
are 1-bit **centered** cells, and those are the systematic pattern —
under-predicted on every model where they exist (Llama-3.2-1B ×1.55 against
×4.00; Llama-3.2-3B ×1.20 against ×2.54; SmolLM2 ×1.16 against ×2.02).

Our first explanation was that one-bit indices are the most structured, least
Gaussian error in the study, so this is where a Gaussian-calibrated curve
should break. **We tested that and it is wrong.** Re-calibrating the same three
models with noise families matched to the quantizer's actual error — uniform
(the error distribution of a scalar quantizer within a bin), Rademacher signs,
and true deterministic rounding to a uniform grid, all normalised to the same
per-vector RMS error — does not fix the residual:

| Calibration family | median error | 90th pct | median error on 1-bit cells |
|---|---:|---:|---:|
| Gaussian | 0.0116 | 0.194 | 0.164 |
| **real scalar quantization** | 0.0108 | 0.222 | **0.206** |
| Rademacher signs | 0.0097 | 0.190 | 0.152 |
| Gaussian, second seed | 0.0118 | 0.207 | 0.182 |

Source: [`experiments/family_test.py`](experiments/family_test.py),
[`results/family_test.json`](experiments/results/family_test.json)

Real scalar-quantization calibration is *worse* than Gaussian on exactly the
cells it was meant to fix, and all four families under-predict the same
centered cells by nearly the same factor. The perturbation's shape is
therefore not the explanation, which strengthens §6.22's central claim — only
the scalar amount of score-space noise matters, across four quite different
noise geometries — while leaving the residual unexplained. What the residual
does track is **centering at very low bit-width**: uncentered 1-bit cells are
predicted well by every family, centered ones are not. We do not have an
account of why, and record it as open.

**Seed variance is small.** Repeating the Gaussian calibration with a different
random seed changes the median prediction error from 0.0116 to 0.0118 log₁₀
(90th percentile 0.194 → 0.207). The curves are not seed-sensitive, which is
the one variance estimate this paper has (§9, Limitation 5).

**What the law is and is not good for.** Accuracy is strongly regime-dependent:

| True damage | cells | median error |
|---|---:|---:|
| near-lossless (<×1.1) | 68 | ×1.00 |
| mild (×1.1–×2) | 18 | ×1.10 |
| moderate (×2–×10) | 6 | ×1.52 |
| catastrophic (>×10) | 4 | ×1.39 |

Used as a **catastrophe gate** it is excellent: at a ×5 threshold it produces
**zero dangerous misses across all 96 cells** (at a ×2 threshold, four misses,
all of them the 1-bit centered cells above). Used as a **fine-grained budget
selector** it is not adequate: inverting each curve to pick the cheapest
configuration meeting a "≤5% perplexity" budget yields recommendations that
actually meet the budget in only 4 of 13 cases
([`experiments/budget_table.py`](experiments/budget_table.py)), because in the
near-lossless regime the quantity being predicted is smaller than the law's own
error. Distinguishing a 2% tax from an 8% tax still requires measuring it. The
law answers *"will this configuration destroy the model?"* cheaply and
reliably; it does not answer *"is this configuration 3% or 6% worse?"*

**Is the curve itself predictable, so the noise passes could be skipped?** No.
The natural candidate is ρ, since §6.22 argues that mean-dominated keys are
what makes a model fragile. Summarising each curve by its **noise tolerance**
(the score-space noise at which damage reaches ×2) and correlating against the
mean-dominance statistics of §6.20 over the eleven models that have both:

| Predictor | Spearman vs noise tolerance |
|---|---:|
| mean ρ | 0.18 |
| max ρ over heads | 0.24 |
| fraction of heads with ρ > 1 | −0.35 |
| shared energy | −0.07 |

Nothing predicts it. (An earlier five-model version of this table showed
Spearman 0.80 for mean ρ; it did not survive the full cohort, which is worth
recording as a caution about small-n structure claims in this literature,
including our own.) The reason is structural rather than incidental: ρ governs
how efficiently *key-space* perturbation converts into *score-space*
perturbation — and the proxy measurement already absorbs that conversion,
since it is measured in score space. What the curve adds is the model's
intrinsic sensitivity of its output to score-space damage, which is a different
quantity. Pythia-2.8B makes the point: it is the most mean-dominated model in
the study (ρ = 8.4) *and* the most tolerant of score-space noise (tolerance
0.73), while Llama-3.2-1B is barely mean-dominated (ρ = 1.06) and among the
least tolerant (0.089).

**The two factors of §6.19 are therefore genuinely independent and both must be
measured.** Structure (ρ, one pass) tells you how much score-noise a compressor
will produce; the curve (five to eleven passes) tells you what that noise
costs. Neither substitutes for the other, and this is why §6.15's proxy alone
could never have been sufficient.

**Position relative to prior work — the method skeleton is not new.** An
adversarial prior-art search run on 2026-08-18 found that the load-bearing
procedural idea is already published, and we state that plainly:

- **The Linearity Theorem / HIGGS** (Malinovskii et al., arXiv:2411.17525,
  NAACL 2025) introduces, for *weight* quantization, exactly this instrument:
  "a synthetic noise insertion procedure whose role is to mimic the error due
  to compression", with "multiple (J) calibration noise levels … uniformly
  sampled from applicability region" (J = 15) used to fit per-layer
  coefficients α_l that convert reconstruction error into predicted
  perplexity. Anyone claiming novelty for "inject Gaussian noise at several
  magnitudes, fit a per-model response, predict perplexity" would be
  contested by this paper, and rightly.
- **RateQuant** (Zuo et al., arXiv:2605.06675, 2026) publishes the
  factorization itself for KV caches — expected loss as a sum of
  per-head *distortion* terms times *sensitivity* weights — and independently
  reports the transfer problem we hit in §6.22 ("distortion model mismatch":
  one quantizer's distortion curve does not carry to another). Its sensitivity
  weights come from squared gradient norms (forward *and* backward passes),
  and its distortion axis is raw key/value MSE.
- **HeadQ** (arXiv:2605.03562, withdrawn) publishes the metric half — that
  storage-space MSE is the wrong coordinate and score space is the right one —
  across six models with falsification controls, predicting attention KL.

What we did not find, and what this section contributes, is the specific
combination: a **quantizer-free calibration in score space** for the **KV
cache**, carried into the **catastrophic regime**. The distinction from HIGGS
is not cosmetic. Its error model is deliberately local — perplexity is
approximated as PPL* + α·t² inside an "applicability region", and the authors
report that it "diverges on lower bitwidths, where the quantization error is
higher", restricting validated use to above about 3 bits. The regime this
paper exists to explain — 2-bit and 3-bit caches, damage of ×100 to ×1,400 —
is precisely where that quadratic model is stated not to hold. The curve here
is nonlinear and fitted across three orders of magnitude of damage, its
abscissa is a score-space quantity rather than an ℓ₂ reconstruction error
(which §6.15 shows carries no signal across models), and it is validated
against end-to-end perplexity with R² and a held-out bit-width test rather than
against a ranking correlation. Whether that combination is worth a claim of
novelty is for reviewers; the honest statement is that the *instrument* is
borrowed, the *coordinate* and the *regime* are ours, and the prior art above
belongs in any writeup of this result.

**Scope.** 13 models, one corpus, one quantizer family, keys only, single seed.
No cell required extrapolation beyond the largest measured σ; 24 of 96 cells
fall *below* the smallest measured σ and are predicted by linear interpolation
from an assumed zero-noise/zero-damage anchor, which is the weakness §6.22's
resolution finding addresses. The models were chosen to span the damage range,
not sampled from any population.

---

## 7. Ablations

Status of the ablation programme *(refreshed 2026-08-18 after the second
campaign; five of the six original axes now have data)*.

| # | Ablation | Status | Evidence |
|---|---|---|---|
| 1 | Mean removal ON/OFF | **Done** | §6.1, §6.2, §6.4, §6.9 |
| 2 | Multiple bit widths (3, 4) | **Done** | §3.2, §6.1 |
| 3 | Multiple Qwen sizes (3B, 7B, 14B) | **Done** | §6.2 |
| 4 | At least one non-Qwen architecture | **Done** | §6.14 — end-to-end perplexity on twelve non-Qwen lineages (Llama 3.2, Gemma 2/3, Phi-4, SmolLM2, OLMo 2, Granite 3.3, Falcon 3, Ministral, OPT, Pythia, Yi 1.5); §6.20, §6.22 add ρ and noise curves for several |
| 5 | Different context lengths | **Partial** | 4,095 vs 8,191 tokens (§6.1 vs §6.2), confounded with harness changes |
| 5b | Different evaluation corpus | **Done** | §6.13 — wikitext-2 and a Gutenberg replicate, same failure (×1,416 and ×2,929) with near-identical detector readings |
| 6 | Different evaluation samples / seeds | **Partial** | §6.13 — seed 42 vs 43 on Qwen2.5-7B (×1,416 vs ×699 uncentered; both catastrophic, detector 0.537 vs 0.535). Every other cell is still n = 1 |
| 7 | **P3: measure the per-head key mean** | **Done** | §6.7 — the mechanism's load-bearing quantity, now measured |
| 8 | **Causal attribution of the shared component** | **Done** | §6.8 — bias hypothesis tested and refuted |
| 9 | **Quantizer-level demonstration on real keys** | **Done** | §6.9 |
| 10 | **Causal induction of the failure in an immune model** | **Done** | §6.18 — synthetic shared-component injection, dose-dependent to ×1,831 |
| 11 | **Origin of the shared component** | **Done** | §6.21 — `μ = W_k E[z] + b_k` measured; boundary layers are bias, middle layers are massive activations |
| 12 | **Predictive model of damage** | **Done** | §6.22 — per-model noise-response curves, R² 0.969 over 96 configurations |

Items 7–12 were added and executed on 2026-08-18. Item 7 was the previous
draft's "highest-priority next experiment"; the result appears in §6.7 and
confirmed the prediction, while item 8 refuted the most natural follow-on
hypothesis. The single axis still genuinely open is variance estimation:
outside the one seed pair of §6.13, every cell in this paper is n = 1.

### 7.1 Highest-priority next experiment

**Measure `ρ_h` on a model that does *not* fail.** Every activation measurement in
this paper is from Qwen2.5-7B — a model that fails. A mechanism claim needs the
contrast: if Gemma 3, Gemma 4, or Qwen 3 shows a comparably large shared key
component (cos ≈ 0.7, energy fraction ≈ 0.5) while quantizing cleanly, then the
shared component is not sufficient to cause the failure and §4.2 is incomplete.

*Completed 2026-08-18 (second campaign). §6.13–§6.17 supply end-to-end perplexity
across 14 models and 6 families, which settles both open items: the non-Qwen
control (item 4 — Llama 3.2, Gemma 2/3, Phi-4, Falcon 3, SmolLM2 are all immune)
and the question of whether newer models fail (they do not). The KV-head
hypothesis is refuted outright.*

**The highest-priority remaining experiment is now a negative control for the
metric claim.** §6.15's proxy separates 72 cells perfectly, but only 8 of them are
broken, and all 8 come from one family. A metric validated on 8 positives from one
architecture is not yet a validated metric. The right next step is to *manufacture*
diverse breakages — inject synthetic shared components of controlled magnitude into
models that do not naturally fail (Llama, Gemma, Phi), confirm perplexity
collapses, and check that the proxy still fires. That converts the claim from
"separates the failures we happened to find" into "detects failures by
construction."*

Two further candidates would be decisive and are available locally as GGUF blobs:

- **gemma4** — 2 KV heads, *fewer* than the Qwen2.5-7B that failed, but no key
  bias and QK-Norm present. It separates the KV-head-count correlation (§6.3)
  from the shared-component mechanism (§4.2) better than any other model here.
- **qwen3.6:27b** — 4 KV heads, matching Qwen2.5-7B exactly, but bias-free with
  QK-Norm.

Neither is currently runnable on this host: the transformers GGUF loader
materializes a full dequantized state dict in RAM, which caps this machine at
roughly 8B parameters, and neither `gemma4` nor `qwen35` is among the
architectures that loader supports (§9, Limitation 11).

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

*Added 2026-08-18:*

- The shared per-head key component the mechanism requires **exists and is large**
  on the failing model: 53% of the average key's squared magnitude, mean cosine
  0.715 to the head mean (§6.7).
- Running the repository's own 3-bit path on those real keys **reproduces the
  failure at the logit level while vector-level metrics stay clean**: at layer 0,
  cosine 0.9950 in both arms, logit correlation 0.5488 uncentered vs 0.9934
  centered (§6.9).
- Qwen2.5 is the **only** family among seven locally scanned models that adds a
  learned bias to the key projection; every newer architecture scanned (Qwen3,
  Qwen3.5/3.6, Gemma 3, Gemma 4) replaced it with QK-Norm (§6.6), and the same
  holds for Qwen3.8-27B read directly from its published weights (§6.11).
- **Removing the bias alone does not remove the shared component**, but the newest
  architecture largely does. Qwen3-8B keeps the bulk (51.0% vs 52.7%) with a
  smaller tail (ρ max 7.0 vs 59.1); Qwen3.5-4B — same model class, KV-head count
  and head dimension as Qwen3.8-27B — halves the bulk (29.5%), has **no** head
  with ρ > 1, and shows no first-layer outlier (§6.10, §6.12).
- The metric dissociation appears in **every** model measured, the newest included:
  Qwen3.5-4B still scores 0.994 vector cosine uncentered while centering
  measurably improves the logits it produces.

### 8.2 What remains a hypothesis

*This section was substantially rewritten on 2026-08-18. The previous draft's
central caveat — that `‖μ_h‖` had never been measured — no longer applies.*

What is now **measured** is that the shared component is large on the failing
model, and that it is what costs the quantizer its logit fidelity (§6.7, §6.9).

What remains a hypothesis:

- **Sufficiency.** We have not shown that a large shared component is what
  *distinguishes* failing from non-failing models, because every activation
  measurement here is from a model that fails. If Gemma 3 turns out to have a
  comparable shared component and quantizes cleanly, §4.2 is incomplete
  (Limitation 11). This is now the load-bearing gap.
- **Origin.** §6.8 rules out the `k_proj` bias as the source of the bulk shared
  component and points to `W_k · E[x]`, but that attribution is by elimination
  rather than measurement (Limitation 12).
- **Compounding.** §6.9 measures per-layer key reconstruction, not end-to-end
  perplexity, so the path from "logit correlation 0.55 at layer 0" to "PPL 9,410"
  is argued rather than traced (Limitation 14).

The **KV-head dependence** is an observed correlation over three models in a single
family, with parameter count fully confounded. It is a lead, not a result — and
§6.6 now supplies a way to test it: gemma4 has 2 KV heads, *fewer* than the model
that failed, with none of the Qwen2.5-specific weight structure.

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

**On whether this is a historical finding.** This took two measurements and two
corrections to answer, and both intermediate answers are recorded above rather than
quietly replaced. §6.6 suggested the finding was historical (only Qwen2.5 has the
key bias). §6.10 refuted that (Qwen3-8B keeps the bulk shared component). §6.12
then showed §6.10 does not extend to the newest family: on the Qwen3.5/3.8
architecture the shared component is roughly halved, no head is mean-dominated,
and the first-layer collapse is gone.

The defensible synthesis is generational rather than binary:

| Generation | Shared component | Tail | Quantizer damage |
|---|---|---|---|
| Qwen2.5 | large (53%) | extreme (ρ to 59) | catastrophic |
| Qwen3 | large (51%) | moderate (ρ to 7) | intermediate |
| Qwen3.5 / 3.8 | reduced (29%) | none (ρ < 1 everywhere) | mild |

So the *severe* failure is largely a property of models one to two generations old
— which still describes a large amount of deployed inference, since Qwen2.5 and
Qwen3 remain widely used. What does not age is the methodological result: a
reconstruction metric read 0.995 while the attention logits it produced were
barely correlated with the truth (§6.9), and that failure of measurement would
have hidden this bug on any architecture.

### 8.4 What cannot yet be claimed

We cannot claim that:

- canonical TurboQuant fails on Qwen (we tested a variant, §5.1);
- Qwen models in general are vulnerable (14B is not, §6.2);
- KV-head count is causal (confounded, `n = 3`);
- the per-head key mean is large in these models (unmeasured);
- the effect holds outside wikitext-2, seed 42, or these context lengths.

### 8.5 A note on metric choice

§6.4 shows attention cosine similarity above 0.97 in configurations whose
end-to-end perplexity is catastrophic, and §6.9 now quantifies the dissociation
directly on real keys: at layer 0 the reconstructed keys score 0.9950 vector
cosine — past this project's own >0.995 success criterion — in the very arm whose
attention-logit correlation is 0.5488. Whatever the resolution of the measurement
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
10. **Context length is confounded with harness.** The 4,095- and 8,191-token
    results also differ in cache integration and in whether values were compressed,
    so §6.2 is a replication under changed conditions rather than a context-length
    ablation.

*Added 2026-08-18, covering §6.6–§6.9:*

11. **Partially addressed 2026-08-18; still open in its strongest form.** §6.10
    adds Qwen3-8B as a bias-free QK-Norm contrast, which is what showed the bulk
    shared component is architecture-independent. Three gaps remain: (a) we never
    measured Qwen3-8B's **end-to-end perplexity** under 3-bit compression, so we
    know its per-layer damage is milder but not whether it fails; (b) Qwen3-8B is
    still a Qwen — the non-Qwen ablation (item 4 of §7) is untouched at the
    activation level; (c) the two most decisive models remain unrunnable here.
    `gemma4` (2 KV heads, bias-free — *fewer* KV heads than the model that failed)
    and `qwen35`/`qwen3.6` (4 KV heads, bias-free) are local but unsupported by the
    transformers GGUF loader, and Qwen3.8-27B additionally exceeds this host: that
    loader materializes a full dequantized state dict in RAM, capping the machine
    near 8B parameters, and ollama 0.20.0 rejects the Qwen3.8 manifest as
    requiring a newer release.
16b. **The generational comparison is one model per generation, with four
    simultaneous confounds.** Between Qwen3-8B and Qwen3.5-4B the head dimension
    (128 → 256), stack type (dense → hybrid linear attention), parameter count
    (8B → 4B) and training recipe all change together. §6.12's improvement cannot
    be attributed to any single one. Nor was any end-to-end perplexity measured on
    Qwen3-8B or Qwen3.5-4B, so "mild per-layer damage" is not "does not fail".
16. **Qwen3.8-27B is verified structurally but never executed.** §6.11's claims come
    from its published tensor manifest and a few kilobytes of range-fetched weight
    data. No activation, quantizer or perplexity measurement was made on it, and
    the Qwen3-8B proxy differs from it on KV-head count (8 vs 4), head dimension
    (128 vs 256) and stack type (dense vs hybrid). The §6.11 prediction was
    tested on the closest available proxy (Qwen3.5-4B, §6.12) and refuted; it
    remains untested on Qwen3.8-27B itself.
17. **Small `n` for the cross-architecture mechanism comparison.** §6.10's
    quantizer-loop comparison rests on three models and the ρ measurement on
    eleven (§6.20, §6.21, §6.22), all single-run. The tail-not-centre inference
    is now supported by the lineage table, where shared *energy* is ~50% on
    both sides of the failure boundary while the ρ tail differs by an order of
    magnitude — but every cell is still n = 1.

*Added for §6.13–§6.17:*

18. **The metric result rests mostly on one lineage's failures — now with the
    boundary mapped rather than open, and mean-dominance shown insufficient.** *(revised 2026-08-18 after §6.18–§6.20.)*
    As originally stated, every broken cell was Qwen-family and the proxy's
    perfect separation could have reflected one failure mode. Both follow-ups
    have now been run. The synthetic-injection experiment (§6.18) shows the
    mechanism is causal and family-independent. The 1-bit stress test (§6.19)
    manufactured genuine non-Qwen natural failures (Llama-3.2, SmolLM2) and
    found the proxy — and every other cheap proxy — misses them, which is why
    the claim is now stated with an explicit scope: reliable for concentrated
    score-space collapse, structurally blind to uniform moderate damage. What
    remains true and unresolved: every *catastrophic* natural failure (>×20)
    observed to date, here and in the independent reports of §2.6, is a
    Qwen-lineage model. But mean-dominance alone is now known to be
    *insufficient*: Pythia-2.8B (mean ρ 8.43) and OPT-2.7B (2.15) are the two
    most mean-dominated models measured anywhere in this study and neither
    collapses (×2.35 and ×1.06). §6.22 supplies the missing factor — those
    models' noise-response curves are shallow, so the same score-space damage
    costs them little. Mean-dominance sets how much score-space noise a
    compressor produces; the curve sets what that noise costs.
19. **Single seed, single corpus, single context length.** Every cell is one run
    on the first 4,095 tokens of wikitext-2 with a 512-token window. No variance
    estimates. The effect sizes for the broken cells are enormous, but the
    near-boundary cells (Qwen3-1.7B at 3 bits, ×1.95) sit close to the ×2 cutoff
    that defines "broken", and that cutoff is a choice.
20. **Keys only.** The harness compresses keys and leaves values in FP16, which
    isolates the key path but does not measure a deployable configuration. The
    April-9 runs that compressed both are the point of comparison.
21. **The bit-sweep is one model.** §6.16's "centering is worth four bits" is
    measured on Qwen2.5-1.5B alone. The direction is very likely general; the
    specific figure of four bits is not established beyond that model.
22. **Resolved.** *(revised 2026-08-18.)* The three GPU-contention casualties
    of the first campaign all completed on retry: OLMo-2-1B and Granite-3.3-2B
    are in the atlas, and Ministral-8B measures immune (×1.07 worst,
    consistent with the independent Mistral-7B observation of §2.6). The one
    remaining absence is Qwen2.5-32B, documented in §6.14.
12. **Resolved 2026-08-18 (§6.21).** `E[z]` was measured with forward hooks
    and propagated through `W_k` exactly. The residual-stream attribution holds
    for the middle layers (cos(W_k E[z], μ) = 0.91–0.96, layers 4–26), while
    the boundary layers turn out to be pure `k_proj` bias (‖b_k‖ 605–921 at
    layers 0 and 27 of Qwen2.5-7B). The original either/or framing of §6.8 was
    the error; both sources are real and they partition by layer.
13. **Corpus deviation in §6.7–§6.9.** wikitext-2 is not cached on this host and
    these runs were performed download-free, so the 1,024-token corpus is
    committed repository prose plus the real wikitext-2 excerpt that already lives
    in `benchmarks/mean_removal_benchmark.py`. Technical documentation has a
    narrower token distribution than wikitext, which could plausibly *inflate* a
    shared key component. The §6.7 statistics should be re-measured on wikitext-2
    before being quoted as characteristic of the model.
14. **§6.9 uses random probe directions, not the model's real queries**, and
    measures single-layer key reconstruction rather than end-to-end perplexity. It
    demonstrates where the discriminative signal is lost; it does not by itself
    establish that this compounds into the PPL 9,410 of §3.1.
15. **Different weight quantization from the primary runs.** §6.7–§6.9 use the
    Q4_K_M GGUF weights that ollama holds, while §3 and §6.1 use bitsandbytes NF4.
    Both are 4-bit, but they are not the same quantizer, so the activation
    statistics are from a near-neighbour of the model that produced the perplexity
    numbers rather than from that exact artifact.

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
`results/` ignore rule in `.gitignore`. They were force-added in commit
`e9ca7e1df070aeca0c7a9029171504125d0c3a4b` (2026-08-17), which also introduced
this manuscript, so that every number cited here resolves to a tracked file.

| Artifact | Cited in | Permalink |
|---|---|---|
| `rotorquant_comprehensive.md` | §6.2, §6.3 | https://github.com/dhawalc/turboQuantDC/blob/e9ca7e1df070aeca0c7a9029171504125d0c3a4b/benchmarks/results/rotorquant_comprehensive.md |
| `rotorquant_comprehensive_20260415_0835.json` (3B) | §6.2 | https://github.com/dhawalc/turboQuantDC/blob/e9ca7e1df070aeca0c7a9029171504125d0c3a4b/benchmarks/results/rotorquant_comprehensive_20260415_0835.json |
| `rotorquant_comprehensive_20260415_0840.json` (3B, 7B) | §6.2 | https://github.com/dhawalc/turboQuantDC/blob/e9ca7e1df070aeca0c7a9029171504125d0c3a4b/benchmarks/results/rotorquant_comprehensive_20260415_0840.json |
| `rotorquant_comprehensive_20260415_0907.json` (14B) | §6.2 | https://github.com/dhawalc/turboQuantDC/blob/e9ca7e1df070aeca0c7a9029171504125d0c3a4b/benchmarks/results/rotorquant_comprehensive_20260415_0907.json |
| `mean_removal_integration_results.json` | §9 | https://github.com/dhawalc/turboQuantDC/blob/e9ca7e1df070aeca0c7a9029171504125d0c3a4b/benchmarks/results/mean_removal_integration_results.json |

This manuscript's own first published revision:
https://github.com/dhawalc/turboQuantDC/blob/e9ca7e1df070aeca0c7a9029171504125d0c3a4b/paper/qwen_kv_quantization_failure.md

Every figure in the §6.2 table was independently read back out of the raw JSON
files, not transcribed from the summary markdown.

**Experiments added 2026-08-18 (§6.6–§6.9).** These were run **download-free**,
against GGUF blobs already present in the host's local ollama store, using a
dependency-free GGUF reader for the structural scan and the transformers GGUF
loader (architectures `qwen2`, `qwen3`, `gemma3`) for the activation work.

| Script | Produces | Reported in |
|---|---|---|
| [`experiments/scan_architectures.py`](experiments/scan_architectures.py) | [`results/architecture_scan.json`](experiments/results/architecture_scan.json) | §6.6 |
| [`experiments/measure_key_mean_gguf.py`](experiments/measure_key_mean_gguf.py) | [`results/key_mean_rho.json`](experiments/results/key_mean_rho.json) | §6.7 |
| [`experiments/bias_ablation.py`](experiments/bias_ablation.py) | [`results/bias_ablation.json`](experiments/results/bias_ablation.json) | §6.8 |
| [`experiments/quantizer_loop.py`](experiments/quantizer_loop.py) | [`results/quantizer_loop.json`](experiments/results/quantizer_loop.json) | §6.9 |
| [`experiments/gguf_reader.py`](experiments/gguf_reader.py) | (library — no external deps) | §6.6 |

```bash
# Structural scan of every model in the local ollama store. Seconds, no GPU.
python paper/experiments/scan_architectures.py

# Activation measurements. CPU, bfloat16; ~5 min load + ~3 min per forward pass.
# Requires: pip install gguf   (a small pure-Python reader, not a model download)
python paper/experiments/measure_key_mean_gguf.py --models qwen2.5:7b --tokens 1024
python paper/experiments/bias_ablation.py
python paper/experiments/quantizer_loop.py
```

Host used for these runs: RTX 4090 (idle for §6.6–§6.9 — all four ran on CPU),
62 GB RAM, Python 3.13.12, PyTorch 2.11.0+cu130, transformers 5.5.0, gguf 0.19.0.
Model weights are the Q4_K_M GGUF builds ollama ships (see Limitation 15).

**Superseded tooling.** [`experiments/measure_key_mean.py`](experiments/measure_key_mean.py)
is the original HuggingFace-hub version of the §6.7 measurement. It is retained
because it is the path to use once the Qwen2.5 checkpoints are re-downloaded, but
the results in this paper come from the GGUF variant above.

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

> Every arXiv identifier below was verified against the arXiv API on
> 2026-08-18: all IDs resolve and titles and first authors match. `[verify]` now
> marks only claims the API cannot confirm — conference-venue attributions and
> the print-era references (Lloyd, Max, Gersho & Gray).

1. Zandieh, A., et al. *TurboQuant: Online Vector Quantization with Near-optimal
   Distortion Rate.* arXiv:2504.19874, 2025. ID and title verified; the ICLR 2026 venue
   attribution used elsewhere in this repository remains unverified.
2. Zandieh, A., Daliri, M., Han, I. *QJL: 1-Bit Quantized JL Transform for KV Cache
   Quantization with Zero Overhead.* arXiv:2406.03482, 2024.
3. Liu, Z., et al. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.*
   ICML, 2024 `[verify venue]`. arXiv:2402.02750.
4. Hooper, C., et al. *KVQuant: Towards 10 Million Context Length LLM Inference with
   KV Cache Quantization.* NeurIPS, 2024 `[verify venue]`. arXiv:2401.18079.
5. Ashkboos, S., et al. *QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs.*
   arXiv:2404.00456, 2024.
6. Liu, Z., et al. *SpinQuant: LLM Quantization with Learned Rotations.*
   arXiv:2405.16406, 2024.
7. Sun, M., Chen, X., Kolter, J. Z., Liu, Z. *Massive Activations in Large Language
   Models.* arXiv:2402.17762, 2024.
8. Xiao, G., Tian, Y., Chen, B., Han, S., Lewis, M. *Efficient Streaming Language
   Models with Attention Sinks.* ICLR, 2024 `[verify venue]`. arXiv:2309.17453.
9. Ainslie, J., et al. *GQA: Training Generalized Multi-Query Transformer Models
   from Multi-Head Checkpoints.* EMNLP, 2023 `[verify venue]`. arXiv:2305.13245.
10. Qwen Team. *Qwen2.5 Technical Report.* arXiv:2412.15115, 2024.
10b. Qwen Team. *Qwen3 Technical Report.* arXiv:2505.09388, 2025. Source for the
    claim in §6.6 that Qwen3 removes the QKV-bias used in Qwen2 and introduces
    QK-Norm.
11. Merity, S., Xiong, C., Bradbury, J., Socher, R. *Pointer Sentinel Mixture
    Models.* arXiv:1609.07843, 2016.
12. Dettmers, T., Pagnoni, A., Holtzman, A., Zettlemoyer, L. *QLoRA: Efficient
    Finetuning of Quantized LLMs.* NeurIPS, 2023 `[verify venue]`. arXiv:2305.14314.
13. Lloyd, S. P. *Least Squares Quantization in PCM.* IEEE Transactions on
    Information Theory, 28(2):129–137, 1982. `[verify]`
14. Max, J. *Quantizing for Minimum Distortion.* IRE Transactions on Information
    Theory, 6(1):7–12, 1960. `[verify]`
15. *(verified online 2026-08-18)* NSNQuant: A Double Normalization Approach for
    Calibration-Free Low-Bit Vector Quantization of KV Cache. arXiv:2505.18231,
    2025. Prior art for channel-wise centering before KV vector quantization
    (§2.3).
16. *(verified online 2026-08-18)* RotateKV: Accurate and Robust 2-Bit KV Cache
    Quantization for LLMs via Outlier-Aware Adaptive Rotations. arXiv:2501.16383,
    2025.
17. *(verified online 2026-08-18)* KVTuner: Sensitivity-Aware Layer-Wise
    Mixed-Precision KV Cache Quantization for Efficient and Nearly Lossless LLM
    Inference. arXiv:2502.04420, 2025.
18. *(verified online 2026-08-18)* Ruiz Williams, J. L. HeadQ: Model-Visible
    Distortion and Score-Space Correction for KV-Cache Quantization.
    arXiv:2605.03562, 2026. **Withdrawn by its author**; cited only as evidence
    that score-space error metrics for KV quantization were proposed
    independently (§2.5).
19. *(verified online 2026-08-18)* AXELRAM: Quantize Once, Never Dequantize.
    arXiv:2604.02638, 2026. Independent report of Qwen2.5-3B-specific
    catastrophic instability under KV-cache perturbation (§2.6).
20. *(verified online 2026-08-18)* Alignment Collapse Under KV Cache
    Quantization: Diagnosis and Mitigation. arXiv:2606.09864, 2026. Independent
    report of model-specific KV-quantization failures invisible to standard
    metrics (§2.5).
21. Gersho, A., Gray, R. M. *Vector Quantization and Signal Compression.* Kluwer,
    1992. Classical treatment of mean-removed vector quantization (§2.3).
    `[verify edition]`
22. *(verified online 2026-08-18)* Malinovskii, V., Panferov, A., Ilin, I.,
    Guo, H., Richtárik, P., Alistarh, D. *Pushing the Limits of Large Language
    Model Quantization via the Linearity Theorem.* arXiv:2411.17525, 2024;
    NAACL 2025 `[verify venue]`. Prior art for noise-insertion calibration of
    a per-model error-to-perplexity coefficient, for weight quantization
    (§6.22).
23. *(verified online 2026-08-18)* Zuo, et al. *RateQuant: Optimal
    Mixed-Precision KV Cache Quantization via Rate-Distortion Theory.*
    arXiv:2605.06675, 2026. Prior art for the distortion × sensitivity
    factorization of KV-cache damage, with gradient-based sensitivity (§6.22).
24. *(observed online 2026-08-18)* ggml-org/llama.cpp issue #21385 and its
    comment thread: independent measurements of Qwen2.5-7B catastrophic key-cache
    quantization failure, Mistral-7B immunity, and Qwen3.5 q4_0 losslessness in a
    different codebase (§2.6).
    https://github.com/ggml-org/llama.cpp/issues/21385
