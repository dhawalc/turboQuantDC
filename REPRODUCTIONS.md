# Independent Reproductions

This page tracks independent attempts to reproduce published TurboQuantDC
experiments.

Positive, negative, and partially matching results are included.

The purpose is to distinguish results produced by the project author from
results independently observed by other engineers or researchers.

**To submit a result**, open an
[Independent Reproduction Report](https://github.com/dhawalc/turboQuantDC/issues/new?template=independent-reproduction.yml).
Results that disagree with the reference experiment are explicitly welcome, and
are recorded in the same table as results that agree.

---

## Qwen2.5-7B — Low-Bit KV-Cache Experiment

Reference experiment:

| Configuration | Reference PPL |
|---|---:|
| FP16 | ~7.52 |
| 3-bit baseline | ~9,410 |
| 3-bit + key-mean correction | ~7.90 |

Reproduction package (immutable):

- Script — [`benchmarks/ppl_for_tom.py` @ `8f54a28`](https://github.com/dhawalc/turboQuantDC/blob/8f54a28e4a0adf139542408efb0f5972eeed1ba0/benchmarks/ppl_for_tom.py)
- Raw results — [`benchmarks/results/ppl_for_tom.json` @ `8f18fae`](https://github.com/dhawalc/turboQuantDC/blob/8f18faef27252910c0bc70ddd87ce427e41ef107/benchmarks/results/ppl_for_tom.json)
- Report — [`benchmarks/results/ppl_for_tom.md` @ `8f54a28`](https://github.com/dhawalc/turboQuantDC/blob/8f54a28e4a0adf139542408efb0f5972eeed1ba0/benchmarks/results/ppl_for_tom.md)
- Analysis and limitations — [`paper/qwen_kv_quantization_failure.md`](https://github.com/dhawalc/turboQuantDC/blob/1654b63efd3c424ecddbfa3ada98d233e81e8360/paper/qwen_kv_quantization_failure.md)

### Exact configuration being reproduced

Reproducers should be aware that **this repository contains two harnesses that
measure the same phenomenon and produce different numbers**, because they
compress different things. Please state which one you ran. Both are reference
experiments; neither supersedes the other.

| | **Reference A** | **Reference B** |
|---|---|---|
| Harness | [`benchmarks/ppl_for_tom.py`](https://github.com/dhawalc/turboQuantDC/blob/8f54a28e4a0adf139542408efb0f5972eeed1ba0/benchmarks/ppl_for_tom.py) | [`paper/experiments/ppl_harness.py`](https://github.com/dhawalc/turboQuantDC/blob/1654b63efd3c424ecddbfa3ada98d233e81e8360/paper/experiments/ppl_harness.py) |
| Date run | 2026-04-09 | 2026-08-18 |
| Keys | 3-bit | 3-bit |
| Values | **3-bit** | **FP16 (uncompressed)** |
| FP16-KV baseline | **7.5225** | **7.5225** |
| 3-bit, no correction | **9410.4876** | **10655.2268** |
| 3-bit + key-mean correction | **7.9029** | **7.7235** |
| 4-bit, no correction | 1048.9915 | 938.4582 |
| 4-bit + correction | 7.7583 | 7.5817 |

Shared settings for both:

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen2.5-7B-Instruct` (HF hub; revision not pinned upstream — please report the revision you resolved) |
| Model weights | 4-bit NF4 via bitsandbytes, `bnb_4bit_compute_dtype=float16` |
| Dataset | `wikitext-2-raw-v1`, `test` split, non-empty lines joined |
| Evaluated tokens | 4,095 (`max_length=4096`) |
| Sliding window / stride | 512 / 256, overlap masked to `-100` |
| Cache | fresh per window (no cross-window state) |
| Quantizer | WHT rotation + Lloyd-Max + residual-sign correction |
| `fp16_window` / `anchor_interval` | 0 / 0 (no uncompressed window, no FP16 anchor layers) |
| Seed | 42 |
| Reference hardware | NVIDIA RTX 4090, 24 GB |

**Note on the "FP16" row.** It denotes an **FP16 KV cache**, not an FP16 model.
Model weights are 4-bit NF4 in every row including the baseline. The comparison
is like-for-like with respect to the KV cache, which is the variable under study.

### What counts as a match

Perplexity in the uncorrected arm is a diverged quantity and is not expected to
reproduce to a specific value. Suggested reporting criteria:

| Row | Reasonable agreement |
|---|---|
| FP16-KV baseline | within ±0.05 of 7.52 — this row should reproduce tightly |
| 3-bit + correction | within ±0.5 of the reference for your harness |
| 3-bit, no correction | **any value above ~100× baseline** reproduces the qualitative claim; the specific magnitude is not meaningful |

A run where the baseline does not reproduce indicates an environment or data
difference and should be reported as such rather than as a failed reproduction of
the compression claim.

Known sources of legitimate variance: bitsandbytes and transformers versions,
upstream re-uploads of the model (the revision is not pinned), GPU/driver
numerics, and tokenizer version affecting the exact 4,095-token window.

---

## Independent Results

| Contributor | Affiliation* | Hardware | Commit | FP16 | 3-bit baseline | 3-bit corrected | Result |
|---|---|---|---|---:|---:|---:|---|
| Awaiting reproduction | — | — | — | — | — | — | — |

\*Affiliation is recorded only when publicly provided by the contributor.

**Result** is one of: `matched`, `partially matched`, `did not match`,
`could not run`.

No entry is added to this table unless the contributor actually ran the
experiment and reported the result themselves. The maintainer does not enter
third parties on their behalf.

---

## Submission Requirements

Independent reproduction reports should include:

- TurboQuantDC commit hash
- exact model revision
- GPU / hardware
- software environment
- evaluation dataset and configuration
- context length
- quantization configuration
- FP16 result
- baseline quantized result
- corrected result
- raw logs or machine-readable output when possible

Results that disagree with the reference experiment are explicitly welcome.

---

## Interpretation

A reproduction appearing in this registry does not imply endorsement of
TurboQuantDC or its conclusions.

Results are recorded to create a transparent experimental record from which
the reliability and generality of individual findings can be evaluated.

Reproductions are not required to accept the project's explanation of *why* the
failure occurs. The reference experiment is a measurement; the proposed
mechanism is documented separately in
[`paper/qwen_kv_quantization_failure.md`](https://github.com/dhawalc/turboQuantDC/blob/1654b63efd3c424ecddbfa3ada98d233e81e8360/paper/qwen_kv_quantization_failure.md)
along with its limitations, including hypotheses the project has itself tested
and refuted.

---

## Other Experiments Open to Reproduction

These are secondary and have no submissions yet. The issue template accepts them
under "other experiment".

| Experiment | Claim | Reference |
|---|---|---|
| **Damage law** | a per-model noise-response curve, calibrated with Gaussian key noise and **no quantizer**, predicts any quantizer configuration's perplexity damage: R² 0.969 over 96 configurations, 13 models, ×1 to ×1,348 damage, zero dangerous misses at a ×5 gate | [`paper/experiments/sensitivity_probe.py`](https://github.com/dhawalc/turboQuantDC/blob/bb2fd5d0a9b03ace3838ab30d2d673e0c5d20f27/paper/experiments/sensitivity_probe.py) + [`factorization.py`](https://github.com/dhawalc/turboQuantDC/blob/bb2fd5d0a9b03ace3838ab30d2d673e0c5d20f27/paper/experiments/factorization.py) |
| **Held-out bit-widths** | the same curves predict 5-, 6- and 8-bit cells that entered no fit, to within ×1.5 | [`results/bitsweep_qwen2.5-1.5b.json`](https://github.com/dhawalc/turboQuantDC/blob/bb2fd5d0a9b03ace3838ab30d2d673e0c5d20f27/paper/experiments/results/bitsweep_qwen2.5-1.5b.json) |
| **Lineage** | the pathology is absent in Qwen1.5, full-severity in Qwen2, and gone by Qwen3.5; ρ triples at the discontinuity while shared key energy stays ~50% | [`paper/experiments/rho_lineage.py`](https://github.com/dhawalc/turboQuantDC/blob/bb2fd5d0a9b03ace3838ab30d2d673e0c5d20f27/paper/experiments/rho_lineage.py) |
| **Boundary-layer bias** | Qwen2 grew a ~20× layer-0 k_proj bias that Qwen1.5 lacks; readable from published checkpoints by HTTP range request, no download | [`results/kproj_bias_norms_lineage.json`](https://github.com/dhawalc/turboQuantDC/blob/bb2fd5d0a9b03ace3838ab30d2d673e0c5d20f27/paper/experiments/results/kproj_bias_norms_lineage.json) |
| **Pre-deployment check** | `kvcheck.py` flags a broken configuration in seconds without a perplexity run | [`paper/experiments/kvcheck.py`](https://github.com/dhawalc/turboQuantDC/blob/bb2fd5d0a9b03ace3838ab30d2d673e0c5d20f27/paper/experiments/kvcheck.py) |
| Cross-architecture atlas | 150 configurations over 26 models; every cell above ×5 belongs to Qwen2 or Qwen2.5 | [`paper/experiments/atlas_run.py`](https://github.com/dhawalc/turboQuantDC/blob/1654b63efd3c424ecddbfa3ada98d233e81e8360/paper/experiments/atlas_run.py) |
| Proxy-metric validation | per-vector cosine similarity does not predict perplexity damage (Pearson −0.003 over 176 cells); worst-layer attention-logit correlation does | [`paper/experiments/metric_analysis.py`](https://github.com/dhawalc/turboQuantDC/blob/1654b63efd3c424ecddbfa3ada98d233e81e8360/paper/experiments/metric_analysis.py) |
| Bit-width sweep | 2-bit corrected outperforms 6-bit uncorrected on Qwen2.5-1.5B, i.e. centering is worth ~4 bits | [`paper/experiments/results/bitsweep_qwen2.5-1.5b.json`](https://github.com/dhawalc/turboQuantDC/blob/1654b63efd3c424ecddbfa3ada98d233e81e8360/paper/experiments/results/bitsweep_qwen2.5-1.5b.json) |
| Needle-in-a-haystack | 0/3 → 3/3 recall at 8K context with the correction | [`benchmarks/niah_for_tom.py` @ `8f18fae`](https://github.com/dhawalc/turboQuantDC/blob/8f18faef27252910c0bc70ddd87ce427e41ef107/benchmarks/niah_for_tom.py) |
