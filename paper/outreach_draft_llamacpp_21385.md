# DRAFT — outreach comment for ggml-org/llama.cpp#21385

**Status: DRAFT. Not posted. Requires explicit approval before posting.**

Channel rationale: neither @jagmarques nor @SCJedi publishes an email address.
The issue thread is where @jagmarques said they would post their preprint link,
so it is a venue both parties actively watch, and the comment is on-topic for
the thread. Alternative channels if preferred: an issue on one of
@jagmarques's repositories, or waiting for their preprint and contacting the
listed author email.

---

@jagmarques @SCJedi — your measurements in this thread appear to be independent
observations of a failure mode we have been characterizing in a different
codebase, and I'd like to invite you to check our numbers rather than take our
word for them.

What you observed, and what we see with a TurboQuant-family quantizer
(WHT rotation + Lloyd-Max + residual signs, no llama.cpp code shared):

- @jagmarques: Qwen2.5-7B, 3-bit keys, layer protection off → PPL 6.12 → ~3,300.
  We measure 7.52 → 9,410 (keys+values) / 10,655 (keys only) on wikitext-2 at
  3-bit, same model. Your Mistral-7B +0.31% under the identical setting matches
  our finding that the failure is family-specific: across 150 configurations on
  26 models, every cell above ×5 is Qwen2 or Qwen2.5. Ministral-8B measures
  immune here too (×1.07 worst), consistent with your Mistral-7B number.
- Your choice of *which* layers to protect looks non-accidental. Our per-layer
  profile on Qwen2.5-7B at 3 bits has its two worst layers at exactly L0 (0.54
  attention-logit correlation) and L27 (0.80), with everything between ≥0.91.
  Tracing it to the weights: Qwen2/Qwen2.5 carry a huge `k_proj` bias at those
  boundary layers (‖b_k‖ ≈ 605 at L0 and 921 at L27 on 7B) against a median of
  ~22 elsewhere. Qwen1.5 has no such spike (max 54) and does not fail. The
  1.5B models have the spike only at L0 — and their damage dips only at L0.
- The mechanism we measure: a per-head shared key mean carrying ~53% of key
  energy; subtracting the running mean before quantization and restoring it after
  moves 3-bit PPL from 9,410 to 7.90. Injecting a synthetic shared component into
  Llama-3.2-1B (which is otherwise immune) reproduces the full failure
  dose-dependently, so the mechanism is causal, not a Qwen quirk.
- @SCJedi: your q4_0-lossless result on Qwen3.5 matches our sweep — the Qwen3.5
  generation measures immune at every bit-width we tested (2/3/4-bit, ≤×1.01),
  and Qwen3.5-0.8B survives even 1-bit key indices at ×1.04.

We run a public registry for independent reproduction attempts — negative and
partial results are recorded in the same table as positive ones, and appearing
in it does not imply endorsement of our interpretation:

https://github.com/dhawalc/turboQuantDC/blob/HEAD/REPRODUCTIONS.md

The reference experiment is pinned to immutable commits (script, raw JSON,
report), runs on a single 24 GB GPU in ~30 minutes, and states explicitly what
counts as a match. If either of you has the time to run it — or to tell us why
the experimental design is wrong — both outcomes go in the registry.

One caveat on our own numbers, since it affects what a reproduction should
expect: the uncorrected arm is a diverged quantity and does not reproduce to a
specific value. We see ×1,416 on wikitext-2 and ×2,929 on a Gutenberg text for
the same configuration. Any value above ~100× baseline reproduces the
qualitative claim; the magnitude is not meaningful.

@jagmarques: when your preprint is up we will cite it. Our working manuscript
already records this thread's measurements as independent concurrent
observations (§2.6), and treats them as evidence we could not have produced
ourselves — a second codebase and a different quantizer reaching the same
place matters more than another run of ours.
