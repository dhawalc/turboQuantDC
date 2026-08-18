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
  our finding that the failure is family-specific: in a 116-configuration sweep
  over 23 models, every case with PPL ratio above ×21 is Qwen2.5.
- Your first/last-layer FP16 protection working is consistent with our per-layer
  measurements: on Qwen2.5-7B the damage concentrates at layer 0 (attention-logit
  correlation 0.55 at L0 vs ~0.99 elsewhere).
- The mechanism we measure: a per-head shared key mean carrying ~53% of key
  energy; subtracting the running mean before quantization and restoring it after
  moves 3-bit PPL from 9,410 to 7.90. Injecting a synthetic shared component into
  Llama-3.2-1B (which is otherwise immune) reproduces the full failure
  dose-dependently, so the mechanism is causal, not a Qwen quirk.
- @SCJedi: your q4_0-lossless result on Qwen3.5 matches our sweep — the Qwen3.5
  generation measures immune at every bit-width we tested (2/3/4-bit, ≤×1.01).

We run a public registry for independent reproduction attempts — negative and
partial results are recorded in the same table as positive ones, and appearing
in it does not imply endorsement of our interpretation:

https://github.com/dhawalc/turboQuantDC/blob/HEAD/REPRODUCTIONS.md

The reference experiment is pinned to immutable commits (script, raw JSON,
report), runs on a single 24 GB GPU in ~30 minutes, and states explicitly what
counts as a match. If either of you has the time to run it — or to tell us why
the experimental design is wrong — both outcomes go in the registry.

@jagmarques: when your preprint is up we will cite it; our working manuscript
already credits this thread's measurements as independent concurrent
observations (paper/qwen_kv_quantization_failure.md §2.6 in the repo above).
