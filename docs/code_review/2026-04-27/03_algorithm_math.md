# Algorithm & Math Review (Opus 4.7)

Scope: paper fidelity (PolarQuant, QJL), E8 / E8P lattice math, Lloyd-Max codebook,
Walsh-Hadamard transform, rotation orthogonality, ResidualQuant, mean removal,
edge cases for d != 128.

Overall verdict: PolarQuant + QJL core is correct. WHT, QR, Givens, Quaternion,
Cayley, PCA whitening, and the Lloyd-Max codebook all match the math. Two
**research artifacts** are mislabeled and one ResidualQuant code path is silently
non-equivalent to its docstring's promise. Findings below.

## CRITICAL  (math is wrong → wrong output)

- `turboquantdc/e8_lattice.py:52` — `nearest_d8` flips the **wrong** coordinate
  when the parity correction triggers. Conway-Sloane's algorithm for the nearest
  point in D_n requires flipping the coordinate with the **largest** rounding
  margin (the one whose rounding was the worst). The code uses
  `margin_masked.argmin(dim=-1)`, which picks the smallest margin — i.e. a
  coordinate that's already exactly an integer (margin 0) — and then flips it,
  pushing it from 0 to ±1. Reproduction:

  ```
  x = (0.7, 0.4, 0, 0, 0, 0, 0, 0)
  rounded = (1, 0, 0, 0, 0, 0, 0, 0)   # sum = 1, odd, parity correction triggers
  margin  = (0.3, 0.4, 0, 0, 0, 0, 0, 0)
  argmin(margin) = 2  # picks a coord with margin 0
  → result (1, 0, 1, 0, 0, 0, 0, 0)  squared dist 1.25
  correct nearest D_8 = (1, 1, 0, 0, 0, 0, 0, 0)  squared dist 0.45
  ```

  Fix: change `margin_masked[~needs_fix] = float('inf')` then
  `margin_masked.argmin(...)` to `margin_masked[~needs_fix] = -float('inf')`
  then `margin_masked.argmax(...)`. Also the flip direction must be
  `flip_sign = sign(x_coord - rounded_coord)` (round AWAY from rounded), not
  the residual sign — when the residual is exactly 0 (integer-valued input),
  the current default-to-+1 behaviour is irrelevant once the largest-margin
  coord is selected.

  Impact: `nearest_e8` (line 71) calls `nearest_d8` twice and is therefore
  also wrong. The `relaxed=True` default of `E8Quantizer` bypasses this code
  path, so the bug is masked when the default is used; but every test or
  benchmark that runs strict E8 (`relaxed=False`) is using a quantizer that
  does NOT find the nearest E8 point.

- `turboquantdc/e8_lattice.py:95-115` — `nearest_e8_relaxed` is **NOT the E8
  lattice**. It rounds each input to the nearest point of (1/2) · Z^8 (the
  half-integer lattice), with no parity / coset constraint. It is correctly
  implemented for what it does, but the docstring frames it as a relaxation
  of E8 with "additional ~22% MSE reduction on KV data vs strict E8". In
  reality (1/2)·Z^8 is denser than E8 (E8 has index 2 in (1/2)·Z^8 union
  D8+1/2 union D8); the apparent MSE win is just from having twice as many
  codepoints, not from any lattice optimality. Conway-Sloane's NSM bound
  (0.07168) does not apply here — that bound is for E8.

  Combined with the fact that `relaxed=True` is the default in `E8Quantizer`,
  every paper-claim of the form "we used E8 lattice VQ" in HANDOFF.md
  (PPL +0.1% on 3B, +0.8% on 7B, "E8 vs scalar Lloyd-Max MSE 86-89% lower",
  "E8 3-bit beats FP16 on BnB") is using **half-integer scalar quantization
  in 8D blocks**, not the E8 lattice. The PPL improvements may still be
  genuine, but the technique should be re-named (e.g. "(1/2)Z^8 block
  quantizer") and the Conway-Sloane / Viazovska references should be
  removed because they don't apply.

- `turboquantdc/e8p_codec.py:24-75` — `_build_source_set` does NOT produce
  the QuIP# E8P12 source set. The docstring (line 25-26) claims "227 D8-hat
  points with ||x||^2 <= 10, plus 29 norm-12 points. All coordinates are
  half-integers (multiples of 0.5)." The actual implementation enumerates
  positive-octant patterns over `pos_vals = arange(0, 8) * 0.5`, which
  includes both integers and half-integers. The result (verified by
  enumeration): 256 patterns of which 26 are all-integer, 8 are all-half-
  integer, and **222 are mixed integer/half-integer** — these mixed
  patterns cannot be coset representatives of D8 or D8+1/2.
  Five of the 256 patterns are duplicates (six (0,0,…,0) entries
  collapse to one), so the codebook effectively has 251 distinct
  patterns rather than 256.

  Combined with the +/- 1/4 coset shift in `_build_full_grid` (line 113-115),
  the resulting 65536-entry grid is NOT the E8P12 codebook from QuIP#
  (Tseng et al. ICML 2024) — it is an arbitrary set of half-integer-like
  points. Verified: grid index 1280 decodes to `(1.25, 0.25, …, 0.25)`,
  which un-shifted is `(1, 0, 0, 0, 0, 0, 0, 0)` — that's not in E8
  (sum = 1, odd; not all-half-integer either).

  The roundtrip is internally consistent (encode/decode round-trips
  to grid points), so this is "a 16-bit-per-block VQ codebook that
  works", not "the QuIP# E8P12 codebook reimplemented". The docstring
  (lines 1-12) and the QuIP# reference need to be retracted.

- `turboquantdc/residual_quant.py:152-156` — In `ResidualQuantEstimator.quantize`
  the centering computes `x.mean(dim=0, keepdim=True)`. When called with
  shape `(B*H*new_seq, d)` (the typical layer flattening, see line 400),
  this is the mean across the **flattened** batch×heads×positions dimension
  → a single d-vector that does not respect the per-head structure. The
  docstring says "subtract per-head mean before quantization" (line 53-58),
  but the implementation subtracts a global mean across heads. For Qwen
  models where the catastrophic-failure fix relies on per-head channel
  centering (NSNQuant May 2025), this subtle aggregation is wrong unless
  the caller pre-flattens with `(B*H, S, d)` and quantizes one head at a
  time. Most call sites in the repo do not do this.

  Fix: take the inputs as `(B, H, S, d)` and compute
  `mean = x.mean(dim=2, keepdim=True)` (per-head, per-batch).

- `turboquantdc/residual_quant.py:382-397` — `ResidualQuantLayer.update`
  centers each new chunk of tokens using a **running mean that changes per
  call**. Cached tokens that arrived earlier were centered by an EARLIER
  running mean; later tokens by a LATER one. The dequantized cache
  (`_dequantize_all`, line 459-460) returns `keys_out = keys_centered`
  with the per-token mean never restored. This destroys softmax
  shift-invariance for the assembled cache:

  ```
  Q @ K_orig^T   - Q @ K_centered^T  =  [<Q, mean_at_arrival_t>]_t
  ```

  When `mean_at_arrival_t` differs across t, the per-key offset is NOT
  a constant, so `softmax(Q @ K_orig) != softmax(Q @ K_centered)`.
  Numerical reproduction with d=8, n_tokens=5: max attention probability
  difference ≈ 0.14 — substantial. This is not a free centering; it is
  a position-dependent shift that the queries do not see.

  Math fix: either (a) save the per-token running-mean-at-time-of-arrival
  and add it back during dequantize (loses the storage benefit), or
  (b) use a fixed offline-calibrated mean (no running update), or
  (c) recenter all cached tokens with the latest mean every call (fp16,
  cheap but breaks the "compressed storage" abstraction). The
  HANDOFF.md "PPL 9410 → 7.90" result was likely with option (b) or
  with full-prefill where the running mean was nearly stationary; the
  layer's autoregressive code path is incorrect.

## HIGH  (math is right but missing a known optimization / has a known footgun)

- `turboquantdc/codebook.py:104-107` — Integration bounds `[lo, hi] =
  [-10.5/sqrt(d), +10.5/sqrt(d)]`. With `use_exact=True` (Beta PDF) for
  `d <= 11`, hi exceeds 1 and the integrand is undefined (Beta support is
  `[-1, 1]`, line 41-44 returns 0 outside). Numerically the integral still
  works because `beta_pdf` returns 0, but the partition's leftmost and
  rightmost cells include large empty regions that don't contribute mass —
  fine. With `use_exact=False` (Gaussian) and `d <= 4`, the +/-10.5*sigma
  truncation throws away enough probability mass that the 1-bit centroid
  is biased. Edge case but documented as "d=64 or d=128"; flag a guard
  for `d < 16`.

- `turboquantdc/codebook.py:111-113` — Initial centroid grid is uniform
  in `[-3.5*sigma, 3.5*sigma]`. For `bits=1` with `n_levels=2`, the two
  initial centroids are at ±0.875*sigma. Lloyd-Max converges to the
  correct values (verified: ±sqrt(2/π)/sqrt(d) ≈ 0.0705 for d=128), but
  for highly non-Gaussian residual distributions (stage-2 of ResidualVQ
  at very low bits) the Gaussian-sigma initialization is suboptimal.
  Consider initializing from quantiles of the actual distribution.

- `turboquantdc/residual_quant.py:177-180` — `residual_signs` stores 1 bit
  per coordinate but the `residual_scale` is a **single** scalar
  `r.abs().mean(dim=-1)` shared across all 128 coordinates. When residuals
  are not isotropic (which they generally aren't post-rotation, because
  the rotated coordinate has Beta-not-Gaussian distribution and the
  centroid quantization error has a structured shape), using one scalar
  for all coordinates is suboptimal. The "correct" math (still biased,
  same storage) is to project the residual onto its sign vector:
  `scale = <r, sign(r)> / d = mean(|r|)`. So the formula matches that
  projection — fine. But a per-coordinate-block scalar
  (e.g. 8 scales, one per 16D block) would buy a few percent at +16
  bytes/key.

- `turboquantdc/residual_vq.py:115-119` — Stage-2 codebook is built for
  N(0, mse_per_coord) but stage-1 distortion is computed against the
  **Gaussian approximation** N(0, 1/d) of the true rotated distribution,
  not the empirical post-rotation residual. For bits=1 stage 1, the
  residual-distribution variance is approximately equal to mse_per_coord
  (verified: std ≈ 0.0302 for d=128, expected sqrt(0.000918) ≈ 0.0303).
  But for bits=4 stage 1, the residual is highly non-Gaussian (almost
  uniform on each cell), and matching by sigma-only is loose. Worth
  documenting that the stage-2 codebook is only "approximately optimal"
  for stage-2 residual bits >= 2.

- `turboquantdc/qjl.py:103` — Correction scale `sqrt(pi/2) / m` matches
  the paper. But the code uses `m = self.m`, which defaults to `d`
  (line 56). Verified numerically: 1.4% relative bias over 1000 random
  S matrices (consistent with Monte Carlo noise sqrt(pi/(2m)) ≈ 0.111
  per trial). Math is correct.

- `turboquantdc/rotation.py:55-65` — `generate_rotation_matrix` does QR of
  Gaussian and then sign-flips the columns of Q to make diag(R) positive,
  yielding a Haar-uniform sample. Math is correct. Note `gen` is a CPU
  generator; the result is moved to device — reproducible across devices.
  Good.

- `turboquantdc/rotation.py:178-216` — `apply_wht_rotation` is correct.
  `H_d @ H_d = d * I`, so a single application followed by `/sqrt(d)` gives
  an orthogonal map; verified via `R @ R.T == I` (max err 1e-6) and
  per-coordinate variance of unit-vector inputs is exactly 1/d. The
  inverse path correctly reverses the order (`H_d` then `D` instead of
  `D` then `H_d`), exploiting that `H_d` and `D` are both their own
  inverses.

- `turboquantdc/rotation.py:122-137` — `fast_wht` mutates the input tensor
  in place via `xe[..., 0, :] = s`. Inside `apply_wht_rotation` the input
  is `result = x.clone()` then passed to `fast_wht`, so the user-facing
  call is non-destructive. But callers that import `fast_wht` directly
  (`from turboquantdc.rotation import fast_wht; y = fast_wht(x)`) will
  silently mutate `x`. Document or rename to `fast_wht_`.

- `turboquantdc/learned_rotation.py:118-127` — Water-filling binary search
  uses `100` iterations regardless of convergence; cheap but wasteful.
  More importantly, the "deficit greedy adjustment" (line 137-152) only
  considers fractional parts of the *continuous* rate, not the actual
  per-coordinate distortion gradient. Consider a Lagrangian update over
  `dD/db_i = -ln(2)/2 * 4^{-b_i} * eigenvalue_i` (rate-distortion slope).
  Practical bit allocation is fine for small d; flag for d >= 256.

- `turboquantdc/spectral_compress.py:53-65` — `dct_type2` uses
  `torch.fft.rfft` and complex twiddles. Works for fp32/fp64 only;
  `dct_type2(x.half())` raises "Unsupported dtype Half" because
  `torch.exp(complex)` requires fp32+. If KV cache lives in fp16 (Qwen,
  Llama), callers must `.float()` before invocation — document this.

- `turboquantdc/cayley_quant.py:107-115` — `rotation_matrix` solves
  `(I + A) R = (I - A)` for skew-symmetric A. This is mathematically
  equivalent to `R = (I - A)(I + A)^{-1}` because for skew-symmetric A,
  `(I + A)` and `(I - A)` commute. Verified: at zero params R = I,
  with random params R is orthogonal (max err 1e-7) and det(R) = 1.
  Math correct. The `init_from_wht` (line 163-209) uses Adam to fit A
  because the closed-form inverse Cayley `(I - R)(I + R)^{-1}` is
  singular when R has any eigenvalue at -1 (which WHT often does);
  this is a sensible workaround.

- `turboquantdc/block_rotation.py:69-208` — Givens (2D) and Quaternion
  (4D) block-diagonal rotations. Both verified orthogonal and round-trip-
  exact. Quaternion sandwich `T(v) = q_L * v * conj(q_R)` is the
  standard SO(4) parameterization; with `q_L`, `q_R` independently random
  unit quaternions, this gives 6 of the 8 SO(4) DoF (the unit constraints
  remove one each; combined with the residual q_L↔(-q_L) ambiguity
  there's full SO(4) coverage). Correct.

- `turboquantdc/asymmetric.py:67-82` — Restricts `key_bits, val_bits ∈
  {2,3,4}`. This is a feature (presets), but blocks d=64 hardware
  configurations where 1-bit values + 4-bit keys is interesting. Soft
  block — flag for future expansion.

- `turboquantdc/residual_quant.py:153` — `vec_mean = x.mean(dim=0,
  keepdim=True).expand_as(x)` returns an expanded view with stride 0
  in the batch dim. Stored in the result dict, then `.reshape(...)` at
  line 410 forces materialization of the full `(B, H, S, d)` tensor at
  fp32 — this is the same storage as the original FP16 keys, so a
  3-bit ResidualQuant + mean-removal layer ends up storing
  `3*d/8 + 4*d` bytes/token = roughly 4.4×d bytes/token, which is
  WORSE than fp16 (2*d). The compression claim is wrong unless
  `vec_mean` is deduplicated upstream. Fix: store one mean per (B, H)
  pair (shape `(B, H, 1, d)`), and broadcast at reconstruction time.

## MEDIUM

- `turboquantdc/codebook.py:100` — Switching between `beta_pdf` and
  `gaussian_pdf` via `use_exact=False` default. For d=128 (typical
  Qwen/Llama), the Gaussian approximation is excellent (verified <
  1.2x ratio against paper distortion bounds at b=3 — ratio 1.152
  could come from either the approximation or integration tolerance).
  For d=64 (some models), the approximation introduces ~3% extra
  distortion. Worth documenting the use_exact path more visibly.

- `turboquantdc/e8_lattice.py:194` — `encode_int8` returns `int_codes`
  as `torch.int16`, despite the method name. Comments say "Stores as
  int8 (1 byte per coordinate)" but the actual storage is 2 bytes
  per coordinate. For lattice points scaled by 2 to integers, the
  range can exceed `[-128, 127]` for large `scale * x` inputs;
  int16 is the safe choice but the docstring lies.

- `turboquantdc/e8p_codec.py:78-159` — Brute-force nearest neighbor
  against 65536 codewords using `torch.cdist`. For sequence lengths
  >= 1000 this is the bottleneck (4096*65536 distances per chunk).
  QuIP# achieves O(d) decoding via the (abs_idx, sign_byte, coset)
  factored representation; here both encode and decode go through
  `cdist` because the codebook is treated as opaque. This is fine
  for research but undermines the speed claim.

- `turboquantdc/e8p_codec.py:172-179` — `memory_bytes` returns
  `compression_vs_fp16 = (n_vectors * d * 2) / (n_blocks * 2 +
  n_vectors * 2)`. Correct arithmetic but implicitly assumes
  `n_blocks = n_vectors * (d/8)` and ignores the source set
  `256 * 8 * 4 = 8192` bytes of one-time codebook overhead. For
  small n_vectors the codebook overhead dominates; this is fine
  to surface but should be in the "compression ratio" calculation.

- `turboquantdc/delta_quant.py:222-225` — Delta codebook
  `LloydMaxCodebook(d * 5, delta_bits)` — the `*5` is a hand-tuned
  approximation for "cos=0.9 -> d_eff = d/0.19 ≈ 5.3*d". When grouping
  produces clusters with cos < 0.9, the delta codebook is too tight
  (centroids at 1/sqrt(d*5) ≈ 0.039 for d=64, vs actual std ≈ 0.07).
  Suggest computing per-encoding delta std and choosing d_eff
  data-dependently, or storing per-block sigma instead of relying on
  the codebook scale.

- `turboquantdc/delta_quant.py:300-302` — `medoid_pos = (medoid_mask
  [:medoid_tok + 1]).sum().item() - 1` — O(N) scan inside a Python
  for-loop; for N=2048 this is 2M masked sums. A pre-computed dense
  array `medoid_to_anchor_pos[group_id] = anchor_array_pos` would be
  O(1) lookup.

- `turboquantdc/delta_quant.py:395` — `delta_sigma` is stored as
  fp32 (32 bits per non-anchor token = 0.5 bits/dim for d=64).
  fp16 would halve this with no measurable accuracy loss
  (delta_sigma is bounded between 1e-10 and ~max-residual ~1).

- `turboquantdc/asymmetric.py:80-83` — Hard-coded `key_bits, val_bits
  ∈ {2, 3, 4}` blocks 1-bit values, 5-bit keys, etc. This is a
  preset-ergonomics decision rather than a math constraint; flag
  for relaxation.

- `turboquantdc/residual_quant.py:265-298` — `inner_product` reconstructs
  the key fully and computes a dot product. This is
  `O(seq_len * d)` per query — fine for HF cache integration where
  reconstruction is needed anyway, but defeats the QJL-style
  "compute IP directly without reconstructing" optimization that the
  paper relies on for speed. Document.

## LOW

- `turboquantdc/codebook.py:131` — Lambda inside `quad`:
  `integrate.quad(lambda x: x * pdf(x, d), left, right)`. Inside a
  for-loop with closure-over-`d`, `c_i`. Works but slightly slower than
  passing `pdf` directly; sub-percent.

- `turboquantdc/e8_lattice.py:166-167` — `lattice_pts.reshape(*batch_shape,
  d) * self.scale`. Unscales after reshape — unnecessary since `*` is
  shape-agnostic. Cosmetic.

- `turboquantdc/e8p_codec.py:118` — `device` parameter is `str`-typed
  but PyTorch idiom is `torch.device | str | None`.

- `turboquantdc/spectral_compress.py:99-106` — `dct_scipy` creates a
  numpy array on the fly each call. Fine for analysis; for inner-loop
  use, inline `scipy.fft.dct` directly to avoid the wrapper overhead.

- `turboquantdc/learned_rotation.py:156` — `compute_adaptive_bit_allocation`
  uses 100 binary-search iterations regardless of convergence. Could exit
  on `(hi_theta - lo_theta) < 1e-9 * theta`.

- `turboquantdc/cayley_quant.py:200-209` — `init_from_wht` runs Adam for
  `fit_steps=200` and reports `fit_error`, `ortho_error` to a local
  variable that's never logged or returned. Either log or remove.

- `turboquantdc/attention_optimal.py:327-328` — Threshold computation
  `cumsum <= total_mass * (1 - high_pct + 0.01)` mixes percentile and
  cumulative mass; the `+0.01` slop is brittle. Cleaner to use
  `torch.quantile(importance, 1 - high_pct)` directly.

