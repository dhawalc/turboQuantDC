# Reference Implementation Analysis — Darwin

## Files Analyzed

| File | Lines | Purpose |
|------|-------|---------|
| `lloyd_max.py` | 132 | Lloyd-Max codebook solver |
| `turboquant.py` | 287 | Core algorithm (MSE, Prod, KVCache) |
| `compressors.py` | 223 | Production compressors for real model validation |
| `test_turboquant.py` | 309 | Synthetic validation suite |
| `validate.py` | 198 | Real model (Qwen2.5-3B) validation |

---

## 1. Lloyd-Max Solver (`lloyd_max.py`)

### Algorithm
- Classical Lloyd-Max iteration: alternate between (a) midpoint boundaries and (b) conditional-expectation centroids.
- Centroids updated via `E[X | X in partition_i]` computed with `scipy.integrate.quad`.
- **NOT** gradient descent or empirical k-means. Pure continuous 1-D optimal quantization.

### Distribution
- Two PDFs: exact Beta `f(x) = Gamma(d/2)/(sqrt(pi)*Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)` and Gaussian approximation `N(0, 1/d)`.
- Default is Gaussian (accurate for d >= 64). Exact Beta available via `use_exact=True` but never used in practice.

### Initialization and Convergence
- Centroids initialized uniformly in `[-3.5*sigma, 3.5*sigma]` where `sigma = 1/sqrt(d)`.
- Integration edges extend to `lo*3 = -10.5*sigma` and `hi*3 = 10.5*sigma`.
- Convergence: `max_shift < 1e-10`, up to 200 iterations.
- Denominator guard: `if denominator > 1e-15`.

### Output
- `centroids`: `(2^bits,)` float32. Symmetric around 0.
- `boundaries`: `(2^bits - 1,)` float32.

### Quantize/Dequantize
- Brute-force nearest centroid: `x.unsqueeze(-1) - centroids` then `abs().argmin(dim=-1)`.
- Could use `torch.searchsorted` on boundaries for better scaling.

---

## 2. Core Algorithm (`turboquant.py`)

### TurboQuantMSE (Stage 1)

**Rotation:**
```python
Q, R = torch.linalg.qr(G)  # G is (d,d) Gaussian
diag_sign = torch.sign(torch.diag(R))
diag_sign[diag_sign == 0] = 1.0
Q = Q * diag_sign.unsqueeze(0)  # Haar-uniform rotation
```
- Generator on CPU for reproducibility, then `.to(device)`.
- Rotate: `x @ Pi.T`, Unrotate: `y @ Pi` (orthogonal).

**Data Flow:**
```
x: (batch, d)
-> rotate: y = x @ Pi.T -> y: (batch, d)
-> quantize: y.unsqueeze(-1) - centroids -> (batch, d, n_levels) -> argmin -> indices: (batch, d)
-> dequantize: centroids[indices] -> y_hat: (batch, d) -> unrotate: x_hat = y_hat @ Pi -> (batch, d)
```

**Critical:** Assumes input vectors are unit-norm. No normalization internally.

### TurboQuantProd (Stage 1 + Stage 2)

- Bit budget: `mse_bits = max(bits - 1, 1)`. 3-bit total = 2-bit MSE + 1-bit QJL.
- QJL Matrix: `S: (m, d)` where `m` defaults to `d`. Seeded with `seed + 1`.
- Sign handling: `torch.sign(projected)` then `qjl_signs[qjl_signs == 0] = 1.0`.

**Inner Product Estimator:**
```
<y, x> ≈ <y, x_mse> + ||r|| * sqrt(pi/2) / m * <S@y, sign(S@r)>
```
- Query `y` is projected through S but NOT quantized (asymmetric).

### TurboQuantKVCache
- Keys: `TurboQuantProd` (need unbiased inner products for attention).
- Values: `TurboQuantMSE` (need MSE reconstruction for weighted sum).
- Seeds: key=`seed`, value=`seed+100`.

---

## 3. Production Compressors (`compressors.py`)

### Key difference: handles non-unit vectors
```
Input: (B, H, S, D) float16
-> flatten to (N, D), cast to float32
-> normalize: flat / (||flat|| + 1e-8), store vec_norms: (N, 1)
-> rotate, quantize, reconstruct, unrotate, rescale
-> residual = flat - k_mse
-> QJL: signs = (projected >= 0) * 2 - 1
```

### Asymmetric attention (batched matmul):
```python
term1 = Q.float() @ k_mse.float().T              # (B, H, Sq, Sk)
q_projected = Q.float() @ S.T                     # (B, H, Sq, D)
qjl_ip = q_projected @ signs.float().T            # (B, H, Sq, Sk)
term2 = sqrt(pi/2)/m * qjl_ip * r_norm[..., None, :]
scores = term1 + term2
```

---

## 4. Test Thresholds

| Test | Threshold |
|------|-----------|
| Centroid symmetry | `sum < 0.01` |
| MSE distortion / bound | `<= 1.5` |
| MSE bound formula | `sqrt(3)*pi/2 * (1/4^b)` |
| IP distortion bound | `sqrt(3)*pi^2/d * (1/4^b)` |
| Inner product bias | Near 0 (reported, not asserted) |

---

## 5. Bugs and Limitations Found

1. **Dead code in `compressors.py` lines 97-99** — wrong rotation computed then overwritten.
2. **k_mse stored as float16** — stores full reconstruction (16 bits/coord). No actual memory savings. Must store only indices + norms.
3. **QJL signs not bit-packed** — stored as int8 (8x waste). Must pack 8 signs per uint8 byte.
4. **No streaming/incremental** — list of dicts, no pre-allocation.
5. **Codebook solved at construction** — ~100ms per codebook via scipy. Must cache.

---

## 6. What to Replicate

1. Lloyd-Max solver with Gaussian N(0, 1/d) and scipy.integrate.quad
2. Rotation via QR with sign(diag(R)) correction
3. Bit budget: (bits-1) MSE + 1 QJL, `max(bits-1, 1)` floor
4. Sign mapping: `(projected >= 0) * 2 - 1` (cleaner than torch.sign + zero check)
5. Inner product estimator formula
6. Normalization: store norms, normalize before rotation, rescale after
7. Asymmetric estimation: only keys compressed, queries projected but not quantized

## 7. What to Do Differently

1. **Store indices, not reconstructed vectors** — reconstruct on-the-fly
2. **Bit-pack QJL signs** — 8 signs per uint8 byte (d=128 → 16 bytes vs 128)
3. **Use `torch.searchsorted`** on boundaries instead of brute-force argmin
4. **Cache codebooks** to disk per (d, bits) pair
5. **Pre-allocate cache** — fixed-size tensor buffer, not growing list
6. **Fused CUDA kernels** — rotation + quantize + QJL in one pass

## 8. Exact Tensor Shapes (Production Path)

| Stage | Tensor | Shape | Dtype |
|-------|--------|-------|-------|
| Input | KV states | (B, H, S, D) | float16 |
| Flatten | flat | (N, D) | float32 |
| Norms | vec_norms | (N, 1) | float16 (stored) |
| Rotated | rotated | (N, D) | float32 |
| Quantized | indices | (N, D) | uint8 |
| Reconstructed | k_mse | (N, D) | float32 (on-the-fly) |
| Residual norm | r_norm | (N,) | float16 (stored) |
| QJL signs | signs | (N, D/8) | uint8 (packed) |

## 9. Key Constants

| Constant | Value |
|----------|-------|
| Correction scale | `sqrt(pi/2) / m` |
| Codebook init range | `[-3.5/sqrt(d), 3.5/sqrt(d)]` |
| Integration bounds | `[-10.5/sqrt(d), 10.5/sqrt(d)]` |
| Convergence tol | `1e-10` |
| Max iterations | 200 |
| Norm epsilon | `1e-8` |
| MSE bound | `sqrt(3)*pi/2 * (1/4^b)` |
| IP distortion bound | `sqrt(3)*pi^2/d * (1/4^b)` |
