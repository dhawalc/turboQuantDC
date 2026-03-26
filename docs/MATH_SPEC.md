# TurboQuant Mathematical Specification

*Extracted from: "TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate"*
*Authors: Zandieh, Daliri, Hadian, Mirrokni (Google Research / NYU / Google DeepMind)*
*Paper: arxiv 2504.19874v1, April 2025 / ICLR 2026*
*Extracted by: Archimedes (Math Researcher Agent), 2026-03-25*

---

## 1. Problem Definition

**Input:** Vector x in R^d (assumed unit norm ||x||_2 = 1 without loss of generality; for non-unit norm vectors, store the norm separately in FP16 and rescale after dequantization).

**Output:** A binary string Q(x) of B = b * d bits (b bits per coordinate).

**Two distortion measures to minimize:**

**(MSE distortion)**
```
D_mse := E_Q [ ||x - Q^{-1}(Q(x))||_2^2 ]                              (Eq. 1)
```

**(Inner product distortion)**
```
D_prod := E_Q [ |<y, x> - <y, Q^{-1}(Q(x))>|^2 ]                       (Eq. 2)
```

**Unbiasedness requirement** (for inner product quantizer):
```
E_Q [ <y, Q^{-1}(Q(x))> ] = <y, x>
```

---

## 2. Coordinate Distribution After Random Rotation

### Lemma 1: Beta Distribution of Coordinates

For any positive integer d, if x is uniformly distributed on the unit hypersphere S^{d-1}, then for any coordinate index j in [d], the coordinate x_j follows:

```
x_j ~ f_X(x) := Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^((d-3)/2)
```

for x in [-1, 1].

### Proof derivation (from the paper):
```
f_X(x) = [ 2*pi^{(d-1)/2} / Gamma((d-1)/2) * (1-x^2)^{(d-2)/2} * 1/sqrt(1-x^2) ]
         / [ 2*pi^{d/2} / Gamma(d/2) ]

       = Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2)) * (1 - x^2)^{(d-3)/2}
```

This is the ratio of the surface area of a (d-1)-sphere of radius sqrt(1-x^2) to the surface area of the full d-sphere, with the Jacobian factor 1/sqrt(1-x^2).

### Gaussian Approximation

In high dimensions, this Beta distribution converges to a normal distribution:

```
f_X(.) --> N(0, 1/d)
```

**For implementation:** Use the Gaussian approximation N(0, 1/d) for d >= 64. For smaller d, use the exact Beta PDF.

The Gaussian approximation PDF:
```
f_X(x) ~ 1/sqrt(2*pi/d) * exp(-x^2 * d / 2)     [i.e., N(0, sigma^2=1/d)]
```

### Key Property: Near-Independence

In high dimensions, distinct coordinates of Pi * x become nearly independent (not just uncorrelated -- a deeper property from concentration of measure). This is what justifies applying INDEPENDENT scalar quantizers to each coordinate.

---

## 3. Random Orthogonal Rotation

### Generation

Generate rotation matrix Pi in R^{d x d} via QR decomposition:

```
1. Sample G in R^{d x d} with i.i.d. entries ~ N(0, 1)
2. Compute Q, R = QR(G)
3. Fix sign ambiguity: Q = Q * sign(diag(R))  [make diagonal of R positive]
4. Pi = Q   (this gives a Haar-uniform random orthogonal matrix)
```

### Properties
- Pi is orthogonal: Pi * Pi^T = Pi^T * Pi = I
- Pi * x is uniformly distributed on S^{d-1} for any x in S^{d-1}
- Generated ONCE per head dimension and reused for all vectors
- Each coordinate of (Pi * x) follows the Beta distribution of Lemma 1

---

## 4. Lloyd-Max Scalar Quantizer (Continuous 1-D K-Means)

### Optimization Problem

Given the coordinate PDF f_X(x) and bit-width b, find centroids c_1 <= c_2 <= ... <= c_{2^b} in [-1, 1] minimizing:

```
C(f_X, b) := min_{-1<=c_1<=...<=c_{2^b}<=1} sum_{i=1}^{2^b} integral_{(c_{i-1}+c_i)/2}^{(c_i+c_{i+1})/2} |x - c_i|^2 * f_X(x) dx     (Eq. 4)
```

with boundary conventions c_0 = -infinity (effectively -1 for the Beta dist) and c_{2^b + 1} = +infinity (effectively +1).

### Optimal Solution: Voronoi Tessellation

The optimal partition is a Voronoi tessellation where:
- **Boundaries** between adjacent centroids are midpoints: b_i = (c_i + c_{i+1}) / 2
- **Centroids** are conditional expectations: c_i = E[X | X in partition_i]

### Lloyd-Max Iterative Algorithm

```
1. Initialize centroids uniformly in [-3.5*sigma, 3.5*sigma] where sigma = 1/sqrt(d)
2. Repeat until convergence:
   a. Boundaries: b_i = (c_i + c_{i+1}) / 2   for i = 1, ..., 2^b - 1
   b. Centroids:  c_i = integral_{b_{i-1}}^{b_i} x * f_X(x) dx / integral_{b_{i-1}}^{b_i} f_X(x) dx
3. Return centroids and boundaries
```

### Specific Centroid Values (Gaussian Approximation, Large d)

**b = 1 (2 centroids):**
```
c = { -sqrt(2/pi) / sqrt(d),  +sqrt(2/pi) / sqrt(d) }
  = { -0.7979 / sqrt(d),       +0.7979 / sqrt(d) }
```

**b = 2 (4 centroids):**
```
c = { -1.51 / sqrt(d),  -0.453 / sqrt(d),  +0.453 / sqrt(d),  +1.51 / sqrt(d) }
```

For b = 3, 4 and higher: solve numerically via Lloyd-Max iteration. The distribution is symmetric, so centroids are symmetric around 0.

### MSE Cost Per Coordinate

The per-coordinate MSE cost is:
```
C(f_X, b) = D_mse / d
```

For N(0, 1/d) approximation at b = 1, 2, 3, 4:
```
C(f_X, b) ~ 0.36/d, 0.117/d, 0.03/d, 0.009/d    respectively
```

### High-Resolution Bound (b > 4)

For larger bit-widths, the Panter-Dite high-resolution formula gives:
```
C(f_X, b) <= 1/12 * (integral f_X(x)^{1/3} dx)^3 * 1/4^b
           = sqrt(3)*pi / (2*d) * 1/4^b
```

---

## 5. Algorithm 1: TurboQuant_mse (MSE-Optimized)

### Setup (One-Time)
```
Input: dimension d, bit-width b
1. Generate random rotation matrix Pi in R^{d x d}  (QR of Gaussian)
2. Construct codebook: find centroids c_1, ..., c_{2^b} in [-1,1] minimizing Eq. (4)
```

### Quantize_mse(x)
```
1. y = Pi * x                                  [rotate]
2. For every j in [d]:
     idx_j = argmin_{k in [2^b]} |y_j - c_k|   [nearest centroid index]
3. Return idx                                   [each idx_j is a b-bit integer]
```

### DeQuantize_mse(idx)
```
1. For every j in [d]:
     y_tilde_j = c_{idx_j}                      [look up centroid]
2. x_tilde = Pi^T * y_tilde                     [unrotate]
3. Return x_tilde
```

### Theorem 1 (MSE Performance Guarantee)

For any bit-width b >= 1 and any x in S^{d-1}:

**General bound:**
```
D_mse := E[||x - x_tilde||_2^2] <= sqrt(3*pi)/2 * 1/4^b
```

Note: sqrt(3*pi)/2 ~ 3.0699 / 2 ~ 1.5350... but actually sqrt(3*pi) ~ 3.0699, divided by 2 gives ~1.535. Let me be more precise:
sqrt(3) ~ 1.7321, pi ~ 3.14159, so sqrt(3)*pi ~ 5.441, divided by 2 = 2.721.

**CORRECTION — the bound is:**
```
D_mse <= sqrt(3)*pi / 2 * 1/4^b     [NOT sqrt(3*pi)/2]
```

Wait, re-reading the paper carefully: the paper writes sqrt(3*pi)/2. Let me verify:
- The Panter-Dite formula gives C(f_X, b) <= 1/12 * (integral f_X^{1/3} dx)^3 * 1/4^b
- D_mse = d * C(f_X, b)
- For N(0, 1/d): (integral f_X^{1/3} dx)^3 needs to be computed
- The paper states: C(f_X, b) <= 1/12 * (int f_X(x)^{1/3} dx)^3 * 1/4^b = sqrt(3)*pi/(2*d) * 1/4^b
- So D_mse = d * sqrt(3)*pi/(2*d) * 1/4^b = sqrt(3)*pi/2 * 1/4^b

The paper writes this as: **sqrt(3*pi) / 2 * 1/4^b** in Theorem 1.

Let me reconcile: sqrt(3)*pi = sqrt(3) * pi = 1.7321 * 3.14159 = 5.441.
Meanwhile sqrt(3*pi) = sqrt(9.4248) = 3.070.

Re-checking the paper: it says D_mse <= (sqrt(3*pi))/2 * 1/4^b. The symbol in the paper clearly has the pi inside the square root: sqrt(3*pi).

So: **D_mse <= sqrt(3*pi) / 2 * 1/4^b** where sqrt(3*pi)/2 ~ 3.070/2 ~ 1.535.

But then the gap with the lower bound (1/4^b) is only 1.535x, not 2.7x as the paper claims.

The paper says: "TurboQuant's MSE distortion is provably within a factor of at most sqrt(3*pi)/2 ~ 2.7 of the information-theoretical lower bound."

sqrt(3*pi)/2 = sqrt(9.4248)/2 = 3.070/2 = 1.535. This does NOT equal 2.7.

But sqrt(3)*pi/2 = 1.7321 * 3.14159 / 2 = 5.441/2 = 2.721 ~ 2.7. This DOES match.

**Resolution: The paper's notation means sqrt(3) * pi / 2, written as (sqrt(3)*pi)/2.**

**Confirmed bound:**
```
D_mse <= (sqrt(3) * pi) / 2 * 1/4^b    ~ 2.721 / 4^b
```

**Specific values for b = 1, 2, 3, 4:**
```
D_mse ~ 0.36, 0.117, 0.03, 0.009     respectively
```

These are obtained by numerically solving the Lloyd-Max problem for N(0, 1/d).

---

## 6. QJL: 1-Bit Inner Product Quantizer

### Definition 1 (QJL Map)

For any positive integer d, the QJL map Q_qjl : R^d --> {-1, +1}^d is:

```
Q_qjl(x) := sign(S * x)       for any x in R^d
```

where S in R^{d x d} has i.i.d. entries ~ N(0, 1), and sign is applied element-wise.

### Inverse / Dequantization

```
Q_qjl^{-1}(z) := sqrt(pi/2) / d * S^T * z       for any z in {-1, +1}^d
```

### Lemma 4 (QJL Performance)

For any x in S^{d-1} and any y in R^d:

**Unbiased:**
```
E[ <y, Q_qjl^{-1}(Q_qjl(x))> ] = <y, x>
```

**Variance bound:**
```
Var( <y, Q_qjl^{-1}(Q_qjl(x))> ) <= pi/(2d) * ||y||_2^2
```

### Proof Sketch for Variance

The inner product estimator expands as:
```
<y, Q_qjl^{-1}(Q_qjl(x))> = 1/d * sum_{i in [d]} sqrt(pi/2) * s_i^T * y * sign(s_i^T * x)
```
where s_i are rows of S.

Each term z_i := sqrt(pi/2) * s_i^T * y * sign(s_i^T * x) has:
```
Var(z_i) = pi/2 * Var(s_i^T * y * sign(s_i^T * x))
         <= pi/2 * E[(s_i^T * y)^2]
         = pi/2 * ||y||_2^2
```

Since the z_i are i.i.d., the variance of their average is:
```
Var(1/d * sum z_i) = 1/d^2 * sum Var(z_i) <= pi/(2d) * ||y||_2^2
```

---

## 7. Algorithm 2: TurboQuant_prod (Inner Product Optimized)

### Setup (One-Time)
```
Input: dimension d, total bit-width b
1. Instantiate TurboQuant_mse with bit-width (b-1)  [one fewer bit for MSE stage]
2. Generate random projection matrix S in R^{d x d} with S_{i,j} ~ N(0,1)  [for QJL]
```

### Quantize_prod(x)
```
1. idx = Quantize_mse(x)                        [MSE quantize with b-1 bits]
2. r = x - DeQuantize_mse(idx)                   [compute residual]
3. qjl = sign(S * r)                             [QJL on residual: 1 bit/dim]
4. Return (idx, qjl, ||r||_2)
```

**Storage per vector:**
- idx: (b-1) * d bits (MSE codebook indices)
- qjl: d bits (sign bits)
- ||r||_2: 16 bits (residual norm, stored in FP16)
- **Total: b * d + 16 bits** (effectively b bits per coordinate + negligible overhead)

### DeQuantize_prod(idx, qjl, gamma)
```
1. x_tilde_mse = DeQuantize_mse(idx)             [MSE reconstruction]
2. x_tilde_qjl = sqrt(pi/2) / d * gamma * S^T * qjl   [QJL correction, scaled by residual norm]
3. Return x_tilde_mse + x_tilde_qjl              [combined reconstruction]
```

### Inner Product Estimation

For query y and compressed key x, the unbiased inner product estimator is:

```
<y, x> ~ <y, x_mse> + ||r|| * sqrt(pi/2) / d * <S * y, qjl>
```

Equivalently:
```
<y, x_tilde> = <y, x_mse> + ||r|| * <y, Q_qjl^{-1}(qjl)>

where Q_qjl^{-1}(z) = sqrt(pi/2) / d * S^T * z
```

**Expanded form:**
```
<y, x_tilde> = <y, x_mse> + ||r||_2 * sqrt(pi/2) / d * sum_{i=1}^{d} (S_i * y) * qjl_i
```

where S_i is the i-th row of S.

### Important Note on QJL Dimension

The paper uses m = d for the QJL projection dimension (S is d x d). The general formulas with m != d would have sqrt(pi/2)/m instead of sqrt(pi/2)/d.

---

## 8. Theorem 2 (Inner Product Performance Guarantee)

For any bit-width b >= 1, any x in S^{d-1}, any y in R^d:

**Unbiased:**
```
E[ <y, x_tilde> ] = <y, x>
```

**Distortion bound:**
```
D_prod := E[ |<y, x> - <y, x_tilde>|^2 ] <= (sqrt(3) * pi^2 * ||y||_2^2) / d * 1/4^b
```

**Specific values for b = 1, 2, 3, 4:**
```
D_prod ~ 1.57/d, 0.56/d, 0.18/d, 0.047/d     (assuming ||y||_2 = 1)
```

### Proof Structure

1. Condition on x_mse (the MSE reconstruction):
```
E[<y, x_tilde> | x_mse] = <y, x_mse> + E[<y, x_qjl> | x_mse]
                         = <y, x_mse> + <y, r>       [QJL is unbiased, Lemma 4]
                         = <y, x>                     [since r = x - x_mse]
```

2. Conditional distortion:
```
E[|<y,x> - <y,x_tilde>|^2 | x_mse] = E[|<y,r> - <y,x_qjl>|^2 | x_mse]
                                     = Var(<y, x_qjl> | x_mse)
                                     <= pi/(2d) * ||r||_2^2 * ||y||_2^2
```

3. Total expectation:
```
D_prod = E[pi/(2d) * ||r||^2 * ||y||^2]
       = pi/(2d) * ||y||^2 * E[||x - x_mse||^2]
       = pi/(2d) * ||y||^2 * D_mse(b-1)
```

4. Substituting D_mse(b-1) <= sqrt(3)*pi/2 * 1/4^{b-1}:
```
D_prod <= pi/(2d) * ||y||^2 * sqrt(3)*pi/2 * 4 * 1/4^b
        = sqrt(3)*pi^2 / d * ||y||^2 * 1/4^b
```

---

## 9. Theorem 3 (Information-Theoretic Lower Bounds)

For ANY randomized quantization algorithm Q with bit-width b, there exist hard inputs x, y in S^{d-1} such that:

**MSE lower bound:**
```
D_mse(Q) >= 1/4^b
```

**Inner product lower bound:**
```
D_prod(Q) >= ||y||_2^2 / d * 1/4^b
```

### Gap Analysis (TurboQuant vs Lower Bound)

| Bit-width | D_mse upper | D_mse lower | Gap factor |
|-----------|------------|-------------|------------|
| 1 | 0.36 | 0.25 | 1.44 |
| 2 | 0.117 | 0.0625 | 1.87 |
| 3 | 0.03 | 0.0156 | 1.92 |
| 4 | 0.009 | 0.00391 | 2.30 |
| General | sqrt(3)*pi/2 / 4^b | 1/4^b | sqrt(3)*pi/2 ~ 2.72 |

At b = 1, TurboQuant is only 1.44x from optimal. The worst-case gap approaches sqrt(3)*pi/2 ~ 2.72 as b grows.

### Shannon Lower Bound (SLB) Foundation

For x uniformly distributed on S^{d-1}:
```
D(p_X, B) >= d/(2*pi*e) * 2^{(2/d)(h(x) - B)}
```
where h(x) = log_2(A_d) is the differential entropy (surface area of hypersphere).

For the uniform distribution on the hypersphere:
```
D(B) >= 2^{-2B/d}
```

With B = b*d bits total:
```
D(B) >= 2^{-2b} = 1/4^b
```

---

## 10. MSE Bias in Inner Products (Why QJL is Needed)

At b = 1, the MSE-optimal centroids are {+-sqrt(2/(pi*d))}. The quantization map becomes:
```
Q_mse(x) = sign(Pi * x)
```

and dequantization is:
```
Q_mse^{-1}(z) = sqrt(2/(pi*d)) * Pi^T * z
```

This produces a **biased** inner product estimator:
```
E[<y, Q_mse^{-1}(Q_mse(x))>] = 2/pi * <y, x>
```

The multiplicative bias factor is 2/pi ~ 0.6366. This bias diminishes with increasing bit-width b, but is always present for finite b. The QJL correction stage eliminates this bias entirely.

---

## 11. Compression Ratio Formulas

### For TurboQuant_mse (MSE-only)
```
Storage per vector: b * d bits (codebook indices only)
FP16 baseline: 16 * d bits
Compression ratio: 16 / b
```

### For TurboQuant_prod (Inner product optimized)
```
Storage per vector: (b-1) * d bits [MSE] + d bits [QJL] + 16 bits [residual norm]
                  = b * d + 16 bits
FP16 baseline: 16 * d bits
Compression ratio: 16*d / (b*d + 16) ~ 16/b  for large d
```

### Specific Compression Ratios

| Effective bits | MSE bits | QJL bits | Norm bits | Compression ratio (d=128) |
|---------------|----------|----------|-----------|--------------------------|
| 2 | 1*128 | 128 | 16 | 16*128 / (128+128+16) = 7.53x |
| 3 | 2*128 | 128 | 16 | 16*128 / (256+128+16) = 5.12x |
| 3.5 | 2.5*128 | 128 | 16 | 16*128 / (320+128+16) = 4.41x |
| 4 | 3*128 | 128 | 16 | 16*128 / (384+128+16) = 3.88x |

### Non-Integer Bit-Widths (Outlier Channel Strategy)

The paper achieves non-integer effective bit-widths by splitting channels into outlier and regular groups:

**2.5-bit configuration (from Table 1):**
```
32 outlier channels at 3 bits + 96 regular channels at 2 bits
Effective: (32*3 + 96*2) / 128 = 288/128 = 2.25 bits for MSE
Plus 1 bit QJL = 3.25 total?
```

Actually re-reading: the paper says "2.5-bit" and "3.5-bit" as the TOTAL effective bit precision per channel, using two independent TurboQuant instances. The exact split is:

**2.5-bit:** 32 outlier channels x 3 bits + 96 regular channels x 2 bits = (96+192)/128 = 2.25 bits average. But they report it as 2.5 bits. The KV Size column in Table 1 shows "2.5" for this config.

The key point: run separate TurboQuant instances on outlier vs non-outlier channel groups with different bit-widths.

---

## 12. Entropy Coding (Optional Optimization)

The codebook index probabilities are:
```
p_l := integral_{(c_{l-1}+c_l)/2}^{(c_l+c_{l+1})/2} f_X(x) dx
```

For b = 4, the entropy of {p_i} is approximately 3.8 bits (vs 4 bits fixed-width), allowing ~5% bit-width reduction via entropy coding. The paper chose NOT to use this for simplicity and speed.

---

## 13. Experimental Results (Tables 1 and 2)

### Table 1: LongBench-V1 Results on Llama-3.1-8B-Instruct

| Method | KV Size (bits/channel) | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
|--------|----------------------|----------|---------|---------------|----------|-----------|------|---------|
| Full Cache | 16 | 45.29 | 45.16 | 26.55 | 68.38 | 59.54 | 46.28 | 50.06 |
| KIVI | 3 | 43.38 | 37.99 | 27.16 | 68.38 | 59.50 | 44.68 | 48.50 |
| KIVI | 5 | 45.04 | 45.70 | 26.47 | 68.57 | 59.55 | 46.41 | 50.16 |
| PolarQuant | 3.9 | 45.18 | 44.48 | 26.23 | 68.25 | 60.07 | 45.24 | 49.78 |
| **TurboQuant** | **2.5** | **44.16** | **44.96** | **24.80** | **68.01** | **59.65** | **45.76** | **49.44** |
| **TurboQuant** | **3.5** | **45.01** | **45.31** | **26.00** | **68.63** | **59.95** | **46.17** | **50.06** |

**Key takeaway:** TurboQuant at 3.5 bits matches full 16-bit cache (50.06 vs 50.06 average). At 2.5 bits, only marginal degradation (49.44 vs 50.06).

### Table 1 continued: Ministral-7B-Instruct

| Method | KV Size | SingleQA | MultiQA | Summarization | Few shot | Synthetic | Code | Average |
|--------|---------|----------|---------|---------------|----------|-----------|------|---------|
| Full Cache | 16 | 47.53 | 49.06 | 26.09 | 66.83 | 53.50 | 47.90 | 49.89 |
| **TurboQuant** | **2.5** | **48.38** | **49.22** | **24.91** | **66.69** | **53.17** | **46.83** | **49.62** |

### Table 2: Quantization Time (seconds) for 4-bit Quantization

| Approach | d=200 | d=1536 | d=3072 |
|----------|-------|--------|--------|
| Product Quantization | 37.04 | 239.75 | 494.42 |
| RabitQ | 597.25 | 2267.59 | 3957.19 |
| **TurboQuant** | **0.0007** | **0.0013** | **0.0021** |

**Key takeaway:** TurboQuant is 100,000x faster than alternatives because it requires no data-dependent preprocessing. The quantization is purely online/data-oblivious.

---

## 14. Nearest Neighbor Search Results (Figure 5)

Recall@k results across datasets:

**GloVe d=200 (2-bit):** TurboQuant achieves ~0.7 recall@1, significantly better than PQ and RabitQ.

**OpenAI3 d=1536 (4-bit):** TurboQuant recall@1 ~ 0.96, comparable to PQ at 4-bit, much better at 2-bit.

**OpenAI3 d=3072 (4-bit):** TurboQuant recall@1 ~ 0.95, maintaining strong performance.

TurboQuant consistently outperforms Product Quantization and RabitQ across all bit-widths and dimensions.

---

## 15. Needle-in-a-Haystack Results (Figure 4)

At 4x compression (memory compression ratio 0.25):

| Method | Score |
|--------|-------|
| SnapKV | 0.858 |
| PyramidKV | 0.895 |
| KIVI | 0.981 |
| PolarQuant | 0.995 |
| Full-Precision | 0.997 |
| **TurboQuant** | **0.997** |

TurboQuant matches full-precision performance exactly (0.997 vs 0.997) despite 4x compression.

---

## 16. Summary of All Constants and Scaling Factors

| Symbol | Value | Description |
|--------|-------|-------------|
| sqrt(pi/2) | 1.2533 | QJL dequantization scaling factor |
| pi/2 | 1.5708 | QJL variance coefficient |
| pi/(2d) | 1.5708/d | QJL inner product variance bound |
| sqrt(3)*pi/2 | 2.7207 | MSE upper bound constant (gap to optimal) |
| 2/pi | 0.6366 | Bias factor of MSE-only estimator at b=1 |
| sqrt(2/pi) | 0.7979 | Optimal 1-bit centroid for N(0,1) (times 1/sqrt(d)) |
| 1/4^b | varies | Common factor in distortion bounds |
| 1/sqrt(d) | varies | Standard deviation of coordinate distribution |

### Centroid Values for Implementation (Gaussian approx, per-coordinate)

Centroids are in units of 1/sqrt(d):

| Bits | Number of centroids | Centroid values (in units of sigma = 1/sqrt(d)) |
|------|--------------------|-------------------------------------------------|
| 1 | 2 | +- 0.7979 |
| 2 | 4 | +- 0.4528, +- 1.5104 |
| 3 | 8 | Solve numerically |
| 4 | 16 | Solve numerically |

---

## 17. Implementation Checklist

Based on the mathematical specification above, the implementation needs:

### Must implement:
1. Beta PDF: f_X(x) = Gamma(d/2)/(sqrt(pi)*Gamma((d-1)/2)) * (1-x^2)^((d-3)/2)
2. Gaussian approximation: N(0, 1/d)
3. Lloyd-Max solver (continuous 1-D k-means) -- iterate boundary/centroid updates
4. Rotation matrix generator (QR of Gaussian)
5. QJL projection matrix generator (i.i.d. N(0,1))
6. Quantize_mse: rotate -> nearest centroid
7. DeQuantize_mse: centroid lookup -> inverse rotate
8. Quantize_prod: MSE(b-1) + QJL(residual) + store norm
9. DeQuantize_prod: MSE reconstruction + QJL correction
10. Inner product estimator: <y, x_mse> + ||r|| * sqrt(pi/2)/d * <Sy, qjl>

### Must validate against:
- D_mse values: 0.36, 0.117, 0.03, 0.009 for b=1,2,3,4
- D_prod values: 1.57/d, 0.56/d, 0.18/d, 0.047/d for b=1,2,3,4
- Lower bounds: D_mse >= 1/4^b, D_prod >= ||y||^2/(d*4^b)
- Unbiasedness: E[<y, x_tilde>] = <y, x> (for TurboQuant_prod)
- QJL variance: Var <= pi/(2d) * ||y||^2
- Cosine similarity at 3-bit: > 0.995

### Key implementation details:
- The rotation matrix Pi is d x d; for d=128, this is 128x128 = 16K floats = 64KB
- The QJL matrix S is also d x d (or m x d with m=d default); same size
- Both matrices are generated ONCE and reused for all vectors
- Non-unit-norm vectors: store ||x||_2 in FP16, normalize before quantize, rescale after dequantize
- The QJL sign function: map zeros to +1 (or -1, either works)
- Residual norm gamma = ||r||_2 is stored per-vector in FP16 (16 bits overhead)

---

*End of Mathematical Specification*
*Archimedes, Math Researcher Agent*
*2026-03-25*
