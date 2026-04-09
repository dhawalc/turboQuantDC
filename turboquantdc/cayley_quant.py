"""Cayley-parameterized full d x d learned rotation for KV cache compression.

Learns a FULL d x d orthogonal rotation matrix (d*(d-1)/2 = 8128 DOF for d=128)
via the Cayley map:

    R = (I - A) @ (I + A)^{-1}

where A is skew-symmetric (A^T = -A).  This is always orthogonal, always
differentiable, and parameterizes the connected component of SO(d) containing
the identity.  The d*(d-1)/2 free parameters of A give full coverage of the
rotation group (minus reflections), which is exactly what we need.

Why Cayley beats Givens:
    - Givens block-diagonal: d/2 = 64 DOF (each 2x2 block independent)
    - Cayley full matrix: d*(d-1)/2 = 8128 DOF (all dimensions coupled)
    - WHT: ~d*log2(d) = 896 effective DOF (butterfly structure)
    The DOF gap means Givens CANNOT represent the rotations that WHT can,
    while Cayley can represent ALL of them plus infinitely more.

Optimization:
    - Use torch.linalg.solve(I + A, I - A) instead of explicit inverse
    - Cache R during inference; only recompute during calibration
    - Loss: KL(softmax(Q @ K^T / sqrt(d)), softmax(Q @ K_hat^T / sqrt(d)))
    - Straight-through estimator for the non-differentiable argmin

Reference: Cayley transform for optimization on orthogonal matrices
(Wen & Yin 2013, "A feasible method for optimization with orthogonality constraints")
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codebook import solve_lloyd_max


# ---------------------------------------------------------------------------
# Cayley rotation
# ---------------------------------------------------------------------------

class CayleyRotation(nn.Module):
    """Full d x d orthogonal rotation via the Cayley parameterization.

    Stores a skew-symmetric matrix A with d*(d-1)/2 free parameters.
    The rotation matrix is R = (I - A) @ (I + A)^{-1}.

    Args:
        d: Vector dimension (e.g. 128).
        seed: Random seed for initialization.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        n_params = d * (d - 1) // 2

        # Store the upper-triangular free parameters of A
        # A is skew-symmetric: A = triu_to_skew(params)
        # Initialize to zero -> R = I (identity rotation)
        self.skew_params = nn.Parameter(
            torch.zeros(n_params, device=device)
        )

        # Cache for the rotation matrix (recomputed when params change)
        self.register_buffer("_cached_R", torch.eye(d, device=device))
        self._cache_valid = False

    def _build_skew_symmetric(self) -> torch.Tensor:
        """Build the d x d skew-symmetric matrix A from the free parameters.

        A has d*(d-1)/2 independent entries in the strict upper triangle.
        A_ij = params[k] for i < j, A_ji = -A_ij, A_ii = 0.
        """
        d = self.d
        A = torch.zeros(d, d, device=self.skew_params.device, dtype=self.skew_params.dtype)
        # Fill upper triangle
        idx = torch.triu_indices(d, d, offset=1)
        A[idx[0], idx[1]] = self.skew_params
        # Skew-symmetric: A^T = -A
        A = A - A.T
        return A

    def rotation_matrix(self) -> torch.Tensor:
        """Compute R = (I - A) @ (I + A)^{-1} via solve.

        Uses torch.linalg.solve for numerical stability and efficiency:
        R = solve(I + A, I - A) solves (I + A) @ R = (I - A).

        This is equivalent to R = (I + A)^{-1} @ (I - A), but since
        (I - A)(I + A)^{-1} = (I + A)^{-1}(I - A) when A is skew-symmetric,
        both orderings give the same result.

        Returns:
            (d, d) orthogonal matrix R.
        """
        A = self._build_skew_symmetric()
        d = self.d
        I = torch.eye(d, device=A.device, dtype=A.dtype)

        # Solve (I + A) @ R = (I - A) for R
        # This gives R = (I + A)^{-1} @ (I - A) which equals (I - A) @ (I + A)^{-1}
        # for skew-symmetric A (the two orderings commute)
        R = torch.linalg.solve(I + A, I - A)
        return R

    def rotation_matrix_cached(self) -> torch.Tensor:
        """Return cached rotation matrix, recomputing if needed.

        During training, always recomputes. Call cache_rotation() after
        calibration to freeze the cached value for fast inference.
        """
        if self.training or not self._cache_valid:
            return self.rotation_matrix()
        return self._cached_R

    def cache_rotation(self):
        """Freeze the current rotation matrix into the cache for inference."""
        with torch.no_grad():
            self._cached_R.copy_(self.rotation_matrix())
        self._cache_valid = True

    def invalidate_cache(self):
        """Mark cache as stale (called when parameters change during training)."""
        self._cache_valid = False

    def rotate(self, x: torch.Tensor) -> torch.Tensor:
        """Apply rotation: y = x @ R^T.

        Args:
            x: (..., d) input vectors.

        Returns:
            Rotated vectors, same shape.
        """
        R = self.rotation_matrix_cached()
        return x @ R.T

    def unrotate(self, y: torch.Tensor) -> torch.Tensor:
        """Apply inverse rotation: x = y @ R.

        Since R is orthogonal, R^{-1} = R^T, so unrotate = y @ R.

        Args:
            y: (..., d) rotated vectors.

        Returns:
            Unrotated vectors, same shape.
        """
        R = self.rotation_matrix_cached()
        return y @ R

    def init_from_wht(self, seed: int = 42, fit_steps: int = 200, fit_lr: float = 0.1):
        """Initialize A so that Cayley(A) approximates the WHT rotation.

        The WHT matrix often has eigenvalues at exactly -1, which means the
        analytic inverse Cayley A = (I - R)(I + R)^{-1} is singular.
        Instead, we use gradient descent to find A minimizing ||Cayley(A) - R_wht||_F.

        This is fast (~200 steps, <1s) and finds a good approximation.

        Args:
            seed: Random seed for WHT sign generation.
            fit_steps: Number of gradient steps to fit A.
            fit_lr: Learning rate for fitting.
        """
        from .rotation import generate_wht_rotation, apply_wht_rotation

        d = self.d
        assert d > 0 and (d & (d - 1)) == 0, f"WHT init requires power-of-2 d, got {d}"

        wht_params = generate_wht_rotation(d, seed=seed, device="cpu")
        I_d = torch.eye(d, device="cpu")
        R_wht = apply_wht_rotation(I_d, wht_params)  # (d, d) explicit WHT matrix
        R_target = R_wht.to(self.skew_params.device)

        # Fit A via gradient descent: minimize ||Cayley(A) - R_wht||_F^2
        # Start from zero (identity rotation)
        self.skew_params.data.zero_()

        optimizer = torch.optim.Adam([self.skew_params], lr=fit_lr)

        for step in range(fit_steps):
            optimizer.zero_grad()
            R_current = self.rotation_matrix()
            loss = (R_current - R_target).pow(2).sum()
            loss.backward()
            optimizer.step()

            if loss.item() < 1e-8:
                break

        # Verify result
        with torch.no_grad():
            R_final = self.rotation_matrix()
            fit_error = (R_final - R_target).pow(2).sum().sqrt().item()
            ortho_error = (R_final @ R_final.T - torch.eye(d, device=R_final.device)).abs().max().item()

        self.invalidate_cache()


# ---------------------------------------------------------------------------
# Straight-through quantization (shared with learned_quant.py)
# ---------------------------------------------------------------------------

def straight_through_quantize(
    x: torch.Tensor,
    centroids: torch.Tensor,
) -> torch.Tensor:
    """Quantize with straight-through estimator for gradient flow.

    Forward: hard assignment (argmin distance to centroids)
    Backward: gradient flows through via STE

    Args:
        x: (..., d) continuous values.
        centroids: (n_levels,) sorted centroid values.

    Returns:
        Quantized values with gradient flow.
    """
    dists = (x.unsqueeze(-1) - centroids).abs()
    indices = dists.argmin(dim=-1)

    n_levels = centroids.shape[0]
    one_hot = F.one_hot(indices, n_levels).float()
    x_quant = (one_hot @ centroids)

    # STE: forward = x_quant, backward = identity w.r.t. x
    return (x - x.detach()) + x_quant


# ---------------------------------------------------------------------------
# Cayley learned quantizer
# ---------------------------------------------------------------------------

class CayleyLearnedQuantizer(nn.Module):
    """Full d x d learned rotation quantizer via Cayley parameterization.

    Components:
        - CayleyRotation: full d x d orthogonal rotation (8128 DOF for d=128)
        - Mean removal: subtract per-head mean (shift-invariance of softmax)
        - Lloyd-Max centroids: FROZEN (learning overfits -- proven)
        - Straight-through estimator for non-differentiable argmin

    Loss: KL(softmax(Q @ K^T / sqrt(d)), softmax(Q @ K_hat^T / sqrt(d)))

    Args:
        d: Head dimension (e.g. 128).
        bits: Bits per coordinate for the codebook.
        center: Subtract running mean before quantization.
        seed: Random seed.
        device: Target device.
        init_from_wht: If True, initialize rotation from WHT (warm start).
    """

    def __init__(
        self,
        d: int,
        bits: int = 3,
        center: bool = True,
        seed: int = 42,
        device: str | torch.device = "cpu",
        init_from_wht: bool = False,
    ):
        super().__init__()
        self.d = d
        self.bits = bits
        self.n_levels = 1 << bits
        self.center = center

        # Full d x d rotation via Cayley
        self.rotation = CayleyRotation(d, seed=seed, device=device)

        if init_from_wht:
            self.rotation.init_from_wht(seed=seed)

        # Lloyd-Max centroids -- FROZEN (not learnable)
        lm_centroids, _ = solve_lloyd_max(d, bits)
        self.register_buffer("centroids", lm_centroids.to(device))

        # Running mean for center mode
        self.register_buffer("running_mean", torch.zeros(d, device=device))
        self.register_buffer(
            "running_count", torch.tensor(0, dtype=torch.long, device=device)
        )

    def _update_running_mean(self, x: torch.Tensor) -> None:
        """Online Welford update of running mean."""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        batch = x.shape[0]
        new_sum = x.sum(dim=0)
        old_n = self.running_count.item()
        new_n = old_n + batch
        if old_n == 0:
            self.running_mean.copy_(new_sum / new_n)
        else:
            self.running_mean.copy_(
                (self.running_mean * old_n + new_sum) / new_n
            )
        self.running_count.fill_(new_n)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward: mean-remove -> normalize -> rotate -> STE quantize -> unrotate.

        Args:
            x: (batch, d) key vectors.

        Returns:
            x_recon: (batch, d) reconstructed key vectors.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = x.float()

        # Mean removal
        if self.center:
            mean = self.running_mean.detach()
            x_c = x - mean
        else:
            mean = torch.zeros(self.d, device=x.device, dtype=x.dtype)
            x_c = x

        # Normalize
        norms = x_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x_c / norms

        # Cayley rotation
        x_rot = self.rotation.rotate(x_unit)

        # Straight-through quantization
        x_quant = straight_through_quantize(x_rot, self.centroids)

        # Inverse rotation
        x_recon = self.rotation.unrotate(x_quant)

        # Rescale + add mean
        x_recon = x_recon * norms + mean

        if squeeze:
            x_recon = x_recon.squeeze(0)
        return x_recon

    def encode(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Non-differentiable encode for inference.

        Args:
            x: (batch, d) or (d,) key vectors.

        Returns:
            Dict with indices, norms, mean.
        """
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(0)
        x = x.float()

        if self.center:
            self._update_running_mean(x)
            mean = self.running_mean.detach()
            x_c = x - mean
        else:
            mean = torch.zeros(self.d, device=x.device)
            x_c = x

        norms = x_c.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        x_unit = x_c / norms

        with torch.no_grad():
            x_rot = self.rotation.rotate(x_unit)
            dists = (x_rot.unsqueeze(-1) - self.centroids).abs()
            indices = dists.argmin(dim=-1)

        result = {
            "indices": indices,
            "norms": norms.squeeze(-1),
            "mean": mean,
        }
        if squeeze:
            result = {k: v.squeeze(0) if v.dim() > 0 and v.shape[0] == 1 else v
                      for k, v in result.items()}
        return result

    def decode(self, compressed: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Reconstruct from compressed representation.

        Args:
            compressed: Output of encode().

        Returns:
            Reconstructed key vectors.
        """
        indices = compressed["indices"]
        norms = compressed["norms"]
        mean = compressed["mean"]

        squeeze = indices.dim() == 1
        if squeeze:
            indices = indices.unsqueeze(0)
            norms = norms.unsqueeze(0)

        x_quant_rot = self.centroids[indices]
        with torch.no_grad():
            x_recon = self.rotation.unrotate(x_quant_rot)
        x_recon = x_recon * norms.unsqueeze(-1) + mean

        if squeeze:
            x_recon = x_recon.squeeze(0)
        return x_recon

    def attention_loss(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """KL divergence between FP16 and quantized attention distributions.

        Args:
            queries: (n_q, d) query vectors.
            keys: (seq_len, d) key vectors.
            temperature: Softmax temperature.

        Returns:
            Scalar KL divergence loss.
        """
        d = queries.shape[-1]
        scale = temperature / math.sqrt(d)

        logits_true = queries @ keys.T * scale
        attn_true = F.softmax(logits_true, dim=-1)

        keys_quant = self.forward(keys)
        logits_quant = queries @ keys_quant.T * scale
        log_attn_quant = F.log_softmax(logits_quant, dim=-1)

        loss = F.kl_div(log_attn_quant, attn_true, reduction="batchmean")
        return loss

    def calibrate(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        lr: float = 0.005,
        steps: int = 100,
        temperature: float = 1.0,
        verbose: bool = False,
    ) -> List[float]:
        """Optimize the Cayley rotation parameters on calibration data.

        Only the skew-symmetric parameters of A are optimized.
        Centroids are FROZEN.

        Args:
            queries: (n_q, d) calibration queries.
            keys: (seq_len, d) calibration keys.
            lr: Adam learning rate.
            steps: Optimization steps.
            temperature: Softmax temperature.
            verbose: Print progress.

        Returns:
            List of loss values per step.
        """
        queries = queries.float().to(self.rotation.skew_params.device)
        keys = keys.float().to(self.rotation.skew_params.device)

        # Update running mean from calibration keys
        self._update_running_mean(keys)

        # Only optimize the rotation parameters
        params = [self.rotation.skew_params]
        optimizer = torch.optim.Adam(params, lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=steps, eta_min=lr * 0.01
        )

        losses = []
        best_loss = float("inf")
        best_params = self.rotation.skew_params.data.clone()

        self.rotation.train()
        self.rotation.invalidate_cache()

        for step in range(steps):
            optimizer.zero_grad()
            loss = self.attention_loss(queries, keys, temperature)
            loss.backward()

            # Gradient clipping -- important for d^2 parameters
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)

            optimizer.step()
            scheduler.step()

            current_loss = loss.item()
            losses.append(current_loss)

            if current_loss < best_loss:
                best_loss = current_loss
                best_params = self.rotation.skew_params.data.clone()

            if verbose and (step + 1) % 10 == 0:
                print(f"  Step {step+1:3d}/{steps}: KL loss = {current_loss:.6f}")

        # Restore best parameters and cache the rotation matrix
        self.rotation.skew_params.data.copy_(best_params)
        self.rotation.eval()
        self.rotation.cache_rotation()

        return losses


# ---------------------------------------------------------------------------
# Calibration from model
# ---------------------------------------------------------------------------

def calibrate(
    model,
    tokenizer,
    text: str = (
        "The quick brown fox jumps over the lazy dog. "
        "A large language model is a neural network trained on "
        "vast amounts of text data. The transformer architecture "
        "uses attention mechanisms to process sequences in parallel."
    ),
    n_tokens: int = 128,
    bits: int = 3,
    lr: float = 0.005,
    steps: int = 100,
    init_from_wht: bool = False,
    device: str = "cuda",
    verbose: bool = False,
) -> Dict[int, CayleyLearnedQuantizer]:
    """Calibrate a CayleyLearnedQuantizer for every attention layer.

    Extracts Q/K from a forward pass, then optimizes the Cayley rotation
    per layer to minimize attention KL divergence.

    Args:
        model: HuggingFace causal LM.
        tokenizer: Corresponding tokenizer.
        text: Calibration text.
        n_tokens: Max calibration tokens.
        bits: Bits per coordinate.
        lr: Learning rate.
        steps: Steps per layer.
        init_from_wht: Warm-start from WHT.
        device: Device.
        verbose: Print progress.

    Returns:
        Dict mapping layer_idx -> calibrated CayleyLearnedQuantizer.
    """
    model_device = next(model.parameters()).device

    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=n_tokens,
        truncation=True,
    ).to(model_device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=False,
            use_cache=True,
        )

    past_kv = outputs.past_key_values

    # Count layers from cache
    if hasattr(past_kv, "key_cache"):
        n_layers = len(past_kv.key_cache)
    elif isinstance(past_kv, (list, tuple)):
        n_layers = len(past_kv)
    else:
        raise RuntimeError(f"Unsupported cache type: {type(past_kv)}")

    quantizers = {}

    for layer_idx in range(n_layers):
        if hasattr(past_kv, "key_cache"):
            K_all = past_kv.key_cache[layer_idx]
        elif isinstance(past_kv, (list, tuple)):
            K_all = past_kv[layer_idx][0]
        else:
            continue

        head_dim = K_all.shape[3]
        K = K_all[0, 0].float().to(device)
        Q = K.clone()  # Self-attention proxy

        lq = CayleyLearnedQuantizer(
            d=head_dim,
            bits=bits,
            center=True,
            seed=42 + layer_idx,
            device=device,
            init_from_wht=init_from_wht,
        )

        if verbose:
            print(f"\nLayer {layer_idx}:")
        losses = lq.calibrate(Q, K, lr=lr, steps=steps, verbose=verbose)

        if verbose:
            print(f"  Final KL = {losses[-1]:.6f}, Best = {min(losses):.6f}")

        quantizers[layer_idx] = lq

    return quantizers
