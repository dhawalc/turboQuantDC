"""QJL — Quantized Johnson-Lindenstrauss 1-bit bias correction.

Implements the QJL map (Definition 1) and inner product correction from Stage 2.

The QJL stage corrects the bias introduced by MSE quantization (Stage 1).
It projects the residual r = x - x_mse through a random Gaussian matrix S,
stores only the signs (1 bit per dimension), and uses these to produce an
unbiased inner product estimator.

Key equations:
    Q_qjl(r) = sign(S @ r)                               (Definition 1)
    Q_qjl^{-1}(z) = sqrt(pi/2) / m * S^T @ z             (Dequantization)
    correction = ||r|| * sqrt(pi/2) / m * <S @ q, signs>  (Inner product term)

Performance (Lemma 4):
    E[<y, Q_qjl^{-1}(Q_qjl(x))>] = <y, x>              (Unbiased)
    Var(...) <= pi/(2m) * ||y||^2                          (Variance bound)

Reference: TurboQuant paper (arxiv 2504.19874), Definition 1 / Lemma 4.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from .rotation import generate_qjl_matrix


class QJL(nn.Module):
    """Stage 2: 1-bit QJL bias correction for inner products.

    The QJL map projects vectors through a random Gaussian matrix and stores
    only the sign bits. Combined with the residual norm, this provides an
    unbiased correction to the MSE-only inner product estimate.

    Args:
        d: Input dimension (head dimension).
        m: Projection dimension. Defaults to d (paper's recommendation).
        seed: Random seed for projection matrix generation.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        m: int | None = None,
        seed: int = 42,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.d = d
        self.m = m if m is not None else d

        # Generate random Gaussian projection matrix S: (m, d)
        S = generate_qjl_matrix(d, m=self.m, seed=seed, device="cpu")
        self.register_buffer("S", S.to(device))

    def project_and_sign(self, residual: torch.Tensor) -> torch.Tensor:
        """Project residual through S and take element-wise sign.

        Computes: signs = sign(residual @ S.T)

        The sign mapping uses (x >= 0) * 2 - 1 instead of torch.sign to avoid
        the zero -> 0 behavior of torch.sign. Zeros are mapped to +1.

        Args:
            residual: Residual vectors r = x - x_mse, shape (batch, d) or (d,).

        Returns:
            Sign tensor of shape (batch, m) or (m,) with values in {-1, +1}.
        """
        # Project: (batch, d) @ (d, m) -> (batch, m)
        projected = residual @ self.S.T
        # Sign mapping: >= 0 -> +1, < 0 -> -1
        signs = (projected >= 0).float() * 2.0 - 1.0
        return signs

    def inner_product_correction(
        self,
        query: torch.Tensor,
        signs: torch.Tensor,
        residual_norm: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the QJL inner product correction term.

        The correction makes the overall estimator unbiased:
            correction = ||r|| * sqrt(pi/2) / m * <S @ q, signs>

        This term is added to the MSE inner product <q, k_mse> to get:
            <q, k> ~ <q, k_mse> + correction

        Args:
            query: Query vectors, shape (batch_q, d) or (d,).
            signs: Stored sign bits, shape (batch_k, m) or (m,).
            residual_norm: Norm of residual ||r||, shape (batch_k,) or scalar.

        Returns:
            Correction term, shape (batch_q, batch_k) or scalar.
        """
        scale = math.sqrt(math.pi / 2.0) / self.m

        # Project query through S: (batch_q, d) @ (d, m) -> (batch_q, m)
        q_projected = query @ self.S.T

        # Inner product of projected query with stored signs
        # If both are batched: (batch_q, m) @ (m, batch_k) -> (batch_q, batch_k)
        if signs.dim() == 1:
            # Single key: (batch_q, m) @ (m,) -> (batch_q,)
            qjl_ip = q_projected @ signs
        else:
            # Batched keys: (batch_q, m) @ (batch_k, m).T -> (batch_q, batch_k)
            qjl_ip = q_projected @ signs.T

        # Scale by residual norm and QJL constant
        # residual_norm shape: (batch_k,) -> broadcast with (batch_q, batch_k)
        return residual_norm * scale * qjl_ip
