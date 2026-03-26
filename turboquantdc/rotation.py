"""Random orthogonal rotation and QJL projection matrix generation.

Implements:
    1. Haar-uniform random orthogonal matrix via QR decomposition of Gaussian
    2. Random Gaussian projection matrix for QJL (Stage 2)

The rotation matrix ensures that coordinates of Pi*x follow the concentrated
Beta distribution (Lemma 1), enabling optimal scalar quantization per coordinate.

The QJL projection matrix is used for the 1-bit bias correction stage.

Reference: TurboQuant paper (arxiv 2504.19874), Section 3 and Definition 1.
"""

from __future__ import annotations

import torch


def generate_rotation_matrix(
    d: int,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate a Haar-uniform random orthogonal matrix via QR decomposition.

    Algorithm (from paper Section 3):
        1. Sample G in R^{d x d} with i.i.d. entries ~ N(0, 1)
        2. Compute Q, R = QR(G)
        3. Fix sign ambiguity: Q = Q * sign(diag(R))
        4. Pi = Q  (Haar-uniform random orthogonal matrix)

    The sign correction (step 3) ensures uniqueness: without it, QR
    decomposition has an arbitrary sign choice on each column. Multiplying
    by sign(diag(R)) makes the diagonal of R positive, yielding a unique
    decomposition and a proper Haar-uniform sample.

    The generator runs on CPU for reproducibility, then the result is
    moved to the target device.

    Args:
        d: Dimension of the rotation matrix (d x d).
        seed: Random seed for reproducibility. None for non-deterministic.
        device: Target device for the output tensor.

    Returns:
        Orthogonal matrix Pi of shape (d, d) satisfying Pi @ Pi.T = I.
    """
    # Generate on CPU for reproducible seeding
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    # Step 1: Random Gaussian matrix
    G = torch.randn(d, d, generator=gen, device="cpu", dtype=torch.float32)

    # Step 2: QR decomposition
    Q, R = torch.linalg.qr(G)

    # Step 3: Fix sign ambiguity — make diagonal of R positive
    diag_sign = torch.sign(torch.diag(R))
    diag_sign[diag_sign == 0] = 1.0  # Handle exact zeros (extremely rare)
    Q = Q * diag_sign.unsqueeze(0)  # Broadcast: (d, d) * (1, d)

    return Q.to(device)


def generate_qjl_matrix(
    d: int,
    m: int | None = None,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Generate a random Gaussian projection matrix for QJL.

    The QJL (Quantized Johnson-Lindenstrauss) map uses a random matrix S
    with i.i.d. N(0, 1) entries to project residuals before taking signs.

    Per Definition 1: Q_qjl(x) = sign(S @ x), where S in R^{m x d}.

    The paper uses m = d (square matrix) as the default projection dimension.

    Args:
        d: Input dimension (head dimension).
        m: Projection dimension. Defaults to d if None.
        seed: Random seed for reproducibility. None for non-deterministic.
        device: Target device for the output tensor.

    Returns:
        Random Gaussian matrix S of shape (m, d).
    """
    if m is None:
        m = d

    # Generate on CPU for reproducible seeding
    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    S = torch.randn(m, d, generator=gen, device="cpu", dtype=torch.float32)

    return S.to(device)
