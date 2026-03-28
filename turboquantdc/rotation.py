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


# ---------------------------------------------------------------------------
# Walsh-Hadamard Transform (O(d log d) rotation)
# ---------------------------------------------------------------------------


def fast_wht(x: torch.Tensor) -> torch.Tensor:
    """Apply the unnormalized Walsh-Hadamard Transform in-place along last dim.

    Uses the iterative butterfly algorithm: O(d log d) operations.
    Works on batched input: x has shape (..., d) where d must be power of 2.

    Args:
        x: Input tensor, last dimension must be power of 2.

    Returns:
        WHT of x along last dimension (unnormalized — divide by sqrt(d) for orthogonal).
    """
    d = x.shape[-1]
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"

    h = 1
    while h < d:
        # View as pairs: (..., d//(2*h), 2, h) then butterfly
        xe = x.view(*x.shape[:-1], -1, 2, h)
        a = xe[..., 0, :]
        b = xe[..., 1, :]
        s = a + b
        diff = a - b
        xe[..., 0, :] = s
        xe[..., 1, :] = diff
        h *= 2

    return x


def generate_wht_rotation(
    d: int,
    seed: int | None = None,
    device: str | torch.device = "cpu",
) -> dict:
    """Generate a randomized Walsh-Hadamard rotation.

    Returns parameters for the rotation Pi = D * H_d / sqrt(d):
    - signs: random +/-1 diagonal (d,)
    - d: dimension

    Unlike generate_rotation_matrix() which returns a d x d dense matrix,
    this returns just the sign vector. The actual rotation is applied via
    apply_wht_rotation() which uses O(d log d) operations.

    Args:
        d: Dimension (must be power of 2).
        seed: Random seed for reproducibility.
        device: Target device.

    Returns:
        Dict with 'signs' tensor and metadata.
    """
    assert d > 0 and (d & (d - 1)) == 0, f"d must be power of 2, got {d}"

    gen = torch.Generator(device="cpu")
    if seed is not None:
        gen.manual_seed(seed)

    signs = torch.where(
        torch.rand(d, generator=gen) < 0.5,
        torch.ones(d),
        -torch.ones(d),
    ).to(device)

    return {"signs": signs, "d": d}


def apply_wht_rotation(
    x: torch.Tensor,
    wht_params: dict,
    inverse: bool = False,
) -> torch.Tensor:
    """Apply randomized Walsh-Hadamard rotation to vectors.

    Forward:  y = (D * H_d / sqrt(d)) @ x = H_d(D @ x) / sqrt(d)
    Inverse:  x = (H_d * D / sqrt(d)) @ y = D @ H_d(y) / sqrt(d)

    (H_d is its own inverse up to scaling: H_d @ H_d = d * I)
    (D is its own inverse: D @ D = I)

    Args:
        x: Input tensor, shape (..., d).
        wht_params: Dict from generate_wht_rotation().
        inverse: If True, apply inverse rotation.

    Returns:
        Rotated tensor, same shape as x.
    """
    signs = wht_params["signs"]
    d = wht_params["d"]

    result = x.clone()

    if not inverse:
        # Forward: multiply by D first, then WHT
        result = result * signs
        result = fast_wht(result)
    else:
        # Inverse: WHT first, then multiply by D
        result = fast_wht(result)
        result = result * signs

    # Normalize (H_d @ H_d = d*I, so single application needs 1/sqrt(d))
    result = result / (d ** 0.5)

    return result
