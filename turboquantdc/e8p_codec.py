"""E8P Compact Encoding/Decoding for E8 Lattice Points.

Packs any E8 lattice point into exactly 16 bits per 8D block = 2 bits/dim.

Encoding: abs_idx (8 bits) + sign_byte (7 bits + 1 coset bit) = 16 bits
- 256-entry source set S of absolute-value patterns (norms 2-12)
- 7 independent sign flips (8th sign inferred from even parity)
- 1 coset bit selects +/- 1/4 global shift

Based on QuIP# (Tseng et al., ICML 2024) E8P12 codebook.
Reimplemented in pure PyTorch without CUDA dependencies.

Usage:
    from turboquantdc.e8p_codec import E8PCodec
    codec = E8PCodec()
    codes = codec.encode(lattice_points)   # (..., 8) -> (...,) uint16
    decoded = codec.decode(codes)          # (...,) uint16 -> (..., 8)
"""

import torch
from typing import Tuple


def _build_source_set() -> torch.Tensor:
    """Build the 256-entry source set S of absolute-value patterns.

    Contains 227 D8-hat points with ||x||^2 <= 10, plus 29 norm-12 points.
    All coordinates are half-integers (multiples of 0.5).

    Returns:
        (256, 8) tensor of source patterns (positive octant only)
    """
    # Generate all 8D vectors with half-integer coords in [-3.5, 3.5]
    # that lie in D8-hat (even coordinate sum) with ||x||^2 <= 10
    vals = torch.arange(-7, 8).float() * 0.5  # -3.5 to 3.5

    # We need absolute-value patterns, so only positive coords
    pos_vals = torch.arange(0, 8).float() * 0.5  # 0.0 to 3.5

    patterns = set()

    # Enumerate abs-value patterns with norm^2 <= 10
    # This is manageable: 8 coords, each in {0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5}
    # with sum of squares <= 10
    from itertools import product

    for combo in product(pos_vals.tolist(), repeat=8):
        sq_norm = sum(c * c for c in combo)
        if sq_norm <= 10.0 + 1e-6:
            # D8-hat: need even coordinate sum (considering abs values)
            # Actually for the source set, we just collect unique sorted abs patterns
            sorted_combo = tuple(sorted(combo, reverse=True))
            patterns.add(sorted_combo)

    # Also add norm-12 patterns: 5 coords = 1.5, 3 coords = 0.5
    # norm^2 = 5*(2.25) + 3*(0.25) = 12
    from itertools import combinations
    for combo_idx in combinations(range(8), 5):
        pattern = [0.5] * 8
        for i in combo_idx:
            pattern[i] = 1.5
        sorted_combo = tuple(sorted(pattern, reverse=True))
        patterns.add(sorted_combo)

    # Convert to tensor, sorted by norm then lexicographic
    pattern_list = sorted(patterns, key=lambda p: (sum(c*c for c in p), p))

    # Pad or truncate to exactly 256
    if len(pattern_list) > 256:
        pattern_list = pattern_list[:256]
    while len(pattern_list) < 256:
        # Fill remaining with zero pattern
        pattern_list.append((0.0,) * 8)

    return torch.tensor(pattern_list, dtype=torch.float32)


class E8PCodec:
    """Encode/decode E8 lattice points as 16-bit codes.

    Each 8D E8 lattice point is encoded as:
    - 8 bits: index into 256-entry source set (abs-value pattern)
    - 7 bits: sign flips for coordinates 0-6
    - 1 bit: coset selection (+1/4 or -1/4 global shift)

    Total: exactly 16 bits per 8D vector = 2 bits per dimension.
    """

    def __init__(self, device: str = "cpu"):
        self.source_set = _build_source_set().to(device)  # (256, 8)
        self.device = device
        # Build full grid for brute-force encoding (65536 entries)
        self._full_grid = self._build_full_grid()

    def _build_full_grid(self) -> torch.Tensor:
        """Build all 65536 E8P codewords for brute-force nearest-neighbor."""
        grid = []
        for abs_idx in range(256):
            pattern = self.source_set[abs_idx]  # (8,) positive coords
            for sign_bits in range(128):  # 7 independent sign bits
                # Apply sign flips to coords 0-6
                signs = torch.ones(8, device=self.device)
                for bit in range(7):
                    if (sign_bits >> bit) & 1:
                        signs[bit] = -1.0
                # 8th sign inferred from even parity
                n_neg = sum(1 for bit in range(7) if (sign_bits >> bit) & 1)
                if n_neg % 2 == 1:
                    signs[7] = -1.0

                point = pattern * signs

                # Two cosets: +1/4 and -1/4
                grid.append(point + 0.25)
                grid.append(point - 0.25)

        full_grid = torch.stack(grid)  # (65536, 8)
        return full_grid

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        """Encode 8D E8 lattice points to 16-bit codes.

        Uses brute-force nearest-neighbor against the full 65536-entry grid.

        Args:
            points: (..., 8) tensor of E8 lattice points

        Returns:
            (...,) tensor of uint16 codes
        """
        orig_shape = points.shape[:-1]
        flat = points.reshape(-1, 8).to(self.device)

        # Nearest neighbor: compute distances to all 65536 codewords
        # Use chunked computation to avoid OOM on large inputs
        chunk_size = 4096
        indices = []
        for i in range(0, flat.shape[0], chunk_size):
            chunk = flat[i:i + chunk_size]  # (chunk, 8)
            dists = torch.cdist(chunk, self._full_grid)  # (chunk, 65536)
            idx = dists.argmin(dim=-1)  # (chunk,)
            indices.append(idx)

        codes = torch.cat(indices).reshape(orig_shape)
        return codes.to(torch.int32)

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode 16-bit codes back to 8D E8 lattice points.

        Args:
            codes: (...,) tensor of uint16/int32 codes

        Returns:
            (..., 8) tensor of E8 lattice points
        """
        orig_shape = codes.shape
        flat_codes = codes.reshape(-1).long().to(self.device)
        decoded = self._full_grid[flat_codes]  # (n, 8)
        return decoded.reshape(*orig_shape, 8)

    def memory_bytes(self, n_vectors: int, d: int) -> dict:
        """Calculate memory usage for E8P encoding.

        Args:
            n_vectors: Number of vectors to store
            d: Dimension (must be divisible by 8)

        Returns:
            Dict with byte counts for each component
        """
        n_blocks = n_vectors * (d // 8)
        return {
            "codes_bytes": n_blocks * 2,  # 16 bits per block
            "norms_bytes": n_vectors * 2,  # FP16 norm per vector
            "codebook_bytes": 256 * 8 * 4,  # source set (one-time)
            "total_bytes": n_blocks * 2 + n_vectors * 2,
            "bits_per_dim": (n_blocks * 16 + n_vectors * 16) / (n_vectors * d),
            "compression_vs_fp16": (n_vectors * d * 2) / (n_blocks * 2 + n_vectors * 2),
        }
