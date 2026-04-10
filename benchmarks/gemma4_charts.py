#!/usr/bin/env python3
"""Generate publication-quality charts for Gemma 4 + TurboQuantDC showcase.

Saves four charts to ~/Downloads/:
    1. gemma4_showcase_quality.png    -- Bar chart: quality at 2/3/4-bit
    2. gemma4_showcase_context.png    -- Line chart: 26B MoE context scaling
    3. gemma4_showcase_architecture.png -- Diagram: mixed head_dim architecture
    4. gemma4_showcase_memory.png      -- Bar chart: memory comparison at 262K

Usage:
    python benchmarks/gemma4_charts.py
"""

from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# ---------------------------------------------------------------------------
# Google color palette
# ---------------------------------------------------------------------------
BLUE = "#4285F4"
GREEN = "#34A853"
YELLOW = "#FBBC04"
RED = "#EA4335"
DARK_GRAY = "#3C4043"
LIGHT_GRAY = "#F1F3F4"
MEDIUM_GRAY = "#9AA0A6"
WHITE = "#FFFFFF"

DPI = 150
OUTPUT_DIR = os.path.expanduser("~/Downloads")


def setup_style():
    """Common matplotlib style for all charts."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": WHITE,
        "axes.facecolor": WHITE,
        "savefig.facecolor": WHITE,
        "savefig.bbox": "tight",
        "savefig.dpi": DPI,
    })


# ═══════════════════════════════════════════════════════════════════════════
# Chart 1: Quality at different bit-widths
# ═══════════════════════════════════════════════════════════════════════════

def chart_quality():
    """Bar chart: Gemma 4 quality at 2/3/4-bit compression."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    bits = [2, 3, 4]
    cosine_sims = [0.997821, 0.999994, 0.999999]
    top5_rates = [100.0, 100.0, 100.0]
    colors = [YELLOW, GREEN, BLUE]

    # Left: Cosine similarity
    bars1 = ax1.bar(bits, cosine_sims, color=colors, width=0.6, edgecolor="white", linewidth=1.5)
    ax1.set_xlabel("Bit-width")
    ax1.set_ylabel("Cosine Similarity")
    ax1.set_title("Attention Score Fidelity")
    ax1.set_xticks(bits)
    ax1.set_xticklabels(["2-bit", "3-bit", "4-bit"])
    ax1.set_ylim(0.995, 1.0005)
    ax1.axhline(y=0.995, color=RED, linestyle="--", linewidth=1.2, alpha=0.7, label="Paper target (0.995)")
    ax1.legend(fontsize=9, loc="lower right")

    # Annotate bars
    for bar, val in zip(bars1, cosine_sims):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0002,
            f"{val:.6f}", ha="center", va="bottom", fontsize=10,
            fontweight="bold", color=DARK_GRAY,
        )

    # Highlight the 0.999994 number
    ax1.annotate(
        "Near-perfect!",
        xy=(3, 0.999994), xytext=(3.5, 0.9985),
        fontsize=11, fontweight="bold", color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=2),
        ha="center",
    )

    # Right: Top-5 attention match
    bars2 = ax2.bar(bits, top5_rates, color=colors, width=0.6, edgecolor="white", linewidth=1.5)
    ax2.set_xlabel("Bit-width")
    ax2.set_ylabel("Top-5 Attention Match (%)")
    ax2.set_title("Top-5 Token Match Rate")
    ax2.set_xticks(bits)
    ax2.set_xticklabels(["2-bit", "3-bit", "4-bit"])
    ax2.set_ylim(0, 115)
    ax2.axhline(y=90, color=RED, linestyle="--", linewidth=1.2, alpha=0.7, label="Paper target (90%)")
    ax2.legend(fontsize=9, loc="lower right")

    for bar, val in zip(bars2, top5_rates):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f"{val:.0f}%", ha="center", va="bottom", fontsize=11,
            fontweight="bold", color=DARK_GRAY,
        )

    fig.suptitle(
        "Gemma 4 + TurboQuantDC: Near-Perfect 3-bit KV Compression",
        fontsize=15, fontweight="bold", color=DARK_GRAY, y=1.02,
    )

    # Subtitle
    fig.text(
        0.5, 0.97,
        "ResidualQuant on Gemma 4 E4B (4B params) | d=256 | RTX 4090",
        ha="center", fontsize=10, color=MEDIUM_GRAY, style="italic",
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "gemma4_showcase_quality.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 2: Context scaling (Gemma 4 26B MoE)
# ═══════════════════════════════════════════════════════════════════════════

def chart_context():
    """Line chart: Gemma 4 26B MoE context scaling on RTX 4090."""
    fig, ax = plt.subplots(figsize=(10, 6))

    ctx_labels = ["4K", "32K", "65K", "131K", "196K", "262K"]
    ctx_values = [4, 32, 65, 131, 196, 262]
    fp16_toks = [152, 159, 159, 161, 158, None]  # None = OOM
    turbo3_toks = [154, 158, 161, 166, 158, 150]

    # Plot turbo3 (full line)
    ax.plot(
        ctx_values, turbo3_toks, "o-",
        color=GREEN, linewidth=2.5, markersize=9,
        label="TurboQuantDC 3-bit", zorder=5,
    )

    # Plot FP16 (stops at 196K, then OOM)
    fp16_valid = [v for v in fp16_toks if v is not None]
    fp16_ctx = ctx_values[:len(fp16_valid)]
    ax.plot(
        fp16_ctx, fp16_valid, "s-",
        color=BLUE, linewidth=2.5, markersize=9,
        label="FP16 KV cache", zorder=4,
    )

    # OOM marker at 262K for FP16
    ax.scatter(
        [262], [20], marker="X", s=200, color=RED, zorder=6,
        label="FP16 OOM",
    )
    ax.annotate(
        "OOM!",
        xy=(262, 20), xytext=(262, 55),
        fontsize=14, fontweight="bold", color=RED,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=RED, lw=2.5),
    )

    # Highlight 262K turbo3 point
    ax.annotate(
        "150 tok/s\nFull native context!",
        xy=(262, 150), xytext=(220, 120),
        fontsize=11, fontweight="bold", color=GREEN,
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=2),
        ha="center",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor=GREEN),
    )

    ax.set_xlabel("Context Length (K tokens)")
    ax.set_ylabel("Generation Speed (tok/s)")
    ax.set_title(
        "Gemma 4 26B: Full 262K Context on Single RTX 4090",
        fontsize=14, fontweight="bold", color=DARK_GRAY,
    )

    ax.set_xticks(ctx_values)
    ax.set_xticklabels(ctx_labels)
    ax.set_ylim(0, 200)
    ax.set_xlim(-5, 280)

    # Shade the "impossible" zone
    ax.axvspan(196, 280, alpha=0.08, color=RED, label="_")
    ax.text(
        238, 185, "FP16 cannot\nreach here",
        ha="center", fontsize=10, color=RED, alpha=0.7,
        style="italic",
    )

    ax.legend(fontsize=11, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)

    # Subtitle
    fig.text(
        0.5, -0.02,
        "Gemma 4 26B MoE via llama.cpp turbo3 backend | RTX 4090 24GB | 262K = full native context",
        ha="center", fontsize=9, color=MEDIUM_GRAY, style="italic",
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gemma4_showcase_context.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 3: Architecture diagram (mixed head_dim)
# ═══════════════════════════════════════════════════════════════════════════

def chart_architecture():
    """Diagram: Gemma 4's mixed head_dim architecture."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(
        6, 7.5, "TurboQuantDC Handles Gemma 4's Mixed Architecture",
        ha="center", fontsize=15, fontweight="bold", color=DARK_GRAY,
    )
    ax.text(
        6, 7.05, "Gemma 4 26B MoE: 24 layers with two distinct head dimensions",
        ha="center", fontsize=10, color=MEDIUM_GRAY, style="italic",
    )

    # --- Left side: Layer stack ---
    # Sliding window layers (20)
    for i in range(5):
        y = 5.8 - i * 0.55
        rect = FancyBboxPatch(
            (0.5, y), 3.5, 0.45,
            boxstyle="round,pad=0.08",
            facecolor="#E3F2FD" if i < 4 else "#E3F2FD",
            edgecolor=BLUE, linewidth=1.5, alpha=0.9,
        )
        ax.add_patch(rect)
        label = f"Layer {i}" if i < 3 else ("..." if i == 3 else "Layer 19")
        ax.text(2.25, y + 0.22, label, ha="center", va="center", fontsize=9,
                color=BLUE, fontweight="bold")

    # Label for sliding window group
    ax.text(2.25, 6.5, "20 Sliding-Window Layers", ha="center", fontsize=11,
            fontweight="bold", color=BLUE)
    ax.text(2.25, 6.15, "head_dim = 256  |  4K window", ha="center", fontsize=9,
            color=BLUE, alpha=0.8)

    # Anchor layers (4)
    for i in range(4):
        y = 2.3 - i * 0.55
        rect = FancyBboxPatch(
            (0.5, y), 3.5, 0.45,
            boxstyle="round,pad=0.08",
            facecolor="#FFF3E0",
            edgecolor=YELLOW, linewidth=1.5, alpha=0.9,
        )
        ax.add_patch(rect)
        ax.text(2.25, y + 0.22, f"Anchor {i}", ha="center", va="center",
                fontsize=9, color="#E65100", fontweight="bold")

    ax.text(2.25, 3.0, "4 Global Anchor Layers", ha="center", fontsize=11,
            fontweight="bold", color="#E65100")
    ax.text(2.25, 2.65, "head_dim = 512  |  Full attention", ha="center",
            fontsize=9, color="#E65100", alpha=0.8)

    # --- Right side: TurboQuantDC pipeline ---
    # Pipeline box
    pipeline_rect = FancyBboxPatch(
        (6.5, 1.0), 5, 5.5,
        boxstyle="round,pad=0.15",
        facecolor=LIGHT_GRAY, edgecolor=DARK_GRAY, linewidth=2,
    )
    ax.add_patch(pipeline_rect)
    ax.text(9, 6.25, "TurboQuantDC Pipeline", ha="center", fontsize=12,
            fontweight="bold", color=DARK_GRAY)

    # Step 1: Detect head_dim
    step_y = 5.6
    r1 = FancyBboxPatch(
        (7, step_y), 4, 0.6,
        boxstyle="round,pad=0.08",
        facecolor="#E8F5E9", edgecolor=GREEN, linewidth=1.5,
    )
    ax.add_patch(r1)
    ax.text(9, step_y + 0.3, "1. Auto-detect head_dim per layer", ha="center",
            fontsize=9.5, fontweight="bold", color="#1B5E20")

    # Step 2: WHT rotation
    step_y = 4.8
    r2 = FancyBboxPatch(
        (7, step_y), 4, 0.6,
        boxstyle="round,pad=0.08",
        facecolor="#E8F5E9", edgecolor=GREEN, linewidth=1.5,
    )
    ax.add_patch(r2)
    ax.text(9, step_y + 0.3, "2. WHT Rotation (O(d log d))", ha="center",
            fontsize=9.5, fontweight="bold", color="#1B5E20")

    # Step 3: ResidualQuant
    step_y = 4.0
    r3 = FancyBboxPatch(
        (7, step_y), 4, 0.6,
        boxstyle="round,pad=0.08",
        facecolor="#E8F5E9", edgecolor=GREEN, linewidth=1.5,
    )
    ax.add_patch(r3)
    ax.text(9, step_y + 0.3, "3. ResidualQuant (MSE + sign bits)", ha="center",
            fontsize=9.5, fontweight="bold", color="#1B5E20")

    # Step 4: CUDA kernel
    step_y = 3.2
    r4 = FancyBboxPatch(
        (7, step_y), 4, 0.6,
        boxstyle="round,pad=0.08",
        facecolor="#E8F5E9", edgecolor=GREEN, linewidth=1.5,
    )
    ax.add_patch(r4)
    ax.text(9, step_y + 0.3, "4. CUDA Dequantize (29x vs Triton)", ha="center",
            fontsize=9.5, fontweight="bold", color="#1B5E20")

    # Dimension badges
    d256_rect = FancyBboxPatch(
        (7.2, 1.6), 1.8, 0.8,
        boxstyle="round,pad=0.1",
        facecolor="#BBDEFB", edgecolor=BLUE, linewidth=2,
    )
    ax.add_patch(d256_rect)
    ax.text(8.1, 2.0, "d=256", ha="center", fontsize=12, fontweight="bold", color=BLUE)

    d512_rect = FancyBboxPatch(
        (9.5, 1.6), 1.8, 0.8,
        boxstyle="round,pad=0.1",
        facecolor="#FFE0B2", edgecolor="#E65100", linewidth=2,
    )
    ax.add_patch(d512_rect)
    ax.text(10.4, 2.0, "d=512", ha="center", fontsize=12, fontweight="bold",
            color="#E65100")

    ax.text(9, 1.3, "Both dimensions: same pipeline, same quality",
            ha="center", fontsize=9, color=DARK_GRAY, style="italic")

    # Arrows from layers to pipeline
    ax.annotate(
        "", xy=(6.5, 4.5), xytext=(4.0, 4.5),
        arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2.5),
    )
    ax.annotate(
        "", xy=(6.5, 2.0), xytext=(4.0, 1.5),
        arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=2.5),
    )

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "gemma4_showcase_architecture.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Chart 4: Memory comparison at 262K
# ═══════════════════════════════════════════════════════════════════════════

def chart_memory():
    """Bar chart: memory comparison at 262K context."""
    fig, ax = plt.subplots(figsize=(9, 6))

    # Gemma 4 26B MoE at 262K tokens
    # 24 layers, mixed: 20 x (d=256, 4 KV heads) + 4 x (d=512, 4 KV heads)
    seq = 262_144

    # FP16 calculation
    sw_fp16 = 20 * 4 * seq * 256 * 2 * 2  # 20 layers, 4 heads, K+V, 2 bytes
    anchor_fp16 = 4 * 4 * seq * 512 * 2 * 2
    fp16_gb = (sw_fp16 + anchor_fp16) / (1024 ** 3)

    # 3-bit ResidualQuant
    bits = 3
    sw_rq = 20 * 4 * seq * (2 * bits * 256 + 48) / 8
    anchor_rq = 4 * 4 * seq * (2 * bits * 512 + 48) / 8
    rq3_gb = (sw_rq + anchor_rq) / (1024 ** 3)

    # GPU VRAM limit
    vram = 24.0

    categories = ["FP16 KV Cache\n(would OOM)", "3-bit TurboQuantDC\n(fits on GPU)", "RTX 4090 VRAM"]
    values = [fp16_gb, rq3_gb, vram]
    colors_list = [RED, GREEN, MEDIUM_GRAY]
    edge_colors = [RED, GREEN, DARK_GRAY]

    bars = ax.bar(
        categories, values,
        color=colors_list, width=0.55, edgecolor=edge_colors,
        linewidth=2, alpha=0.9,
    )

    # VRAM bar is just an outline
    bars[2].set_alpha(0.2)
    bars[2].set_hatch("///")

    # Value labels
    ax.text(0, fp16_gb + 0.8, f"{fp16_gb:.1f} GB", ha="center", fontsize=14,
            fontweight="bold", color=RED)
    ax.text(1, rq3_gb + 0.8, f"{rq3_gb:.1f} GB", ha="center", fontsize=14,
            fontweight="bold", color=GREEN)
    ax.text(2, vram + 0.8, f"{vram:.0f} GB", ha="center", fontsize=14,
            fontweight="bold", color=DARK_GRAY)

    # OOM line
    ax.axhline(y=vram, color=RED, linestyle="--", linewidth=1.5, alpha=0.5)
    ax.text(2.4, vram + 0.3, "GPU limit", fontsize=9, color=RED, alpha=0.7)

    # Savings annotation
    savings_pct = (1 - rq3_gb / fp16_gb) * 100
    ratio = fp16_gb / rq3_gb
    ax.annotate(
        f"{ratio:.1f}x smaller\n{savings_pct:.0f}% savings",
        xy=(1, rq3_gb), xytext=(1.7, fp16_gb * 0.6),
        fontsize=12, fontweight="bold", color=GREEN,
        ha="center",
        arrowprops=dict(arrowstyle="->", color=GREEN, lw=2),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E8F5E9", edgecolor=GREEN),
    )

    ax.set_ylabel("Memory (GB)")
    ax.set_title(
        "262K Context: From OOM to Fits Comfortably",
        fontsize=14, fontweight="bold", color=DARK_GRAY,
    )
    ax.set_ylim(0, max(fp16_gb, vram) * 1.25)

    # Subtitle
    fig.text(
        0.5, -0.02,
        f"Gemma 4 26B MoE | 262,144 tokens | 24 layers (mixed d=256/d=512) | 4 KV heads",
        ha="center", fontsize=9, color=MEDIUM_GRAY, style="italic",
    )

    ax.grid(axis="y", alpha=0.2)
    plt.tight_layout()

    path = os.path.join(OUTPUT_DIR, "gemma4_showcase_memory.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    setup_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Generating Gemma 4 + TurboQuantDC showcase charts...")
    print()

    chart_quality()
    chart_context()
    chart_architecture()
    chart_memory()

    print()
    print(f"All charts saved to {OUTPUT_DIR}/gemma4_showcase_*.png")


if __name__ == "__main__":
    main()
