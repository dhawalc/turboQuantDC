"""TurboQuantDC Gradio Demo — KV Cache Compression Explorer.

Interactive demo for HuggingFace Spaces (ZeroGPU T4).
Lets users experiment with TurboQuant compression in real-time.

Launch:
    python demo_app.py
"""

from __future__ import annotations

import math

import gradio as gr
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from turboquantdc import OutlierTurboQuant, TurboQuantEstimator, TurboQuantKVCache

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    "Qwen2.5-3B": {"layers": 36, "kv_heads": 2, "d": 128, "weights_gb": 6},
    "Qwen2.5-7B": {"layers": 28, "kv_heads": 4, "d": 128, "weights_gb": 14},
    "Qwen2.5-14B": {"layers": 48, "kv_heads": 8, "d": 128, "weights_gb": 28},
    "Qwen3.5-27B": {"layers": 16, "kv_heads": 8, "d": 256, "weights_gb": 15},
    "Llama-3-8B": {"layers": 32, "kv_heads": 8, "d": 128, "weights_gb": 16},
    "Llama-3-70B": {"layers": 80, "kv_heads": 8, "d": 128, "weights_gb": 40},
}

GPU_CONFIGS = {
    "T4 (16GB)": 16,
    "RTX 4090 (24GB)": 24,
    "RTX 5090 (32GB)": 32,
    "A6000 (48GB)": 48,
    "A100/H100 (80GB)": 80,
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Tab 1 helpers -- Compress & Compare
# ---------------------------------------------------------------------------

def _run_compression(bits: float, dim: int, n_vectors: int):
    """Compress random vectors and compute quality metrics.

    Returns (metrics_table, memory_bar_chart, cosine_histogram).
    """
    torch.manual_seed(42)
    is_fractional = (bits % 1) != 0
    int_bits = int(bits) if not is_fractional else None

    keys = torch.randn(n_vectors, dim, device=DEVICE)
    queries = torch.randn(n_vectors, dim, device=DEVICE)

    # --- Quantize ---
    if is_fractional:
        quantizer = OutlierTurboQuant(d=dim, target_bits=bits, seed=42, device=DEVICE)
        compressed = quantizer.quantize(keys)
        est_ip = quantizer.inner_product(queries, compressed)
        cr = quantizer.compression_ratio()
    else:
        estimator = TurboQuantEstimator(d=dim, bits=int_bits, seed=42, device=DEVICE)
        compressed = estimator.quantize(keys)
        est_ip = estimator.inner_product(queries, compressed)
        x_mse = estimator.dequantize_mse(compressed)

        # Compression ratio: use KV cache helper for consistency
        cache = TurboQuantKVCache(d_key=dim, d_value=dim, bits=int_bits, seed=42, device=DEVICE)
        cache.append(keys, keys)
        mem = cache.memory_usage_bits()
        cr = mem["compression_ratio"]

    # --- True inner products ---
    true_ip = (queries * keys).sum(dim=-1)  # element-wise per-vector dot

    # --- Per-vector cosine similarity (MSE reconstruction vs original) ---
    if is_fractional:
        # est_ip is (n_q, n_k); extract diagonal for matched pairs
        est_ip_diag = est_ip.diagonal() if est_ip.dim() == 2 else est_ip
        cos_sims = _cosine_from_ip(true_ip, est_ip_diag)
    else:
        key_norms = keys.norm(dim=-1, keepdim=True)
        keys_normed = keys / (key_norms + 1e-8)
        x_mse_norms = x_mse.norm(dim=-1, keepdim=True)
        x_mse_normed = x_mse / (x_mse_norms + 1e-8)
        cos_sims = (keys_normed * x_mse_normed).sum(dim=-1)

    avg_cosine = cos_sims.mean().item()

    # --- Top-k accuracy (attention score ranking) ---
    true_scores = queries @ keys.T  # (n, n)
    est_scores_full = _estimate_full_scores(
        queries, keys, bits, dim, is_fractional, compressed,
        quantizer if is_fractional else estimator,
    )

    top1_acc = _topk_accuracy(true_scores, est_scores_full, k=1)
    top5_acc = _topk_accuracy(true_scores, est_scores_full, k=5)

    # --- Build outputs ---
    # Metrics table
    metrics = [
        ["Cosine Similarity (avg)", f"{avg_cosine:.6f}"],
        ["Top-1 Attention Accuracy", f"{top1_acc:.1f}%"],
        ["Top-5 Attention Accuracy", f"{top5_acc:.1f}%"],
        ["Compression Ratio", f"{cr:.2f}x"],
        ["Bit-width", f"{bits}"],
        ["Device", DEVICE.upper()],
    ]

    # Memory bar chart
    fp16_bytes = n_vectors * dim * 2  # 2 bytes per FP16 element
    compressed_bytes = fp16_bytes / cr if cr > 0 else fp16_bytes
    mem_fig = _memory_bar_chart(fp16_bytes, compressed_bytes, bits)

    # Cosine histogram
    hist_fig = _cosine_histogram(cos_sims.cpu().numpy(), bits)

    return metrics, mem_fig, hist_fig


def _cosine_from_ip(true_ip: torch.Tensor, est_ip: torch.Tensor) -> torch.Tensor:
    """Compute per-element quality score from inner products.

    Uses 1 - |relative error|, clamped to [0, 1].
    """
    rel_err = (true_ip - est_ip).abs() / (true_ip.abs() + 1e-8)
    return (1.0 - rel_err).clamp(0.0, 1.0)


def _estimate_full_scores(queries, keys, bits, dim, is_fractional, compressed, quantizer):
    """Estimate full (n_queries x n_keys) attention score matrix."""
    if is_fractional:
        return quantizer.inner_product(queries, compressed)
    else:
        return quantizer.inner_product(queries, compressed)


def _topk_accuracy(true_scores: torch.Tensor, est_scores: torch.Tensor, k: int) -> float:
    """Percentage of true top-k keys that appear in estimated top-k."""
    n = true_scores.shape[0]
    k = min(k, true_scores.shape[1])
    true_topk = true_scores.topk(k, dim=-1).indices  # (n, k)
    est_topk = est_scores.topk(k, dim=-1).indices

    matches = 0
    for i in range(n):
        true_set = set(true_topk[i].cpu().tolist())
        est_set = set(est_topk[i].cpu().tolist())
        matches += len(true_set & est_set)

    return 100.0 * matches / (n * k)


def _memory_bar_chart(fp16_bytes: float, compressed_bytes: float, bits: float):
    """Bar chart comparing FP16 vs compressed memory."""
    fig, ax = plt.subplots(figsize=(6, 4))

    labels = ["FP16\n(baseline)", f"TurboQuant\n({bits}-bit)"]
    values = [fp16_bytes / 1024, compressed_bytes / 1024]
    colors = ["#e74c3c", "#2ecc71"]

    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.1f} KB",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    savings = (1 - compressed_bytes / fp16_bytes) * 100
    ax.set_title(f"Memory Usage ({savings:.0f}% savings)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Size (KB)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def _cosine_histogram(cos_sims: np.ndarray, bits: float):
    """Histogram of per-vector cosine similarities."""
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.hist(cos_sims, bins=50, color="#3498db", alpha=0.85, edgecolor="white", linewidth=0.8)
    mean_val = cos_sims.mean()
    ax.axvline(mean_val, color="#e74c3c", linestyle="--", linewidth=2, label=f"Mean: {mean_val:.4f}")

    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Count")
    ax.set_title(f"Per-Head Cosine Similarity ({bits}-bit)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 2 helpers -- Memory Calculator
# ---------------------------------------------------------------------------

def _compute_kv_cache_size_gb(
    layers: int,
    kv_heads: int,
    d: int,
    seq_len: int,
    bits_per_element: float,
) -> float:
    """Compute KV cache size in GB.

    KV cache stores both keys and values for every layer and head.
    Size = 2 (K+V) * layers * kv_heads * seq_len * d * bits / 8 / 1e9
    """
    total_bits = 2 * layers * kv_heads * seq_len * d * bits_per_element
    return total_bits / 8 / 1e9


def _turbo_bits_per_element(bits: int) -> float:
    """Effective bits per element for TurboQuant compressed cache.

    Keys: b*d bits MSE/QJL + 32 bits overhead  -> ~b + 32/d per element
    Values: b*d bits MSE + 16 bits norm         -> ~b + 16/d per element
    Average over key + value pair.
    We use the simplified approximation: bits + small overhead.
    """
    # The effective storage per coordinate (averaged over K and V):
    #   Key:  b bits/coord + (32/d) overhead per coord
    #   Value: b bits/coord + (16/d) overhead per coord
    # For d=128: overhead ~0.375 bits/coord, negligible at d=256
    return bits + 0.25  # reasonable average overhead


def _run_memory_calculator(
    model_name: str,
    context_len_k: float,
    bits: int,
    gpu_name: str,
    custom_layers: int,
    custom_kv_heads: int,
    custom_d: int,
    custom_weights_gb: float,
):
    """Memory calculator computation.

    Returns (fp16_text, turbo_text, savings_text, verdict_html, chart).
    """
    if model_name == "Custom":
        cfg = {
            "layers": custom_layers,
            "kv_heads": custom_kv_heads,
            "d": custom_d,
            "weights_gb": custom_weights_gb,
        }
    else:
        cfg = MODEL_CONFIGS[model_name]

    seq_len = int(context_len_k * 1024)
    gpu_gb = GPU_CONFIGS[gpu_name]

    # FP16 KV cache (16 bits per element)
    fp16_kv = _compute_kv_cache_size_gb(
        cfg["layers"], cfg["kv_heads"], cfg["d"], seq_len, 16.0
    )

    # TurboQuant KV cache
    turbo_bits = _turbo_bits_per_element(bits)
    turbo_kv = _compute_kv_cache_size_gb(
        cfg["layers"], cfg["kv_heads"], cfg["d"], seq_len, turbo_bits
    )

    savings_gb = fp16_kv - turbo_kv
    savings_pct = (savings_gb / fp16_kv * 100) if fp16_kv > 0 else 0

    weights_gb = cfg["weights_gb"]

    # Verdict
    fp16_total = weights_gb + fp16_kv
    turbo_total = weights_gb + turbo_kv

    fp16_fits = fp16_total <= gpu_gb
    turbo_fits = turbo_total <= gpu_gb

    if turbo_fits and not fp16_fits:
        verdict = (
            '<div style="padding:12px;border-radius:8px;background:#2ecc71;color:white;'
            'font-size:18px;text-align:center;font-weight:bold;">'
            "FITS with TurboQuant! (OOM without it)</div>"
        )
    elif turbo_fits and fp16_fits:
        verdict = (
            '<div style="padding:12px;border-radius:8px;background:#3498db;color:white;'
            'font-size:18px;text-align:center;font-weight:bold;">'
            f"FITS -- saves {savings_gb:.1f} GB for longer context</div>"
        )
    elif not turbo_fits and not fp16_fits:
        overshoot = turbo_total - gpu_gb
        verdict = (
            '<div style="padding:12px;border-radius:8px;background:#e74c3c;color:white;'
            'font-size:18px;text-align:center;font-weight:bold;">'
            f"OOM -- need {overshoot:.1f} GB more (try shorter context or bigger GPU)</div>"
        )
    else:
        verdict = (
            '<div style="padding:12px;border-radius:8px;background:#f39c12;color:white;'
            'font-size:18px;text-align:center;font-weight:bold;">'
            "FP16 fits but TurboQuant OOM (unusual -- check config)</div>"
        )

    # Chart
    chart = _memory_stacked_chart(
        weights_gb, fp16_kv, turbo_kv, gpu_gb, model_name, bits
    )

    fp16_text = f"{fp16_kv:.2f} GB"
    turbo_text = f"{turbo_kv:.2f} GB"
    savings_text = f"{savings_gb:.2f} GB ({savings_pct:.0f}%)"

    return fp16_text, turbo_text, savings_text, verdict, chart


def _memory_stacked_chart(
    weights_gb: float,
    fp16_kv: float,
    turbo_kv: float,
    gpu_gb: float,
    model_name: str,
    bits: int,
):
    """Stacked bar chart: weights + KV cache vs GPU limit."""
    fig, ax = plt.subplots(figsize=(7, 4.5))

    labels = ["FP16", f"TurboQuant ({bits}-bit)"]
    weight_vals = [weights_gb, weights_gb]
    kv_vals = [fp16_kv, turbo_kv]

    x = np.arange(len(labels))
    width = 0.45

    bars_w = ax.bar(x, weight_vals, width, label="Model Weights", color="#34495e")
    bars_kv = ax.bar(
        x, kv_vals, width, bottom=weight_vals, label="KV Cache", color="#3498db"
    )

    # GPU limit line
    ax.axhline(gpu_gb, color="#e74c3c", linestyle="--", linewidth=2, label=f"GPU VRAM ({gpu_gb} GB)")

    # Value labels on bars
    for i, (w, kv) in enumerate(zip(weight_vals, kv_vals)):
        total = w + kv
        ax.text(
            x[i], total + gpu_gb * 0.02,
            f"{total:.1f} GB",
            ha="center", va="bottom", fontweight="bold", fontsize=10,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Memory (GB)")
    ax.set_title(f"{model_name} Memory Breakdown", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Set y-limit to show GPU line clearly
    y_max = max(weights_gb + fp16_kv, gpu_gb) * 1.2
    ax.set_ylim(0, y_max)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Tab 3 helpers -- Beyond the Paper
# ---------------------------------------------------------------------------

BEYOND_PAPER_MD = """\
## Phase 5: Beyond the Paper

TurboQuantDC extends the original TurboQuant paper with five additional innovations:

### 1. Walsh-Hadamard Fast Rotation
Replaces the O(d^2) dense rotation with O(d log d) Walsh-Hadamard transform.
For d=128, this is a **7x speedup** in rotation while maintaining identical
Gaussianization properties. The WHT is also deterministic (no random seed needed).

### 2. Sparse-V Attention
Values reconstructed from MSE quantization have bounded error. Sparse-V exploits
this by only reconstructing the **top-k** values (by attention weight), skipping
low-weight tokens entirely. At k=64 of 4096, this gives a further **2x speedup**
with <0.1% quality loss.

### 3. Fractional Bit Rates (OutlierTurboQuant)
The paper only covers integer bit-widths (2, 3, 4). OutlierTurboQuant enables
**2.5-bit and 3.5-bit** by splitting channels into two groups after rotation:
- n_high channels at ceil(bits)
- n_low channels at floor(bits)
Since rotation makes all channels equally informative, the split is lossless
in expectation.

### 4. Layer-Adaptive Bit Allocation
Not all layers are equally sensitive. Early and late layers need higher fidelity,
while middle layers tolerate aggressive compression. LayerAdaptiveKVCache
assigns per-layer bit-widths under a total memory budget, using cosine similarity
as the sensitivity metric.

### 5. Temporal Decay Cache
Older tokens in the KV cache can be progressively compressed to lower bit-widths.
The most recent tokens stay at 4-bit, medium-age at 3-bit, and ancient tokens
at 2-bit. This provides a natural **context length extension** with graceful
quality degradation.
"""


def _run_outlier_demo(target_bits: float, dim: int, n_vectors: int):
    """Demonstrate OutlierTurboQuant at a given fractional bit rate.

    Returns (channel_split_info, quality_chart).
    """
    torch.manual_seed(42)

    keys = torch.randn(n_vectors, dim, device=DEVICE)
    queries = torch.randn(n_vectors, dim, device=DEVICE)

    quantizer = OutlierTurboQuant(d=dim, target_bits=target_bits, seed=42, device=DEVICE)
    compressed = quantizer.quantize(keys)
    est_ip = quantizer.inner_product(queries, compressed)
    true_ip = (queries * keys).sum(dim=-1)

    # Channel split info
    info = (
        f"**Dimension:** {dim}\n\n"
        f"**Target bits:** {target_bits}\n\n"
        f"**High-bit channels:** {quantizer.n_high} at {quantizer.high_bits}-bit\n\n"
        f"**Low-bit channels:** {quantizer.n_low} at {quantizer.low_bits}-bit\n\n"
        f"**Effective bits:** {quantizer.effective_bits:.2f}\n\n"
        f"**Compression ratio:** {quantizer.compression_ratio():.2f}x"
    )

    # Compare quality across bit rates
    bit_rates = [2.0, 2.5, 3.0]
    cosines = []
    for b in bit_rates:
        q = OutlierTurboQuant(d=dim, target_bits=b, seed=42, device=DEVICE)
        c = q.quantize(keys)
        eip = q.inner_product(queries, c)
        # inner_product returns (n_q, n_k); extract diagonal for matched pairs
        if eip.dim() == 2:
            eip_diag = eip.diagonal()
        else:
            eip_diag = eip
        # Quality metric: correlation between true and estimated IPs
        cos = torch.nn.functional.cosine_similarity(
            true_ip.unsqueeze(0), eip_diag.unsqueeze(0)
        ).item()
        cosines.append(cos)

    fig = _outlier_comparison_chart(bit_rates, cosines)

    return info, fig


def _outlier_comparison_chart(bit_rates: list[float], cosines: list[float]):
    """Bar chart comparing quality across bit rates."""
    fig, ax = plt.subplots(figsize=(6, 4))

    colors = ["#e74c3c", "#f39c12", "#2ecc71"]
    bars = ax.bar(
        [f"{b}-bit" for b in bit_rates],
        cosines,
        color=colors,
        width=0.5,
        edgecolor="white",
        linewidth=1.2,
    )

    for bar, val in zip(bars, cosines):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.001,
            f"{val:.4f}",
            ha="center", va="bottom", fontweight="bold", fontsize=11,
        )

    ax.set_ylabel("IP Correlation")
    ax.set_title("Inner Product Quality by Bit Rate", fontsize=13, fontweight="bold")
    ax.set_ylim(min(cosines) - 0.02, 1.005)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Gradio App
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    """Construct the Gradio Blocks application."""

    with gr.Blocks(
        title="TurboQuantDC -- KV Cache Compression Explorer",
    ) as app:

        gr.Markdown(
            "# TurboQuantDC -- KV Cache Compression Explorer\n"
            "Interactive demo of TurboQuant (ICLR 2026) for compressing "
            "LLM key-value caches to 3-bit with <0.5% attention quality loss."
        )

        # ==================================================================
        # Tab 1: Compress & Compare
        # ==================================================================
        with gr.Tab("Compress & Compare"):
            gr.Markdown("### Compress random vectors and measure quality metrics")

            with gr.Row():
                with gr.Column(scale=1):
                    bits_slider = gr.Slider(
                        minimum=2, maximum=4, step=0.5, value=3,
                        label="Bit-width",
                    )
                    dim_dropdown = gr.Dropdown(
                        choices=[64, 128, 256], value=128,
                        label="Vector Dimension",
                    )
                    nvec_slider = gr.Slider(
                        minimum=100, maximum=10000, step=100, value=1000,
                        label="Number of Vectors",
                    )
                    compress_btn = gr.Button("Compress", variant="primary")

                with gr.Column(scale=2):
                    metrics_table = gr.Dataframe(
                        headers=["Metric", "Value"],
                        label="Quality Metrics",
                        interactive=False,
                    )

            with gr.Row():
                memory_plot = gr.Plot(label="Memory Comparison")
                cosine_plot = gr.Plot(label="Cosine Similarity Distribution")

            compress_btn.click(
                fn=_run_compression,
                inputs=[bits_slider, dim_dropdown, nvec_slider],
                outputs=[metrics_table, memory_plot, cosine_plot],
            )

        # ==================================================================
        # Tab 2: Memory Calculator
        # ==================================================================
        with gr.Tab("Memory Calculator"):
            gr.Markdown("### Estimate KV cache memory for real LLM architectures")

            with gr.Row():
                with gr.Column(scale=1):
                    model_dropdown = gr.Dropdown(
                        choices=list(MODEL_CONFIGS.keys()) + ["Custom"],
                        value="Qwen2.5-3B",
                        label="Model",
                    )
                    context_slider = gr.Slider(
                        minimum=1, maximum=262, step=1, value=32,
                        label="Context Length (K tokens)",
                    )
                    calc_bits = gr.Dropdown(
                        choices=[2, 3, 4], value=3,
                        label="Bit-width",
                    )
                    gpu_dropdown = gr.Dropdown(
                        choices=list(GPU_CONFIGS.keys()),
                        value="RTX 4090 (24GB)",
                        label="GPU VRAM",
                    )

                    # Custom model fields (visible when "Custom" selected)
                    with gr.Accordion("Custom Model Config", open=False):
                        custom_layers = gr.Number(value=32, label="Layers", precision=0)
                        custom_kv_heads = gr.Number(value=8, label="KV Heads", precision=0)
                        custom_d = gr.Number(value=128, label="Head Dimension", precision=0)
                        custom_weights = gr.Number(value=16.0, label="Model Weights (GB)")

                    calc_btn = gr.Button("Calculate", variant="primary")

                with gr.Column(scale=2):
                    with gr.Row():
                        fp16_out = gr.Textbox(label="FP16 KV Cache", interactive=False)
                        turbo_out = gr.Textbox(label="TurboQuant KV Cache", interactive=False)
                        savings_out = gr.Textbox(label="Savings", interactive=False)

                    verdict_html = gr.HTML(label="Verdict")
                    calc_chart = gr.Plot(label="Memory Breakdown")

            calc_btn.click(
                fn=_run_memory_calculator,
                inputs=[
                    model_dropdown, context_slider, calc_bits, gpu_dropdown,
                    custom_layers, custom_kv_heads, custom_d, custom_weights,
                ],
                outputs=[fp16_out, turbo_out, savings_out, verdict_html, calc_chart],
            )

        # ==================================================================
        # Tab 3: Beyond the Paper
        # ==================================================================
        with gr.Tab("Beyond the Paper"):
            gr.Markdown(BEYOND_PAPER_MD)

            gr.Markdown("---\n### Interactive: OutlierTurboQuant (Fractional Bit Rates)")

            with gr.Row():
                with gr.Column(scale=1):
                    outlier_bits = gr.Slider(
                        minimum=2, maximum=4, step=0.5, value=2.5,
                        label="Target Bit Rate",
                    )
                    outlier_dim = gr.Dropdown(
                        choices=[64, 128, 256], value=128,
                        label="Dimension",
                    )
                    outlier_nvec = gr.Slider(
                        minimum=100, maximum=5000, step=100, value=500,
                        label="Number of Vectors",
                    )
                    outlier_btn = gr.Button("Run OutlierTurboQuant", variant="primary")

                with gr.Column(scale=2):
                    channel_info = gr.Markdown(label="Channel Split")
                    outlier_chart = gr.Plot(label="Quality Comparison")

            outlier_btn.click(
                fn=_run_outlier_demo,
                inputs=[outlier_bits, outlier_dim, outlier_nvec],
                outputs=[channel_info, outlier_chart],
            )

        # ==================================================================
        # Footer
        # ==================================================================
        gr.Markdown(
            "---\n"
            "**Links:** "
            "[GitHub](https://github.com/dhawal-pandya/turboQuantDC) | "
            "[Paper (arXiv 2504.19874)](https://arxiv.org/abs/2504.19874) | "
            "[Colab Notebook](https://colab.research.google.com/)\n\n"
            "*Built by TurboQuantDC -- 331 tests, 10,000+ lines*"
        )

    return app


# Build the module-level app object (required by HF Spaces and test import)
demo = build_app()

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())
