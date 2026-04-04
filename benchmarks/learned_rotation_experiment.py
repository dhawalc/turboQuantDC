"""Learned rotation experiment: Can PCA beat random WHT for KV compression?

Hypothesis:  PCA rotation concentrates variance into the first coordinates,
enabling sparser representation and potentially better compression.

Experiment outline:
    1. Load Qwen2.5-3B (BnB 4-bit), extract KV cache from a long prompt.
    2. For every layer+head, compare three quantisers at 3 bits:
       (a) WHT rotation + 3-bit Lloyd-Max  (current baseline)
       (b) PCA rotation + 3-bit Lloyd-Max  (learned, uniform bits)
       (c) PCA rotation + adaptive bits    (learned, variable bits)
    3. Measure cosine similarity of attention scores, top-1/5 match.
    4. Test calibration-size sensitivity: does PCA from 100 tokens suffice?
    5. Transfer test: fit PCA on Prompt A, evaluate on Prompt B.

Usage:
    python benchmarks/learned_rotation_experiment.py
"""

from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from turboquantdc.codebook import LloydMaxCodebook
from turboquantdc.polarquant import PolarQuant
from turboquantdc.learned_rotation import (
    PCARotatedQuantizer,
    compute_adaptive_bit_allocation,
    compute_pca_rotation,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
CACHE_DIR = "/media/dhawal/Beast/cache/hub/"
BITS = 3
ADAPTIVE_TARGET_BITS = 3.0

# Prompts for calibration and evaluation
PROMPT_A = (
    "The quarterly financial review meeting covered several topics including "
    "budget allocations for the upcoming fiscal year, departmental spending reports, "
    "and projected revenue streams from various business units. The committee discussed "
    "infrastructure upgrades planned for the western regional offices and noted that "
    "maintenance schedules should be coordinated with the facilities management team. "
    "Several action items were assigned to team leads for follow-up. "
    "Meanwhile, the engineering division reported progress on the Artemis satellite "
    "navigation system, achieving sub-meter accuracy in urban canyon environments. "
    "The team demonstrated real-time kinematic processing capabilities with latency "
    "under 50 milliseconds, meeting the specification for autonomous vehicle integration. "
    "The neural beamforming antenna array passed environmental stress testing at "
    "temperatures ranging from minus 40 to plus 85 degrees Celsius. "
    "The secret code for project Aurora is NEXUS-4477. "
    "In the research lab, Dr. Chen presented findings on topological quantum error "
    "correction using surface codes with distance seven. The logical error rate dropped "
    "below the threshold required for fault-tolerant computation, representing a "
    "significant milestone in the quantum computing roadmap. The cryogenic control "
    "electronics achieved 99.7% gate fidelity across all 72 qubits. "
    "The materials science team synthesised a new high-entropy alloy with exceptional "
    "strength-to-weight ratio, outperforming titanium-6-4 by 23% in fatigue testing. "
    "Applications in aerospace structural members are being explored. "
    "The bioinformatics group completed genome-wide association studies on a cohort of "
    "500,000 participants, identifying 47 novel loci associated with cardiovascular "
    "disease risk. Machine learning models trained on this data achieved an AUC of 0.89 "
    "for ten-year risk prediction. "
    "Climate modelling simulations projected a 2.7 degree warming scenario under current "
    "emission trajectories, with significant regional variability. The Arctic region "
    "showed amplified warming of 4.3 degrees, consistent with satellite observations "
    "from the past decade of environmental monitoring. "
) * 7  # repeat for length

PROMPT_B = (
    "Advanced photovoltaic research has yielded tandem solar cells with 33.7% efficiency "
    "under standard test conditions, surpassing the Shockley-Queisser limit for single "
    "junction devices. The perovskite top cell was deposited using a scalable slot-die "
    "coating process compatible with roll-to-roll manufacturing lines. Stability testing "
    "under damp heat conditions showed less than 5% degradation after 1000 hours. "
    "In the adjacent building, the robotics team demonstrated a bipedal walking system "
    "capable of traversing uneven terrain at 1.2 meters per second. The reinforcement "
    "learning policy, trained in simulation and transferred to hardware with minimal "
    "fine-tuning, handled step heights up to 25 centimeters. The actuator system uses "
    "quasi-direct-drive motors for high backdrivability and impact resilience. "
    "The password for the secure vault is DIAMOND-8812. "
    "The computational linguistics group released a new multilingual benchmark covering "
    "87 languages with parallel evaluation sets. Results showed that language models "
    "with shared subword vocabularies outperformed language-specific models on low-resource "
    "languages by 12 BLEU points on average. The dataset includes dialects and code-mixed "
    "samples that have been historically underrepresented. "
    "Superconducting radio-frequency cavities for the next-generation particle accelerator "
    "achieved quality factors exceeding 3 times 10 to the 10th at 2 Kelvin, representing "
    "a factor-of-two improvement over the previous generation. The nitrogen doping recipe "
    "was optimised through a systematic study of annealing temperature and duration. "
    "Marine biologists documented a previously unknown deep-sea hydrothermal vent community "
    "at 4,200 meters depth in the Mariana Trough, featuring chemosynthetic organisms "
    "adapted to extreme pressure and temperature gradients. "
    "Urban transportation planners deployed an adaptive traffic signal network across 340 "
    "intersections, reducing average commute times by 18% and emissions by 14% during "
    "peak hours. The system uses real-time vehicle counting from edge-deployed vision "
    "models with privacy-preserving inference. "
) * 7  # repeat for length


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------
@dataclass
class MethodResult:
    """Results for one quantisation method on one head."""
    method: str
    cosine_sim: float
    top1_match: bool
    top5_match: bool
    mse: float
    effective_bits: float


@dataclass
class HeadComparison:
    """All methods compared on one layer+head pair."""
    layer: int
    head: int
    results: Dict[str, MethodResult] = field(default_factory=dict)


@dataclass
class CalibSizeResult:
    """Result of calibration-size sweep for one size."""
    n_calib: int
    pca_cos_sim: float
    pca_mse: float
    transfer_cos_sim: float
    transfer_mse: float


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_model():
    """Load Qwen2.5-3B-Instruct in BnB 4-bit."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading {MODEL_NAME} (4-bit NF4)...", flush=True)
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=CACHE_DIR,
    )
    model.eval()

    elapsed = time.time() - t0
    gpu_mb = torch.cuda.memory_allocated() // (1 << 20)
    config = model.config
    n_layers = config.num_hidden_layers
    n_heads = config.num_attention_heads
    head_dim = config.hidden_size // n_heads
    n_kv_heads = getattr(config, "num_key_value_heads", n_heads)

    print(f"  Loaded in {elapsed:.1f}s | GPU: {gpu_mb} MB")
    print(f"  layers={n_layers} heads={n_heads} kv_heads={n_kv_heads} d={head_dim}")
    return model, tokenizer, n_layers, n_kv_heads, head_dim


# ---------------------------------------------------------------------------
# KV cache extraction
# ---------------------------------------------------------------------------
def extract_kv_cache(model, tokenizer, prompt: str, max_tokens: int = 2048):
    """Run forward pass and return key cache as list of (n_kv_heads, seq, d) tensors."""
    inputs = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=max_tokens
    ).to("cuda")
    seq_len = inputs["input_ids"].shape[1]
    print(f"  Prompt tokenised to {seq_len} tokens")

    with torch.no_grad():
        outputs = model(**inputs, use_cache=True, output_attentions=False)

    cache = outputs.past_key_values
    key_cache = []

    if hasattr(cache, "key_cache"):
        for layer_keys in cache.key_cache:
            key_cache.append(layer_keys[0].float())  # (kv_heads, seq, d)
    elif hasattr(cache, "layers"):
        for layer in cache.layers:
            key_cache.append(layer.keys[0].float())
    else:
        for layer_data in cache:
            key_cache.append(layer_data[0][0].float())

    return key_cache, seq_len


# ---------------------------------------------------------------------------
# Eigenvalue analysis
# ---------------------------------------------------------------------------
def analyse_eigenspectrum(key_cache, n_layers, n_kv_heads, head_dim):
    """Analyse the eigenvalue spectrum across layers.  Returns per-layer stats."""
    print("\n" + "=" * 70)
    print("EIGENVALUE SPECTRUM ANALYSIS")
    print("=" * 70)

    layer_stats = []

    for li in range(n_layers):
        layer_keys = key_cache[li]  # (kv_heads, seq, d)
        for hi in range(n_kv_heads):
            keys = layer_keys[hi]  # (seq, d)
            # Normalize before PCA (matches what the quantizer sees)
            key_norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            keys_normed = keys / key_norms
            pca = compute_pca_rotation(keys_normed.cpu())
            eigs = pca["eigenvalues"]

            total = eigs.sum().item()
            top10_frac = eigs[:head_dim // 10].sum().item() / total if total > 0 else 0
            top25_frac = eigs[:head_dim // 4].sum().item() / total if total > 0 else 0
            top50_frac = eigs[:head_dim // 2].sum().item() / total if total > 0 else 0
            condition = (eigs[0] / eigs[-1]).item() if eigs[-1] > 0 else float("inf")

            layer_stats.append({
                "layer": li, "head": hi,
                "top10_var": top10_frac,
                "top25_var": top25_frac,
                "top50_var": top50_frac,
                "condition": condition,
                "max_eig": eigs[0].item(),
                "min_eig": eigs[-1].item(),
            })

    # Print summary table
    print(f"\n{'Layer':>5} {'Head':>4} {'Top-10%':>8} {'Top-25%':>8} {'Top-50%':>8} {'Cond#':>10}")
    print("-" * 50)
    for s in layer_stats[:40]:  # first 40 entries
        print(f"{s['layer']:>5} {s['head']:>4} {s['top10_var']:>8.1%} "
              f"{s['top25_var']:>8.1%} {s['top50_var']:>8.1%} "
              f"{s['condition']:>10.1f}")

    avg_top10 = sum(s["top10_var"] for s in layer_stats) / len(layer_stats)
    avg_top25 = sum(s["top25_var"] for s in layer_stats) / len(layer_stats)
    avg_top50 = sum(s["top50_var"] for s in layer_stats) / len(layer_stats)
    avg_cond = sum(min(s["condition"], 1e6) for s in layer_stats) / len(layer_stats)

    print(f"\n  AVERAGE: top-10%={avg_top10:.1%}  top-25%={avg_top25:.1%}  "
          f"top-50%={avg_top50:.1%}  cond#={avg_cond:.1f}")

    return layer_stats


# ---------------------------------------------------------------------------
# Head-level comparison (WHT vs PCA-uniform vs PCA-adaptive)
# ---------------------------------------------------------------------------
def compare_methods_on_head(
    keys: torch.Tensor,    # (seq, d)
    head_dim: int,
    bits: int,
    layer_idx: int,
    head_idx: int,
    calib_keys: Optional[torch.Tensor] = None,
) -> HeadComparison:
    """Compare WHT, PCA-uniform, PCA-adaptive on one head's keys."""
    seq_len = keys.shape[0]
    device = keys.device

    # Query = last token attending to all keys
    query = keys[-1:]  # (1, d)
    real_scores = (query @ keys.T).squeeze(0)  # (seq,)

    comparison = HeadComparison(layer=layer_idx, head=head_idx)

    # --- Method A: WHT rotation + uniform 3-bit (baseline) ---
    seed = layer_idx * 10000 + head_idx
    pq_wht = PolarQuant(head_dim, bits, seed=seed, device=str(device))

    # Normalise for fair comparison
    norms = keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    keys_norm = keys / norms
    query_norm = query / query.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    wht_recon, _ = pq_wht(keys_norm)
    wht_recon = wht_recon * norms
    wht_scores = (query @ wht_recon.T).squeeze(0)

    wht_cos = F.cosine_similarity(real_scores.unsqueeze(0), wht_scores.unsqueeze(0)).item()
    wht_mse = (keys - wht_recon).pow(2).mean().item()
    wht_top1 = real_scores.argmax().item() == wht_scores.argmax().item()
    wht_top5 = real_scores.argmax().item() in wht_scores.topk(min(5, seq_len)).indices.tolist()

    comparison.results["wht_3bit"] = MethodResult(
        method="wht_3bit", cosine_sim=wht_cos, top1_match=wht_top1,
        top5_match=wht_top5, mse=wht_mse, effective_bits=bits,
    )

    # --- Calibration data for PCA (must be normalized, matching quantizer input) ---
    calib = calib_keys if calib_keys is not None else keys
    calib_norms = calib.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    calib_norm = calib / calib_norms
    pca_data = compute_pca_rotation(calib_norm.cpu())

    # --- Method B: PCA rotation + uniform 3-bit ---
    pca_q = PCARotatedQuantizer(
        d=head_dim, bits=bits, rotation_data=pca_data,
        adaptive_bits=False, device=str(device),
    )
    pca_recon, _ = pca_q(keys_norm)
    pca_recon = pca_recon * norms
    pca_scores = (query @ pca_recon.T).squeeze(0)

    pca_cos = F.cosine_similarity(real_scores.unsqueeze(0), pca_scores.unsqueeze(0)).item()
    pca_mse = (keys - pca_recon).pow(2).mean().item()
    pca_top1 = real_scores.argmax().item() == pca_scores.argmax().item()
    pca_top5 = real_scores.argmax().item() in pca_scores.topk(min(5, seq_len)).indices.tolist()

    comparison.results["pca_3bit"] = MethodResult(
        method="pca_3bit", cosine_sim=pca_cos, top1_match=pca_top1,
        top5_match=pca_top5, mse=pca_mse, effective_bits=bits,
    )

    # --- Method C: PCA rotation + adaptive bits ---
    pca_adap = PCARotatedQuantizer(
        d=head_dim, bits=bits, rotation_data=pca_data,
        adaptive_bits=True, target_avg_bits=ADAPTIVE_TARGET_BITS,
        device=str(device),
    )
    adap_recon, _ = pca_adap(keys_norm)
    adap_recon = adap_recon * norms
    adap_scores = (query @ adap_recon.T).squeeze(0)

    adap_cos = F.cosine_similarity(real_scores.unsqueeze(0), adap_scores.unsqueeze(0)).item()
    adap_mse = (keys - adap_recon).pow(2).mean().item()
    adap_top1 = real_scores.argmax().item() == adap_scores.argmax().item()
    adap_top5 = real_scores.argmax().item() in adap_scores.topk(min(5, seq_len)).indices.tolist()

    comparison.results["pca_adaptive"] = MethodResult(
        method="pca_adaptive", cosine_sim=adap_cos, top1_match=adap_top1,
        top5_match=adap_top5, mse=adap_mse,
        effective_bits=pca_adap.effective_bits_per_coord(),
    )

    return comparison


# ---------------------------------------------------------------------------
# Full comparison across all layers/heads
# ---------------------------------------------------------------------------
def run_full_comparison(key_cache, n_layers, n_kv_heads, head_dim, bits):
    """Compare all methods across all layers and heads."""
    print("\n" + "=" * 70)
    print(f"FULL COMPARISON: WHT vs PCA vs PCA-Adaptive @ {bits}-bit")
    print("=" * 70)

    all_comparisons: List[HeadComparison] = []
    t0 = time.time()

    for li in range(n_layers):
        layer_keys = key_cache[li]
        for hi in range(n_kv_heads):
            keys = layer_keys[hi].cuda()  # (seq, d)
            comp = compare_methods_on_head(keys, head_dim, bits, li, hi)
            all_comparisons.append(comp)

    elapsed = time.time() - t0

    # Aggregate results
    methods = ["wht_3bit", "pca_3bit", "pca_adaptive"]
    print(f"\n{'Method':<18} {'CosSim':>8} {'Top-1':>7} {'Top-5':>7} {'MSE':>10} {'Bits':>6}")
    print("-" * 60)

    aggregated = {}
    for method in methods:
        cos_sims = [c.results[method].cosine_sim for c in all_comparisons]
        top1s = [c.results[method].top1_match for c in all_comparisons]
        top5s = [c.results[method].top5_match for c in all_comparisons]
        mses = [c.results[method].mse for c in all_comparisons]
        eff_bits = [c.results[method].effective_bits for c in all_comparisons]

        avg_cos = sum(cos_sims) / len(cos_sims)
        pct_top1 = 100.0 * sum(top1s) / len(top1s)
        pct_top5 = 100.0 * sum(top5s) / len(top5s)
        avg_mse = sum(mses) / len(mses)
        avg_bits = sum(eff_bits) / len(eff_bits)

        print(f"  {method:<16} {avg_cos:>8.4f} {pct_top1:>6.1f}% {pct_top5:>6.1f}% "
              f"{avg_mse:>10.6f} {avg_bits:>6.2f}")

        aggregated[method] = {
            "cos_sim": avg_cos, "top1": pct_top1, "top5": pct_top5,
            "mse": avg_mse, "bits": avg_bits,
        }

    print(f"\n  [{elapsed:.1f}s across {len(all_comparisons)} heads]")
    return all_comparisons, aggregated


# ---------------------------------------------------------------------------
# Calibration size sensitivity
# ---------------------------------------------------------------------------
def run_calibration_sweep(key_cache, n_layers, n_kv_heads, head_dim, bits):
    """Test how much calibration data the PCA rotation needs."""
    print("\n" + "=" * 70)
    print("CALIBRATION SIZE SENSITIVITY")
    print("=" * 70)

    calib_sizes = [32, 64, 128, 256, 512]
    # Pick a representative subset of heads to keep this fast
    test_heads = []
    for li in [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]:
        if li < n_layers:
            for hi in range(min(n_kv_heads, 2)):
                test_heads.append((li, hi))

    results: List[CalibSizeResult] = []

    for n_calib in calib_sizes:
        cos_sims = []
        mses = []

        for li, hi in test_heads:
            keys = key_cache[li][hi].cuda()
            seq_len = keys.shape[0]
            if seq_len < n_calib + 50:
                continue

            # Calibrate on first n_calib tokens, evaluate on ALL tokens
            calib_data = keys[:n_calib]
            eval_data = keys  # full sequence

            norms = eval_data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            eval_norm = eval_data / norms
            query = eval_data[-1:]
            real_scores = (query @ eval_data.T).squeeze(0)

            # PCA must be computed on normalized vectors
            calib_norms = calib_data.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            pca_data = compute_pca_rotation((calib_data / calib_norms).cpu())
            pca_q = PCARotatedQuantizer(
                d=head_dim, bits=bits, rotation_data=pca_data,
                adaptive_bits=False, device="cuda",
            )
            recon, _ = pca_q(eval_norm)
            recon = recon * norms
            pca_scores = (query @ recon.T).squeeze(0)

            cos = F.cosine_similarity(real_scores.unsqueeze(0), pca_scores.unsqueeze(0)).item()
            mse = (eval_data - recon).pow(2).mean().item()
            cos_sims.append(cos)
            mses.append(mse)

        if cos_sims:
            results.append(CalibSizeResult(
                n_calib=n_calib,
                pca_cos_sim=sum(cos_sims) / len(cos_sims),
                pca_mse=sum(mses) / len(mses),
                transfer_cos_sim=0.0,  # filled in transfer test
                transfer_mse=0.0,
            ))

    print(f"\n{'N_calib':>8} {'CosSim':>8} {'MSE':>10}")
    print("-" * 30)
    for r in results:
        print(f"  {r.n_calib:>6} {r.pca_cos_sim:>8.4f} {r.pca_mse:>10.6f}")

    return results


# ---------------------------------------------------------------------------
# Transfer test: PCA from Prompt A evaluated on Prompt B
# ---------------------------------------------------------------------------
def run_transfer_test(model, tokenizer, n_layers, n_kv_heads, head_dim, bits,
                      key_cache_a):
    """Fit PCA on prompt A, evaluate on prompt B."""
    print("\n" + "=" * 70)
    print("TRANSFER TEST: PCA from Prompt A -> Prompt B")
    print("=" * 70)

    # Extract cache from prompt B
    print("  Extracting KV cache for Prompt B...")
    key_cache_b, seq_len_b = extract_kv_cache(model, tokenizer, PROMPT_B)

    # Also test self-consistency: PCA from A -> eval on A (should be best)
    configs = [
        ("self (A->A)", key_cache_a, key_cache_a),
        ("transfer (A->B)", key_cache_a, key_cache_b),
        ("self (B->B)", key_cache_b, key_cache_b),
    ]

    transfer_results = {}

    for label, calib_cache, eval_cache in configs:
        cos_sims = []
        mses = []
        top1s = []
        top5s = []

        for li in range(n_layers):
            for hi in range(n_kv_heads):
                calib_keys = calib_cache[li][hi].cuda()
                eval_keys = eval_cache[li][hi].cuda()

                norms = eval_keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                eval_norm = eval_keys / norms
                query = eval_keys[-1:]
                real_scores = (query @ eval_keys.T).squeeze(0)
                seq_len = eval_keys.shape[0]

                # PCA on normalized calibration keys
                calib_norms = calib_keys.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                pca_data = compute_pca_rotation((calib_keys / calib_norms).cpu())
                pca_q = PCARotatedQuantizer(
                    d=head_dim, bits=bits, rotation_data=pca_data,
                    adaptive_bits=False, device="cuda",
                )
                recon, _ = pca_q(eval_norm)
                recon = recon * norms
                pca_scores = (query @ recon.T).squeeze(0)

                cos = F.cosine_similarity(
                    real_scores.unsqueeze(0), pca_scores.unsqueeze(0)
                ).item()
                mse = (eval_keys - recon).pow(2).mean().item()
                top1 = real_scores.argmax().item() == pca_scores.argmax().item()
                top5 = real_scores.argmax().item() in pca_scores.topk(
                    min(5, seq_len)
                ).indices.tolist()

                cos_sims.append(cos)
                mses.append(mse)
                top1s.append(top1)
                top5s.append(top5)

        avg_cos = sum(cos_sims) / len(cos_sims)
        avg_mse = sum(mses) / len(mses)
        pct_top1 = 100.0 * sum(top1s) / len(top1s)
        pct_top5 = 100.0 * sum(top5s) / len(top5s)

        transfer_results[label] = {
            "cos_sim": avg_cos, "mse": avg_mse,
            "top1": pct_top1, "top5": pct_top5,
        }

    print(f"\n{'Config':<20} {'CosSim':>8} {'Top-1':>7} {'Top-5':>7} {'MSE':>10}")
    print("-" * 55)
    for label, r in transfer_results.items():
        print(f"  {label:<18} {r['cos_sim']:>8.4f} {r['top1']:>6.1f}% "
              f"{r['top5']:>6.1f}% {r['mse']:>10.6f}")

    return transfer_results


# ---------------------------------------------------------------------------
# Per-layer breakdown: which layers benefit most from PCA?
# ---------------------------------------------------------------------------
def run_layer_breakdown(all_comparisons, n_layers, n_kv_heads):
    """Show per-layer delta between PCA and WHT."""
    print("\n" + "=" * 70)
    print("PER-LAYER BREAKDOWN: PCA advantage over WHT")
    print("=" * 70)

    layer_deltas = {}
    for comp in all_comparisons:
        li = comp.layer
        if li not in layer_deltas:
            layer_deltas[li] = {"cos_delta": [], "mse_ratio": []}
        wht = comp.results["wht_3bit"]
        pca = comp.results["pca_3bit"]
        layer_deltas[li]["cos_delta"].append(pca.cosine_sim - wht.cosine_sim)
        layer_deltas[li]["mse_ratio"].append(
            pca.mse / wht.mse if wht.mse > 1e-12 else 1.0
        )

    print(f"\n{'Layer':>5} {'CosSim delta':>13} {'MSE ratio':>10} {'Winner':>8}")
    print("-" * 40)
    pca_wins = 0
    wht_wins = 0
    for li in sorted(layer_deltas.keys()):
        d = layer_deltas[li]
        avg_delta = sum(d["cos_delta"]) / len(d["cos_delta"])
        avg_ratio = sum(d["mse_ratio"]) / len(d["mse_ratio"])
        winner = "PCA" if avg_delta > 0.0001 else ("WHT" if avg_delta < -0.0001 else "TIE")
        if winner == "PCA":
            pca_wins += 1
        elif winner == "WHT":
            wht_wins += 1
        print(f"  {li:>3}   {avg_delta:>+12.5f} {avg_ratio:>10.4f}   {winner:>6}")

    print(f"\n  PCA wins: {pca_wins}/{n_layers}  |  WHT wins: {wht_wins}/{n_layers}")
    return layer_deltas


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
def save_results(
    aggregated,
    calib_results,
    transfer_results,
    layer_stats,
    layer_deltas,
    n_layers, n_kv_heads, head_dim, seq_len,
):
    """Save results to markdown file."""
    out_dir = os.path.join(REPO_ROOT, "benchmarks", "results")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "learned_rotation_results.md")

    lines = []
    lines.append("# Learned Rotation Experiment Results")
    lines.append("")
    lines.append(f"**Model:** {MODEL_NAME}  ")
    lines.append(f"**Layers:** {n_layers} | **KV heads:** {n_kv_heads} | **d:** {head_dim} | **seq:** {seq_len}  ")
    lines.append(f"**Baseline bits:** {BITS} | **Adaptive target:** {ADAPTIVE_TARGET_BITS}  ")
    lines.append(f"**Date:** {time.strftime('%Y-%m-%d %H:%M')}  ")
    lines.append("")

    # Eigenvalue spectrum
    lines.append("## 1. Eigenvalue Spectrum")
    lines.append("")
    if layer_stats:
        avg_t10 = sum(s["top10_var"] for s in layer_stats) / len(layer_stats)
        avg_t25 = sum(s["top25_var"] for s in layer_stats) / len(layer_stats)
        avg_t50 = sum(s["top50_var"] for s in layer_stats) / len(layer_stats)
        avg_cond = sum(min(s["condition"], 1e6) for s in layer_stats) / len(layer_stats)
        lines.append(f"Average across all heads:  ")
        lines.append(f"- Top 10% of coordinates hold **{avg_t10:.1%}** of variance  ")
        lines.append(f"- Top 25% of coordinates hold **{avg_t25:.1%}** of variance  ")
        lines.append(f"- Top 50% of coordinates hold **{avg_t50:.1%}** of variance  ")
        lines.append(f"- Condition number: **{avg_cond:.1f}**  ")
        lines.append("")
        lines.append("**Interpretation:** " + (
            "High concentration means PCA can exploit structure that WHT ignores."
            if avg_t25 > 0.65 else
            "Moderate concentration -- PCA has some advantage but spectrum is not extremely skewed."
            if avg_t25 > 0.50 else
            "Low concentration -- KV vectors are already near-isotropic, PCA offers little advantage."
        ))
    lines.append("")

    # Main comparison
    lines.append("## 2. Compression Quality Comparison")
    lines.append("")
    lines.append("| Method | Cosine Sim | Top-1 | Top-5 | MSE | Eff. Bits |")
    lines.append("|--------|-----------|-------|-------|-----|-----------|")
    for method in ["wht_3bit", "pca_3bit", "pca_adaptive"]:
        if method in aggregated:
            a = aggregated[method]
            lines.append(
                f"| {method} | {a['cos_sim']:.4f} | {a['top1']:.1f}% | "
                f"{a['top5']:.1f}% | {a['mse']:.6f} | {a['bits']:.2f} |"
            )
    lines.append("")

    # Delta analysis
    if aggregated.get("pca_3bit") and aggregated.get("wht_3bit"):
        pca = aggregated["pca_3bit"]
        wht = aggregated["wht_3bit"]
        cos_delta = pca["cos_sim"] - wht["cos_sim"]
        mse_ratio = pca["mse"] / wht["mse"] if wht["mse"] > 0 else 1.0
        lines.append(f"**PCA vs WHT delta:** CosSim {cos_delta:+.4f} | "
                      f"MSE ratio {mse_ratio:.3f}x  ")
        lines.append("")
        if cos_delta > 0.001:
            lines.append("**Verdict:** PCA rotation provides a meaningful improvement "
                          "over random WHT.")
        elif cos_delta > -0.001:
            lines.append("**Verdict:** PCA and WHT are roughly equivalent -- the random "
                          "rotation already does a good job.")
        else:
            lines.append("**Verdict:** WHT actually outperforms PCA rotation. "
                          "The random rotation's distribution assumptions are better "
                          "suited to Lloyd-Max than the data-adapted eigenbasis.")

    lines.append("")

    # Calibration sweep
    lines.append("## 3. Calibration Size Sensitivity")
    lines.append("")
    if calib_results:
        lines.append("| N_calib | Cosine Sim | MSE |")
        lines.append("|---------|-----------|-----|")
        for r in calib_results:
            lines.append(f"| {r.n_calib} | {r.pca_cos_sim:.4f} | {r.pca_mse:.6f} |")
        lines.append("")
        if len(calib_results) >= 2:
            small = calib_results[0]
            large = calib_results[-1]
            gap = abs(large.pca_cos_sim - small.pca_cos_sim)
            lines.append(f"**Gap from {small.n_calib} to {large.n_calib} tokens:** "
                          f"{gap:.4f} cosine sim  ")
            if gap < 0.002:
                lines.append("PCA generalises well from very few tokens -- "
                              "even 32 tokens suffice.")
            elif gap < 0.005:
                lines.append("PCA needs modest calibration (~128-256 tokens) "
                              "to stabilise.")
            else:
                lines.append("PCA is sensitive to calibration size -- "
                              "needs 500+ tokens for stable rotation.")
    lines.append("")

    # Transfer test
    lines.append("## 4. Transfer Test (Cross-Prompt Generalisation)")
    lines.append("")
    if transfer_results:
        lines.append("| Config | Cosine Sim | Top-1 | Top-5 | MSE |")
        lines.append("|--------|-----------|-------|-------|-----|")
        for label, r in transfer_results.items():
            lines.append(
                f"| {label} | {r['cos_sim']:.4f} | {r['top1']:.1f}% | "
                f"{r['top5']:.1f}% | {r['mse']:.6f} |"
            )
        lines.append("")

        self_a = transfer_results.get("self (A->A)", {})
        cross = transfer_results.get("transfer (A->B)", {})
        if self_a and cross:
            drop = self_a.get("cos_sim", 0) - cross.get("cos_sim", 0)
            lines.append(f"**Transfer drop:** {drop:.4f} cosine sim  ")
            if drop < 0.002:
                lines.append("PCA rotation transfers almost perfectly across prompts -- "
                              "a single calibration pass suffices.")
            elif drop < 0.005:
                lines.append("PCA rotation transfers reasonably well -- "
                              "modest degradation on unseen prompts.")
            else:
                lines.append("PCA rotation is prompt-specific -- "
                              "requires per-prompt or periodic recalibration.")
    lines.append("")

    # Per-layer breakdown
    lines.append("## 5. Per-Layer Breakdown")
    lines.append("")
    if layer_deltas:
        pca_win_layers = sum(
            1 for d in layer_deltas.values()
            if sum(d["cos_delta"]) / len(d["cos_delta"]) > 0.0001
        )
        wht_win_layers = sum(
            1 for d in layer_deltas.values()
            if sum(d["cos_delta"]) / len(d["cos_delta"]) < -0.0001
        )
        lines.append(f"PCA wins on **{pca_win_layers}/{len(layer_deltas)}** layers  ")
        lines.append(f"WHT wins on **{wht_win_layers}/{len(layer_deltas)}** layers  ")
        lines.append("")

        lines.append("| Layer | CosSim delta | MSE ratio | Winner |")
        lines.append("|-------|-------------|-----------|--------|")
        for li in sorted(layer_deltas.keys()):
            d = layer_deltas[li]
            avg_d = sum(d["cos_delta"]) / len(d["cos_delta"])
            avg_r = sum(d["mse_ratio"]) / len(d["mse_ratio"])
            w = "PCA" if avg_d > 0.0001 else ("WHT" if avg_d < -0.0001 else "TIE")
            lines.append(f"| {li} | {avg_d:+.5f} | {avg_r:.4f} | {w} |")
    lines.append("")

    # Key findings
    lines.append("## 6. Key Findings")
    lines.append("")

    with open(path, "w") as f:
        f.write("\n".join(lines))

    print(f"\nResults saved to {path}")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print("=" * 70)
    print("LEARNED ROTATION EXPERIMENT")
    print("Can PCA beat random WHT for KV cache compression?")
    print("=" * 70)

    # 1. Load model
    model, tokenizer, n_layers, n_kv_heads, head_dim = load_model()

    # 2. Extract KV cache from Prompt A
    print("\nExtracting KV cache for Prompt A...")
    key_cache_a, seq_len = extract_kv_cache(model, tokenizer, PROMPT_A)

    # 3. Eigenvalue spectrum analysis
    layer_stats = analyse_eigenspectrum(key_cache_a, n_layers, n_kv_heads, head_dim)

    # 4. Full comparison: WHT vs PCA-uniform vs PCA-adaptive
    all_comparisons, aggregated = run_full_comparison(
        key_cache_a, n_layers, n_kv_heads, head_dim, BITS
    )

    # 5. Per-layer breakdown
    layer_deltas = run_layer_breakdown(all_comparisons, n_layers, n_kv_heads)

    # 6. Calibration size sensitivity
    calib_results = run_calibration_sweep(
        key_cache_a, n_layers, n_kv_heads, head_dim, BITS
    )

    # 7. Transfer test: PCA from Prompt A -> Prompt B
    transfer_results = run_transfer_test(
        model, tokenizer, n_layers, n_kv_heads, head_dim, BITS,
        key_cache_a,
    )

    # 8. Save results
    save_results(
        aggregated, calib_results, transfer_results, layer_stats,
        layer_deltas, n_layers, n_kv_heads, head_dim, seq_len,
    )

    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
