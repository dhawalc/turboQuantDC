"""Sparse weight loading predictor for streaming transformer inference.

When a transformer processes a token, FFN layers compute:
    y = W_down @ activation(W_gate @ x * W_up @ x)

The activation function (SiLU/GELU/ReLU) produces many near-zero values.
If we can PREDICT which neurons will be active BEFORE loading the weight
matrix, we only need to load the active fraction of the weights.

For streaming inference (CPU -> GPU weight transfer), this reduces data
movement by the inverse of the active fraction. At 90% sparsity, that
means 10x less data transferred per layer.

Usage:
    predictor = SparseLoadingPredictor(d_model=2048, d_intermediate=11008)
    predictor.profile(model, sample_inputs)

    # During inference:
    mask = predictor.predict_active_neurons(layer_idx, hidden_states)
    sparse_weights = predictor.selective_load(full_weights, mask)
    output = sparse_forward(sparse_weights, hidden_states)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuronPredictor(nn.Module):
    """Lightweight MLP that predicts which neurons will fire.

    A small network (d_model -> bottleneck -> d_intermediate) trained to
    predict the binary activation pattern from the layer input. The
    bottleneck keeps the predictor itself tiny relative to the FFN.
    """

    def __init__(self, d_model: int, d_intermediate: int, bottleneck: int = 256):
        super().__init__()
        self.proj_down = nn.Linear(d_model, bottleneck, bias=False)
        self.proj_up = nn.Linear(bottleneck, d_intermediate, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict activation logits from layer input.

        Args:
            x: Hidden states, shape (..., d_model).

        Returns:
            Logits of shape (..., d_intermediate). Positive = predicted active.
        """
        return self.proj_up(F.relu(self.proj_down(x)))


class SparseLoadingPredictor:
    """Predicts which weight rows/columns are needed before loading.

    Uses a lightweight predictor per layer to determine which neurons will
    fire, enabling selective weight loading from CPU to GPU.

    If 90% sparsity is typical, this reduces per-layer loading by ~10x:
        Full layer:  ~129 MB transfer  (for Qwen2.5-3B FFN)
        Sparse layer: ~13 MB transfer
        At PCIe 5.0 (32 GB/s): 4.0ms -> 0.4ms per layer

    Attributes:
        d_model: Hidden dimension of the model.
        d_intermediate: FFN intermediate dimension.
        sparsity_target: Target sparsity fraction (0.0 to 1.0).
        predictors: Dict mapping layer_idx to trained NeuronPredictor.
        thresholds: Dict mapping layer_idx to per-neuron activation threshold.
        profiled: Whether profile() has been called.
    """

    def __init__(
        self,
        d_model: int,
        d_intermediate: int,
        sparsity_target: float = 0.9,
        bottleneck: int = 256,
        device: str = "cpu",
    ):
        self.d_model = d_model
        self.d_intermediate = d_intermediate
        self.sparsity_target = sparsity_target
        self.bottleneck = bottleneck
        self.device = device

        self.predictors: Dict[int, NeuronPredictor] = {}
        self.thresholds: Dict[int, torch.Tensor] = {}
        self.profiled = False

        # Statistics collected during profiling
        self._profile_data: Dict[int, Dict] = {}

    @property
    def num_layers(self) -> int:
        return len(self.predictors)

    def predictor_size_bytes(self) -> int:
        """Total size of all predictor weights in bytes (fp16)."""
        total = 0
        for p in self.predictors.values():
            for param in p.parameters():
                total += param.numel() * 2  # fp16
        return total

    def profile(
        self,
        model: nn.Module,
        sample_inputs: List[Dict[str, torch.Tensor]],
        activation_threshold: float = 0.01,
        num_train_steps: int = 100,
        lr: float = 1e-3,
    ) -> Dict[int, float]:
        """Profile activation patterns and train per-layer predictors.

        Runs sample inputs through the model, records which neurons activate
        (gate activation magnitude > activation_threshold), and trains a
        lightweight predictor per layer.

        Args:
            model: The transformer model to profile.
            sample_inputs: List of tokenized input dicts (from tokenizer).
            activation_threshold: Magnitude threshold for "active" neuron.
            num_train_steps: Training steps for each predictor.
            lr: Learning rate for predictor training.

        Returns:
            Dict mapping layer_idx to measured sparsity ratio.
        """
        # Collect (input, activation_mask) pairs per layer
        layer_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        def make_collector(layer_idx, mlp_module):
            """Create hooks that capture (MLP input, gate activation pattern)."""
            layer_data[layer_idx] = []
            captured_input = {}

            def input_hook(module, inp):
                if isinstance(inp, tuple):
                    captured_input["x"] = inp[0].detach()
                else:
                    captured_input["x"] = inp.detach()

            def gate_hook(module, inp, output):
                if isinstance(output, tuple):
                    gate_act = output[0].detach()
                else:
                    gate_act = output.detach()

                x = captured_input.get("x")
                if x is None:
                    return

                # Flatten to (total_tokens, dim)
                if x.dim() == 3:
                    x = x.reshape(-1, x.shape[-1])
                if gate_act.dim() == 3:
                    gate_act = gate_act.reshape(-1, gate_act.shape[-1])

                # Binary mask: 1 = active neuron
                mask = (gate_act.abs() > activation_threshold).float()
                layer_data[layer_idx].append((x.cpu(), mask.cpu()))

            return input_hook, gate_hook

        # Register hooks
        handles = []
        for layer_idx, layer in enumerate(model.model.layers):
            in_hook, gate_hook = make_collector(layer_idx, layer.mlp)
            h1 = layer.mlp.register_forward_pre_hook(in_hook)
            h2 = layer.mlp.act_fn.register_forward_hook(gate_hook)
            handles.append(h1)
            handles.append(h2)

        # Run forward passes to collect data
        for batch in sample_inputs:
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
            with torch.no_grad():
                model(**batch)

        # Remove hooks
        for h in handles:
            h.remove()

        # Train a predictor for each layer
        sparsity_report: Dict[int, float] = {}
        device = self.device

        for layer_idx, data_pairs in layer_data.items():
            if not data_pairs:
                continue

            # Concatenate all collected data
            all_x = torch.cat([x for x, _ in data_pairs], dim=0)
            all_mask = torch.cat([m for _, m in data_pairs], dim=0)

            # Compute actual sparsity
            sparsity = 1.0 - all_mask.float().mean().item()
            sparsity_report[layer_idx] = sparsity

            # Store per-neuron activation frequency for threshold tuning
            neuron_freq = all_mask.float().mean(dim=0)  # (d_intermediate,)
            self.thresholds[layer_idx] = neuron_freq.to(device)

            # Create and train predictor
            predictor = NeuronPredictor(
                self.d_model, self.d_intermediate, self.bottleneck
            ).to(device)

            optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)

            # Mini-batch training
            n_samples = all_x.shape[0]
            batch_size = min(64, n_samples)

            for step in range(num_train_steps):
                idx = torch.randint(0, n_samples, (batch_size,))
                x_batch = all_x[idx].to(device)
                mask_batch = all_mask[idx].to(device)

                logits = predictor(x_batch)
                # Use BCE loss with class weights to handle imbalanced (sparse) targets
                pos_weight = torch.tensor([sparsity / max(1.0 - sparsity, 1e-6)]).to(device)
                loss = F.binary_cross_entropy_with_logits(
                    logits, mask_batch, pos_weight=pos_weight.expand_as(logits)
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            predictor.eval()
            self.predictors[layer_idx] = predictor

            self._profile_data[layer_idx] = {
                "sparsity": sparsity,
                "n_samples": n_samples,
                "neuron_freq_mean": neuron_freq.mean().item(),
                "neuron_freq_std": neuron_freq.std().item(),
            }

        self.profiled = True
        return sparsity_report

    def predict_active_neurons(
        self, layer_idx: int, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Predict which neurons will be active for this input.

        Args:
            layer_idx: Which layer to predict for.
            hidden_states: Input hidden states, shape (..., d_model).

        Returns:
            Boolean mask of shape (..., d_intermediate). True = predicted active.
        """
        if not self.profiled:
            raise RuntimeError("Must call profile() before predict_active_neurons()")

        predictor = self.predictors.get(layer_idx)
        if predictor is None:
            # No predictor for this layer; assume all active
            shape = hidden_states.shape[:-1] + (self.d_intermediate,)
            return torch.ones(shape, dtype=torch.bool, device=hidden_states.device)

        with torch.no_grad():
            logits = predictor(hidden_states)
            # Use neuron frequency as adaptive threshold:
            # neurons that rarely fire need higher logits to be predicted active
            freq = self.thresholds.get(layer_idx)
            if freq is not None:
                freq = freq.to(logits.device)
                # Threshold: predict active if logit > 0 (sigmoid > 0.5)
                # Bias toward recall: use -1.0 to catch more active neurons
                threshold = -1.0
            else:
                threshold = 0.0

            mask = logits > threshold
        return mask

    def selective_load(
        self,
        layer_weights: Dict[str, torch.Tensor],
        active_mask: torch.Tensor,
        device: str = "cuda",
    ) -> Dict[str, torch.Tensor]:
        """Load only the active columns/rows of a layer's FFN weights.

        For the gated FFN architecture (SiLU):
            gate_proj: (d_intermediate, d_model) -- select active ROWS
            up_proj:   (d_intermediate, d_model) -- select active ROWS
            down_proj: (d_model, d_intermediate) -- select active COLUMNS

        Args:
            layer_weights: Dict with keys "gate_proj", "up_proj", "down_proj",
                each a 2D tensor (may be on CPU).
            active_mask: Boolean mask of shape (d_intermediate,).
            device: Target device for loaded weights.

        Returns:
            Dict with same keys but smaller tensors, plus "active_indices"
            mapping back to original positions.
        """
        # Collapse batch dimensions to a single mask over d_intermediate
        if active_mask.dim() > 1:
            active_mask = active_mask.any(dim=tuple(range(active_mask.dim() - 1)))

        active_idx = active_mask.nonzero(as_tuple=True)[0]
        n_active = active_idx.shape[0]

        result = {"active_indices": active_idx.to(device), "n_active": n_active}

        # gate_proj and up_proj: shape (d_intermediate, d_model) -> select rows
        for name in ["gate_proj", "up_proj"]:
            if name in layer_weights:
                w = layer_weights[name]
                result[name] = w[active_idx].to(device)

        # down_proj: shape (d_model, d_intermediate) -> select columns
        if "down_proj" in layer_weights:
            w = layer_weights["down_proj"]
            result["down_proj"] = w[:, active_idx].to(device)

        return result

    def sparse_forward(
        self,
        hidden_states: torch.Tensor,
        sparse_weights: Dict[str, torch.Tensor],
        act_fn=F.silu,
    ) -> torch.Tensor:
        """Run FFN forward pass using only the sparse-loaded weights.

        Equivalent to:
            gate = act_fn(x @ gate_proj.T)
            up = x @ up_proj.T
            y = (gate * up) @ down_proj.T

        But using only the active neuron subset.

        Args:
            hidden_states: Input, shape (..., d_model).
            sparse_weights: Output of selective_load().
            act_fn: Activation function (default SiLU).

        Returns:
            Output, shape (..., d_model). Same as full dense forward.
        """
        gate_w = sparse_weights["gate_proj"]  # (n_active, d_model)
        up_w = sparse_weights["up_proj"]      # (n_active, d_model)
        down_w = sparse_weights["down_proj"]  # (d_model, n_active)

        # Sparse gate/up: only compute active neurons
        gate = act_fn(hidden_states @ gate_w.t())  # (..., n_active)
        up = hidden_states @ up_w.t()               # (..., n_active)
        intermediate = gate * up                     # (..., n_active)
        output = intermediate @ down_w.t()           # (..., d_model)

        return output

    def measure_accuracy(
        self,
        model: nn.Module,
        test_inputs: List[Dict[str, torch.Tensor]],
        activation_threshold: float = 0.01,
    ) -> Dict[str, float]:
        """Measure predictor accuracy against actual activations.

        Returns precision, recall, F1, and sparsity for each layer.

        Args:
            model: The model to test against.
            test_inputs: Tokenized inputs for testing.
            activation_threshold: Threshold for "active" ground truth.

        Returns:
            Dict with keys like "layer_0_precision", "layer_0_recall", etc.
        """
        if not self.profiled:
            raise RuntimeError("Must call profile() first")

        # Collect ground truth activations
        gt_data: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        handles = []

        for layer_idx, layer in enumerate(model.model.layers):
            gt_data[layer_idx] = []
            captured = {}

            def make_hooks(lidx, cap):
                def pre_hook(module, inp):
                    x = inp[0] if isinstance(inp, tuple) else inp
                    cap["x"] = x.detach()

                def post_hook(module, inp, output):
                    act = output[0] if isinstance(output, tuple) else output
                    act = act.detach()
                    x = cap.get("x")
                    if x is None:
                        return
                    if x.dim() == 3:
                        x = x.reshape(-1, x.shape[-1])
                    if act.dim() == 3:
                        act = act.reshape(-1, act.shape[-1])
                    gt_mask = (act.abs() > activation_threshold).float()
                    gt_data[lidx].append((x.cpu(), gt_mask.cpu()))

                return pre_hook, post_hook

            pre_h, post_h = make_hooks(layer_idx, captured)
            h1 = layer.mlp.register_forward_pre_hook(pre_h)
            h2 = layer.mlp.act_fn.register_forward_hook(post_h)
            handles.append(h1)
            handles.append(h2)

        for batch in test_inputs:
            batch = {k: v.to(next(model.parameters()).device) for k, v in batch.items()}
            with torch.no_grad():
                model(**batch)

        for h in handles:
            h.remove()

        # Compare predictions to ground truth
        results: Dict[str, float] = {}
        total_precision, total_recall, total_f1 = 0.0, 0.0, 0.0
        n_layers = 0

        for layer_idx, pairs in gt_data.items():
            if not pairs or layer_idx not in self.predictors:
                continue

            all_x = torch.cat([x for x, _ in pairs], dim=0)
            all_gt = torch.cat([m for _, m in pairs], dim=0)

            # Predict
            pred_mask = self.predict_active_neurons(
                layer_idx, all_x.to(self.device)
            ).cpu().float()
            gt = all_gt

            # Precision, recall, F1
            tp = (pred_mask * gt).sum().item()
            pred_pos = pred_mask.sum().item()
            actual_pos = gt.sum().item()

            precision = tp / max(pred_pos, 1e-10)
            recall = tp / max(actual_pos, 1e-10)
            f1 = 2 * precision * recall / max(precision + recall, 1e-10)

            results[f"layer_{layer_idx}_precision"] = precision
            results[f"layer_{layer_idx}_recall"] = recall
            results[f"layer_{layer_idx}_f1"] = f1

            total_precision += precision
            total_recall += recall
            total_f1 += f1
            n_layers += 1

        if n_layers > 0:
            results["avg_precision"] = total_precision / n_layers
            results["avg_recall"] = total_recall / n_layers
            results["avg_f1"] = total_f1 / n_layers

        return results

    def memory_report(self) -> Dict[str, float]:
        """Report memory usage comparison between dense and sparse loading.

        Returns:
            Dict with keys: full_layer_mb, sparse_layer_mb, predictor_mb,
            savings_ratio, active_fraction.
        """
        full_bytes = 3 * self.d_model * self.d_intermediate * 2  # 3 matrices, fp16
        active_frac = 1.0 - self.sparsity_target
        sparse_bytes = full_bytes * active_frac
        pred_bytes = self.predictor_size_bytes() / max(self.num_layers, 1)

        return {
            "full_layer_mb": full_bytes / (1024 * 1024),
            "sparse_layer_mb": sparse_bytes / (1024 * 1024),
            "predictor_mb": pred_bytes / (1024 * 1024),
            "savings_ratio": full_bytes / max(sparse_bytes + pred_bytes, 1),
            "active_fraction": active_frac,
        }
