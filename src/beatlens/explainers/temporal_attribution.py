"""Gradient and temporal perturbation explanations for scalar ECG model outputs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import Ridge

ScalarPredictor = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class LimeResult:
    """Temporal LIME coefficients and weighted local-fit quality."""

    coefficients: np.ndarray
    intercept: float
    weighted_r2: float
    segment_samples: int


def integrated_gradients(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    output_ms: Callable[[torch.Tensor], torch.Tensor],
    *,
    baseline: torch.Tensor | None = None,
    steps: int = 64,
) -> tuple[torch.Tensor, float]:
    """Return input attributions and the absolute completeness residual in ms."""
    if steps < 2:
        raise ValueError("steps must be at least 2")
    baseline = torch.zeros_like(inputs) if baseline is None else baseline
    difference = inputs - baseline
    gradients: list[torch.Tensor] = []
    alphas = torch.linspace(0.0, 1.0, steps + 1, device=inputs.device)
    model.eval()
    for alpha in alphas:
        interpolated = (baseline + alpha * difference).detach().requires_grad_(True)
        output = output_ms(model(interpolated)).sum()
        gradient = torch.autograd.grad(output, interpolated, retain_graph=False)[0]
        gradients.append(gradient)
    stacked = torch.stack(gradients)
    weights = torch.ones(steps + 1, device=inputs.device)
    weights[[0, -1]] = 0.5
    average_gradient = (stacked * weights[:, None, None, None]).sum(dim=0) / steps
    attribution = difference * average_gradient
    with torch.no_grad():
        output_difference = (output_ms(model(inputs)) - output_ms(model(baseline))).item()
    completeness_residual = abs(float(attribution.sum().item()) - float(output_difference))
    return attribution.detach(), completeness_residual


def temporal_lime(
    inputs: np.ndarray,
    predict: ScalarPredictor,
    *,
    segment_samples: int,
    replacement: np.ndarray | None = None,
    perturbations: int = 1024,
    kernel_width: float = 0.25,
    alpha: float = 1.0,
    seed: int = 0,
) -> LimeResult:
    """Fit a weighted linear surrogate over contiguous, zero-replaced time segments."""
    if inputs.ndim != 2:
        raise ValueError("inputs must have shape (leads, samples)")
    if segment_samples < 1 or perturbations < 2:
        raise ValueError("segment_samples must be positive and perturbations at least 2")
    replacement = np.zeros_like(inputs) if replacement is None else replacement
    if replacement.shape != inputs.shape:
        raise ValueError("replacement must have the same shape as inputs")
    n_samples = inputs.shape[1]
    n_segments = int(np.ceil(n_samples / segment_samples))
    rng = np.random.default_rng(seed)
    masks = (rng.random((perturbations, n_segments)) < 0.8).astype(np.float32)
    masks[0] = 1.0
    masked_inputs = np.broadcast_to(
        replacement.astype(np.float32), (perturbations, *inputs.shape)
    ).copy()
    for segment in range(n_segments):
        start = segment * segment_samples
        end = min(start + segment_samples, n_samples)
        present = masks[:, segment].astype(bool)
        masked_inputs[present, :, start:end] = inputs[None, :, start:end]
    outcomes = np.asarray(predict(masked_inputs), dtype=np.float64).reshape(-1)

    distances = np.sqrt(1.0 - masks.mean(axis=1))
    weights = np.exp(-((distances / kernel_width) ** 2))
    regressor = Ridge(alpha=alpha, fit_intercept=True)
    regressor.fit(masks, outcomes, sample_weight=weights)
    predictions = regressor.predict(masks)
    weighted_mean = float(np.average(outcomes, weights=weights))
    residual_sum = float(np.sum(weights * (outcomes - predictions) ** 2))
    total_sum = float(np.sum(weights * (outcomes - weighted_mean) ** 2))
    weighted_r2 = 1.0 - residual_sum / total_sum if total_sum > 0 else 1.0
    return LimeResult(
        coefficients=np.asarray(regressor.coef_, dtype=np.float64),
        intercept=float(regressor.intercept_),
        weighted_r2=weighted_r2,
        segment_samples=segment_samples,
    )
