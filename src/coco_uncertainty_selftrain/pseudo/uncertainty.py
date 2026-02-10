from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class UncertaintyConfig:
    signal: str  # entropy | margin | mask_mean_conf | mc_dropout_var
    num_classes: int  # includes background


def approx_entropy_from_top1(score: float, num_classes: int) -> float:
    """
    Torchvision's Mask R-CNN inference only exposes the top-1 class score.
    We approximate the full distribution by placing remaining mass uniformly on other classes.

    Returns normalized entropy in [0, 1].
    """
    k = max(2, int(num_classes) - 1)  # exclude background; at least 2 to avoid log(1)
    p1 = min(max(score, 1e-6), 1.0 - 1e-6)
    prem = (1.0 - p1) / (k - 1) if k > 1 else 1.0 - p1
    ps = [p1] + [prem] * (k - 1)
    h = -sum(p * math.log(p) for p in ps)
    return float(h / math.log(k))


def approx_margin_from_top1(score: float, num_classes: int) -> float:
    """
    Approximate top1-top2 margin with the same uniform-rest assumption as entropy.
    """
    k = max(2, int(num_classes) - 1)
    p1 = min(max(score, 0.0), 1.0)
    p2 = (1.0 - p1) / (k - 1) if k > 1 else 0.0
    return float(max(0.0, p1 - p2))


def weight_from_uncertainty(signal: str, *, score: float, mask_mean_conf: float | None, mc_var: float | None, num_classes: int) -> float:
    if signal == "entropy":
        u = approx_entropy_from_top1(score, num_classes)
        return float(max(0.0, min(1.0, 1.0 - u)))
    if signal == "margin":
        m = approx_margin_from_top1(score, num_classes)
        return float(max(0.0, min(1.0, m)))
    if signal == "mask_mean_conf":
        if mask_mean_conf is None:
            return float(score)
        return float(max(0.0, min(1.0, mask_mean_conf)))
    if signal == "mc_dropout_var":
        # Higher variance => lower weight. Scale with a soft inverse.
        if mc_var is None:
            return float(score)
        return float(1.0 / (1.0 + mc_var))
    raise ValueError(f"Unknown uncertainty signal: {signal}")


def mask_mean_confidence(mask_probs: torch.Tensor, mask_thresh: float = 0.5) -> float:
    """
    mask_probs: (H, W) in [0,1]
    """
    m = mask_probs >= mask_thresh
    if m.sum().item() == 0:
        return float(mask_probs.mean().item())
    return float(mask_probs[m].mean().item())

