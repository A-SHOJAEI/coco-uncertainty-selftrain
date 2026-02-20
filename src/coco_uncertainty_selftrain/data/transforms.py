from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
from torchvision import transforms as T


class ComposeWithTargets:
    """Apply a list of transforms that accept (image, target) and return (image, target)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor:
    def __call__(self, image, target):
        image = T.functional.to_tensor(image)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, image, target):
        if torch.rand(1).item() < self.p:
            image = T.functional.hflip(image)
            _, h, w = image.shape
            boxes = target["boxes"]
            if boxes.numel() > 0:
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
                target["boxes"] = boxes
            if "masks" in target and target["masks"].numel() > 0:
                target["masks"] = target["masks"].flip(-1)
        return image, target


class PhotometricJitter:
    def __init__(self, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1):
        self.jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, target):
        # ColorJitter works on PIL or tensor; convert to PIL if needed.
        from PIL import Image
        if isinstance(image, torch.Tensor):
            image = T.functional.to_pil_image(image)
            image = self.jitter(image)
            image = T.functional.to_tensor(image)
        else:
            image = self.jitter(image)
        return image, target


def build_transforms(augment: str = "none") -> ComposeWithTargets:
    """Build transform pipeline.

    augment: "none" | "weak" | "strong"
    """
    transforms = []

    if augment == "none":
        transforms.append(ToTensor())
    elif augment == "weak":
        transforms.append(ToTensor())
        transforms.append(RandomHorizontalFlip(p=0.5))
    elif augment == "strong":
        transforms.append(ToTensor())
        transforms.append(RandomHorizontalFlip(p=0.5))
        transforms.append(PhotometricJitter())
    else:
        raise ValueError(f"Unknown augment mode: {augment}")

    return ComposeWithTargets(transforms)
