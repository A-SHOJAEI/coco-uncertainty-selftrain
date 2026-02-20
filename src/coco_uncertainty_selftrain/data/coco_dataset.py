from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from torch.utils.data import Dataset


class CocoInstanceSegmentation(Dataset):
    """COCO-format instance segmentation dataset for Mask R-CNN."""

    def __init__(
        self,
        coco_root: Path,
        ann_file: Path,
        image_ids: List[int],
        transforms: Optional[Any] = None,
    ) -> None:
        self.coco_root = Path(coco_root)
        self.coco = COCO(str(ann_file))
        self.image_ids = sorted(image_ids)
        self.transforms = transforms

        # Build a category mapping: COCO id -> contiguous [1..N].
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_contig = {cid: i + 1 for i, cid in enumerate(cat_ids)}

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]

        # Try several possible image paths.
        img_path = self.coco_root / img_info["file_name"]
        if not img_path.exists():
            # Try under images/ subdirectory
            for sub in ["images/train", "images/val", "images", "train2017", "val2017"]:
                candidate = self.coco_root / sub / img_info["file_name"]
                if candidate.exists():
                    img_path = candidate
                    break

        img = Image.open(str(img_path)).convert("RGB")
        w_img, h_img = img.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []
        iscrowd = []
        weights = []

        for ann in anns:
            x, y, bw, bh = ann["bbox"]
            if bw < 1 or bh < 1:
                continue
            boxes.append([x, y, x + bw, y + bh])
            contig = self.cat_id_to_contig.get(int(ann["category_id"]), 1)
            labels.append(contig)
            mask = self.coco.annToMask(ann)
            masks.append(torch.from_numpy(mask).to(torch.uint8))
            areas.append(ann.get("area", bw * bh))
            iscrowd.append(ann.get("iscrowd", 0))
            weights.append(float(ann.get("weight", 1.0)))

        target: Dict[str, Any] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) if boxes else torch.zeros((0, 4), dtype=torch.float32),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": torch.stack(masks) if masks else torch.zeros((0, h_img, w_img), dtype=torch.uint8),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(iscrowd, dtype=torch.int64),
            "instance_weights": torch.as_tensor(weights, dtype=torch.float32),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[List[torch.Tensor], List[Dict[str, Any]]]:
    images = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    return images, targets
