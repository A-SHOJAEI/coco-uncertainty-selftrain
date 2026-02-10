#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from coco_uncertainty_selftrain.data.coco_dataset import CocoInstanceSegmentation
from coco_uncertainty_selftrain.data.transforms import build_transforms
from coco_uncertainty_selftrain.eval.coco_eval import evaluate_coco
from coco_uncertainty_selftrain.models.maskrcnn import build_maskrcnn
from coco_uncertainty_selftrain.utils.io import mkdirp, read_yaml, write_json
from coco_uncertainty_selftrain.utils.meta import collect_meta
from coco_uncertainty_selftrain.utils.repro import ReproConfig, seed_everything


def _device_from_config(cfg: dict) -> torch.device:
    d = cfg.get("device", "cpu")
    if d == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)


def _encode(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    if isinstance(rle["counts"], (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--split", choices=["val"], default="val")
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    seed_everything(ReproConfig(seed=int(cfg.get("seed", 1337)), deterministic=True))
    device = _device_from_config(cfg)

    coco_root = Path(cfg["data"]["coco_root"])
    val_ann = coco_root / cfg["data"]["val_ann"]
    if Path(cfg["data"]["val_ann"]).is_absolute():
        val_ann = Path(cfg["data"]["val_ann"])

    coco_gt = COCO(str(val_ann))
    cat_ids = sorted(coco_gt.getCatIds())
    cat_id_to_contig = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
    contig_to_cat_id = {v: k for k, v in cat_id_to_contig.items()}

    ds = CocoInstanceSegmentation(
        coco_root=coco_root,
        ann_file=val_ann,
        image_ids=sorted(list(coco_gt.imgs.keys())),
        transforms=build_transforms(augment="none"),
    )
    max_images = int(cfg.get("eval", {}).get("max_images", len(ds)))

    num_classes = int(cfg["model"].get("num_classes", len(cat_ids) + 1))
    model = build_maskrcnn(num_classes=num_classes, mc_dropout_p=0.0)
    try:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    results_segm = []
    results_bbox = []
    with torch.no_grad():
        for i in range(min(len(ds), max_images)):
            img, _t = ds[i]
            image_id = int(ds.image_ids[i])
            out = model([img.to(device)])[0]

            boxes = out["boxes"].detach().cpu().numpy().astype(np.float32)
            labels = out["labels"].detach().cpu().numpy().astype(np.int64)
            scores = out["scores"].detach().cpu().numpy().astype(np.float32)
            masks = out["masks"].detach().cpu().numpy().astype(np.float32)  # (N,1,H,W)

            for b, l, s, m in zip(boxes, labels, scores, masks):
                cat_id = contig_to_cat_id.get(int(l))
                if cat_id is None:
                    continue
                xywh = [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])]
                mask_bin = (m[0] >= 0.5).astype(np.uint8)
                segm = _encode(mask_bin)

                results_segm.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(cat_id),
                        "segmentation": segm,
                        "score": float(s),
                    }
                )
                results_bbox.append(
                    {
                        "image_id": int(image_id),
                        "category_id": int(cat_id),
                        "bbox": xywh,
                        "score": float(s),
                    }
                )

    metrics = {
        "segm": evaluate_coco(coco_gt, results_segm, "segm"),
        "bbox": evaluate_coco(coco_gt, results_bbox, "bbox"),
    }

    out_dir = mkdirp(args.out_dir)
    payload = {
        "metrics": metrics,
        "meta": {
            **collect_meta(),
            "evaluated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "weights": str(args.weights),
            "config": str(args.config),
            "split": args.split,
            "max_images": int(max_images),
        },
        "notes": "",
    }
    write_json(out_dir / "eval.json", payload)
    print(f"Wrote {out_dir / 'eval.json'}")


if __name__ == "__main__":
    main()
