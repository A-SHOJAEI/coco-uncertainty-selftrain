#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
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
from coco_uncertainty_selftrain.models.maskrcnn import build_maskrcnn, set_dropout_train_only
from coco_uncertainty_selftrain.pseudo.uncertainty import mask_mean_confidence, weight_from_uncertainty
from coco_uncertainty_selftrain.utils.io import mkdirp, read_json, read_yaml, write_json
from coco_uncertainty_selftrain.utils.meta import collect_meta
from coco_uncertainty_selftrain.utils.repro import ReproConfig, seed_everything


def _device_from_config(cfg: dict) -> torch.device:
    d = cfg.get("device", "cpu")
    if d == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)


def _encode_binary_mask(mask: np.ndarray) -> dict:
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    # pycocotools returns bytes for historical reasons.
    if isinstance(rle["counts"], (bytes, bytearray)):
        rle["counts"] = rle["counts"].decode("ascii")
    return rle


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def _mc_variance_for_det(base_det, other_dets, iou_thresh: float = 0.5) -> float:
    # base_det: dict with box (xyxy), label (int), score (float)
    base_box = base_det["box"]
    base_label = base_det["label"]
    scores = [base_det["score"]]
    for dets in other_dets:
        best = None
        best_iou = -1.0
        for d in dets:
            if d["label"] != base_label:
                continue
            iou = _iou_xyxy(base_box, d["box"])
            if iou > best_iou:
                best_iou = iou
                best = d
        if best is None or best_iou < iou_thresh:
            scores.append(0.0)
        else:
            scores.append(best["score"])
    return float(np.var(np.array(scores, dtype=np.float32)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--unlabeled_split", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--uncertainty", choices=["entropy", "margin", "mask_mean_conf", "mc_dropout_var"], default="entropy")
    ap.add_argument("--mc_dropout", type=int, default=0, help="Number of stochastic passes (0 disables MC dropout)")
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--filter", choices=["fixed_thresh", "adaptive_class", "topk_per_image"], default="fixed_thresh")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--adaptive_keep", type=float, default=0.2, help="Keep fraction per class for adaptive_class")
    ap.add_argument("--no_uncertainty_weighting", action="store_true")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    seed_everything(ReproConfig(seed=int(cfg.get("seed", 1337)), deterministic=True))
    device = _device_from_config(cfg)

    unl = read_json(args.unlabeled_split)
    image_ids = [int(x) for x in unl["image_ids"]]

    coco_root = Path(cfg["data"]["coco_root"])
    train_ann = coco_root / cfg["data"]["train_ann"]
    if Path(cfg["data"]["train_ann"]).is_absolute():
        train_ann = Path(cfg["data"]["train_ann"])

    coco_gt = COCO(str(train_ann))
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    cat_ids = sorted([c["id"] for c in cats])
    cat_id_to_contig = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
    contig_to_cat_id = {v: k for k, v in cat_id_to_contig.items()}
    num_classes = int(cfg["model"].get("num_classes", len(cat_ids) + 1))

    model = build_maskrcnn(num_classes=num_classes, mc_dropout_p=float(cfg["model"].get("mc_dropout_p", 0.0)))
    try:
        state = torch.load(args.weights, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    mkdirp(args.out_dir)

    # For pseudo-label generation, use no geometry transforms to avoid misalignment.
    ds = CocoInstanceSegmentation(
        coco_root=coco_root,
        ann_file=train_ann,
        image_ids=image_ids,
        transforms=build_transforms(augment="none"),
    )

    # Optional first pass to compute per-class adaptive thresholds.
    class_thresh = {cid: args.score_thresh for cid in cat_ids}
    if args.filter == "adaptive_class":
        per_class_scores: dict[int, list[float]] = {cid: [] for cid in cat_ids}
        with torch.no_grad():
            for img, _t in ds:
                img = img.to(device)
                out = model([img])[0]
                labels = out["labels"].detach().cpu().numpy().tolist()
                scores = out["scores"].detach().cpu().numpy().tolist()
                for lab, sc in zip(labels, scores):
                    cid = contig_to_cat_id.get(int(lab))
                    if cid is None:
                        continue
                    per_class_scores[cid].append(float(sc))
        for cid, scores in per_class_scores.items():
            if not scores:
                class_thresh[cid] = 1.0
                continue
            scores_sorted = sorted(scores)
            k = max(1, int(math.ceil((1.0 - float(args.adaptive_keep)) * len(scores_sorted))))
            class_thresh[cid] = float(scores_sorted[min(len(scores_sorted) - 1, k - 1)])

    pseudo_images = []
    pseudo_anns = []
    ann_id = 1
    stats = {"accepted": 0, "seen": 0, "per_class": {str(cid): 0 for cid in cat_ids}}

    meta = collect_meta()
    meta.update(
        {
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "uncertainty": args.uncertainty,
            "mc_dropout": int(args.mc_dropout),
            "score_thresh": float(args.score_thresh),
            "filter": args.filter,
            "no_uncertainty_weighting": bool(args.no_uncertainty_weighting),
            "class_thresh": class_thresh if args.filter == "adaptive_class" else None,
        }
    )

    with torch.no_grad():
        for idx in range(len(ds)):
            img, _t = ds[idx]
            image_id = int(ds.image_ids[idx])
            img_info = coco_gt.loadImgs([image_id])[0]

            img_dev = img.to(device)

            if args.mc_dropout and args.mc_dropout > 0:
                # Keep dropout stochastic while the model is otherwise in eval mode.
                set_dropout_train_only(model, True)
                outs = [model([img_dev])[0] for _ in range(int(args.mc_dropout) + 1)]
                set_dropout_train_only(model, False)
            else:
                outs = [model([img_dev])[0]]

            base = outs[0]
            stats["seen"] += int(base["scores"].shape[0])

            # Convert outputs to lightweight python dicts for MC matching.
            other_for_mc = []
            for o in outs[1:]:
                dets = []
                for b, l, s in zip(
                    o["boxes"].detach().cpu().numpy(),
                    o["labels"].detach().cpu().numpy(),
                    o["scores"].detach().cpu().numpy(),
                ):
                    dets.append({"box": b.astype(np.float32), "label": int(l), "score": float(s)})
                other_for_mc.append(dets)

            scores = base["scores"].detach().cpu().numpy()
            order = np.argsort(-scores)
            if args.filter == "topk_per_image":
                order = order[: int(args.topk)]

            accepted_this_img = 0
            for j in order.tolist():
                score = float(base["scores"][j].item())
                label = int(base["labels"][j].item())
                cat_id = contig_to_cat_id.get(label)
                if cat_id is None:
                    continue

                if args.filter == "fixed_thresh":
                    if score < float(args.score_thresh):
                        continue
                elif args.filter == "adaptive_class":
                    if score < float(class_thresh.get(cat_id, 1.0)):
                        continue
                elif args.filter == "topk_per_image":
                    pass

                box = base["boxes"][j].detach().cpu().numpy().astype(np.float32)
                xywh = [float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1])]

                mask_prob = base["masks"][j, 0].detach().cpu()
                mmc = mask_mean_confidence(mask_prob)
                mask_bin = (mask_prob >= 0.5).numpy().astype(np.uint8)
                rle = _encode_binary_mask(mask_bin)
                area = float(mask_utils.area(rle))

                mc_var = None
                if args.uncertainty == "mc_dropout_var" and other_for_mc:
                    base_det = {"box": box, "label": label, "score": score}
                    mc_var = _mc_variance_for_det(base_det, other_for_mc)

                w = 1.0 if args.no_uncertainty_weighting else weight_from_uncertainty(
                    args.uncertainty,
                    score=score,
                    mask_mean_conf=mmc,
                    mc_var=mc_var,
                    num_classes=num_classes,
                )

                pseudo_anns.append(
                    {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": int(cat_id),
                        "bbox": xywh,
                        "area": float(area),
                        "segmentation": rle,
                        "iscrowd": 0,
                        "score": float(score),
                        "weight": float(w),
                    }
                )
                ann_id += 1
                accepted_this_img += 1
                stats["accepted"] += 1
                stats["per_class"][str(cat_id)] += 1

            if accepted_this_img > 0:
                pseudo_images.append(
                    {
                        "id": int(img_info["id"]),
                        "file_name": img_info["file_name"],
                        "width": int(img_info["width"]),
                        "height": int(img_info["height"]),
                    }
                )

    out_json = args.out_dir / "pseudo_instances.json"
    write_json(
        out_json,
        {
            "info": {"description": "Pseudo instances", "meta": meta},
            "images": pseudo_images,
            "annotations": pseudo_anns,
            "categories": coco_gt.dataset.get("categories", []),
            "stats": stats,
        },
    )
    write_json(args.out_dir / "pseudo_stats.json", {"meta": meta, "stats": stats})
    print(f"Wrote pseudo labels: {out_json} (accepted={stats['accepted']}, seen={stats['seen']})")


if __name__ == "__main__":
    main()
