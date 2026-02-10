from __future__ import annotations

from typing import Any

from pycocotools.cocoeval import COCOeval


COCO_METRIC_NAMES = [
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
]


def _summarize_stats(stats: list[float]) -> dict[str, float]:
    # COCOeval.stats is length 12; first 6 are standard AP numbers.
    out = {}
    for i, k in enumerate(COCO_METRIC_NAMES):
        out[k] = float(stats[i])
    return out


def evaluate_coco(coco_gt, coco_results: list[dict[str, Any]], iou_type: str) -> dict[str, float]:
    coco_dt = coco_gt.loadRes(coco_results) if coco_results else coco_gt.loadRes([])
    ev = COCOeval(coco_gt, coco_dt, iouType=iou_type)
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return _summarize_stats(list(ev.stats))

