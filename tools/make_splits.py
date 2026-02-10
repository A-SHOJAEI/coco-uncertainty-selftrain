#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

from pycocotools.coco import COCO

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coco_uncertainty_selftrain.utils.io import mkdirp, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coco_root", type=Path, required=True)
    ap.add_argument("--ann_file", type=Path, required=True)
    ap.add_argument("--labeled_fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out_dir", type=Path, required=True)
    args = ap.parse_args()

    coco = COCO(str(args.ann_file))
    image_ids = sorted(list(coco.imgs.keys()))
    if not image_ids:
        raise SystemExit(f"No images found in {args.ann_file}")

    rng = random.Random(args.seed)
    rng.shuffle(image_ids)

    n_labeled = max(1, int(round(len(image_ids) * float(args.labeled_fraction))))
    labeled = sorted(image_ids[:n_labeled])
    unlabeled = sorted(image_ids[n_labeled:])

    mkdirp(args.out_dir)
    common = {
        "coco_root": str(args.coco_root),
        "ann_file": str(args.ann_file),
        "seed": int(args.seed),
        "labeled_fraction": float(args.labeled_fraction),
    }
    write_json(args.out_dir / "labeled.json", {**common, "split": "labeled", "image_ids": labeled})
    write_json(args.out_dir / "unlabeled.json", {**common, "split": "unlabeled", "image_ids": unlabeled})


if __name__ == "__main__":
    main()
