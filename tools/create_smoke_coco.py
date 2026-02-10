#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coco_uncertainty_selftrain.utils.io import mkdirp, write_json


def _rect_poly(x0: int, y0: int, x1: int, y1: int) -> list[float]:
    # COCO polygon format: flat list of x,y coordinates.
    return [float(x0), float(y0), float(x1), float(y0), float(x1), float(y1), float(x0), float(y1)]


def generate_split(root: Path, split: str, *, seed: int, n_images: int, start_image_id: int, start_ann_id: int):
    rng = random.Random(seed + (0 if split == "train" else 10_000))
    images_dir = mkdirp(root / "images" / split)
    ann_dir = mkdirp(root / "annotations")

    categories = [
        {"id": 1, "name": "square", "supercategory": "shape"},
        {"id": 2, "name": "rectangle", "supercategory": "shape"},
    ]

    images = []
    annotations = []

    image_id = start_image_id
    ann_id = start_ann_id

    for i in range(n_images):
        w, h = 128, 128
        bg = np.uint8(rng.randint(0, 30))
        arr = np.full((h, w, 3), bg, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        draw = ImageDraw.Draw(img)

        file_name = f"images/{split}/img_{image_id:012d}.jpg"
        abs_path = root / file_name

        # 1-2 instances with simple rectangles.
        n_inst = 2 if (i % 2 == 0) else 1
        for j in range(n_inst):
            cat_id = 1 if (j % 2 == 0) else 2
            if cat_id == 1:
                side = rng.randint(20, 45)
                x0 = rng.randint(5, w - side - 5)
                y0 = rng.randint(5, h - side - 5)
                x1 = x0 + side
                y1 = y0 + side
                color = (200, 50, 50)
            else:
                rw = rng.randint(25, 55)
                rh = rng.randint(15, 40)
                x0 = rng.randint(5, w - rw - 5)
                y0 = rng.randint(5, h - rh - 5)
                x1 = x0 + rw
                y1 = y0 + rh
                color = (50, 200, 50)

            draw.rectangle([x0, y0, x1, y1], outline=color, fill=color)

            bbox = [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]
            area = float((x1 - x0) * (y1 - y0))
            poly = _rect_poly(x0, y0, x1, y1)
            annotations.append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": cat_id,
                    "bbox": bbox,
                    "area": area,
                    "segmentation": [poly],
                    "iscrowd": 0,
                }
            )
            ann_id += 1

        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": w,
                "height": h,
            }
        )

        mkdirp(abs_path.parent)
        img.save(abs_path, quality=90)
        image_id += 1

    ann = {
        "info": {
            "description": "Synthetic COCO-format smoke dataset",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    out_json = ann_dir / f"instances_{split}.json"
    write_json(out_json, ann)
    return out_json, image_id, ann_id


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--train_images", type=int, default=8)
    ap.add_argument("--val_images", type=int, default=4)
    args = ap.parse_args()

    mkdirp(args.out_dir)
    image_id, ann_id = 1, 1
    _train_json, image_id, ann_id = generate_split(
        args.out_dir, "train", seed=args.seed, n_images=args.train_images, start_image_id=image_id, start_ann_id=ann_id
    )
    generate_split(args.out_dir, "val", seed=args.seed, n_images=args.val_images, start_image_id=image_id, start_ann_id=ann_id)


if __name__ == "__main__":
    main()
