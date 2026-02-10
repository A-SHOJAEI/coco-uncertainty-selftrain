#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coco_uncertainty_selftrain.utils.io import mkdirp


COCO_2017_URLS = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
]

# COCO does not publish official checksums on the download page. When you have a trusted checksum
# (e.g., internal mirror or a verified source), pass --sha256 or extend this mapping.
KNOWN_SHA256: dict[str, str] = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_with_resume(url: str, out_path: Path) -> None:
    mkdirp(out_path.parent)
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    resume_from = tmp_path.stat().st_size if tmp_path.exists() else 0

    headers = {}
    if resume_from > 0:
        headers["Range"] = f"bytes={resume_from}-"

    with requests.get(url, stream=True, headers=headers, timeout=60) as r:
        r.raise_for_status()
        total = r.headers.get("Content-Length")
        total_size = int(total) + resume_from if total is not None else None

        mode = "ab" if resume_from > 0 else "wb"
        with open(tmp_path, mode) as f, tqdm(
            total=total_size, initial=resume_from, unit="B", unit_scale=True, desc=out_path.name
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    os.replace(tmp_path, out_path)


def verify_sha256(path: Path, expected: Optional[str]) -> None:
    got = sha256_file(path)
    if expected is None:
        # Still compute and persist the checksum for reproducibility, but do not claim verification.
        (path.with_suffix(path.suffix + ".sha256")).write_text(got + "\n", encoding="utf-8")
        print(
            f"WARNING: No trusted SHA256 provided for {path.name}. "
            f"Computed SHA256={got} (written to {path.name}.sha256)."
        )
        return
    if got.lower() != expected.lower():
        raise SystemExit(f"SHA256 mismatch for {path.name}: expected {expected}, got {got}")


def unzip(path: Path, out_dir: Path) -> None:
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=Path, required=True, help="Directory to unzip into (e.g. data/coco)")
    ap.add_argument("--verify_sha256", action="store_true")
    ap.add_argument("--sha256", action="append", default=[], help="Override checksum: URL=HEX (can be repeated)")
    ap.add_argument("--skip_unzip", action="store_true")
    args = ap.parse_args()

    mkdirp(args.out_dir)

    overrides: dict[str, str] = {}
    for item in args.sha256:
        if "=" not in item:
            raise SystemExit(f"--sha256 must be URL=HEX, got: {item}")
        url, hex_ = item.split("=", 1)
        overrides[url.strip()] = hex_.strip()

    for url in COCO_2017_URLS:
        name = url.split("/")[-1]
        zip_path = args.out_dir / name
        print(f"Downloading {url} -> {zip_path}")
        download_with_resume(url, zip_path)

        if args.verify_sha256:
            expected = overrides.get(url) or KNOWN_SHA256.get(url)
            verify_sha256(zip_path, expected)

        if not args.skip_unzip:
            print(f"Unzipping {zip_path} -> {args.out_dir}")
            unzip(zip_path, args.out_dir)


if __name__ == "__main__":
    main()
