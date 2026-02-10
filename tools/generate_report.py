#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coco_uncertainty_selftrain.utils.io import read_json, mkdirp


def _fmt(x):
    if x is None:
        return "n/a"
    if isinstance(x, (int, float)):
        return f"{x:.4f}"
    return str(x)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_json", type=Path, required=True)
    ap.add_argument("--out_md", type=Path, required=True)
    args = ap.parse_args()

    res = read_json(args.results_json)
    runs = res.get("runs", [])

    lines = []
    lines.append("# COCO Uncertainty Self-Training Report\n")
    lines.append("This report is generated from `artifacts/results.json`.\n")

    lines.append("## Summary\n")
    lines.append("| Run | segm AP | segm AP50 | segm AP75 | bbox AP | Notes |")
    lines.append("|---|---:|---:|---:|---:|---|")

    for r in runs:
        m = r.get("metrics", {})
        segm = m.get("segm", {})
        bbox = m.get("bbox", {})
        name = Path(r.get("run_dir", "")).name
        notes = r.get("notes", "")
        lines.append(
            "| "
            + " | ".join(
                [
                    name,
                    _fmt(segm.get("AP")),
                    _fmt(segm.get("AP50")),
                    _fmt(segm.get("AP75")),
                    _fmt(bbox.get("AP")),
                    notes,
                ]
            )
            + " |"
        )

    lines.append("\n## Details\n")
    for r in runs:
        name = Path(r.get("run_dir", "")).name
        lines.append(f"### {name}\n")
        lines.append("```json")
        # Keep it small; eval.json already has the full content.
        lines.append(
            json.dumps(
                {"metrics": r.get("metrics", {}), "meta": r.get("meta", {})},
                indent=2,
                sort_keys=True,
            )
        )
        lines.append("```\n")

    mkdirp(args.out_md.parent)
    args.out_md.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
