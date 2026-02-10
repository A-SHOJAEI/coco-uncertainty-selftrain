#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from coco_uncertainty_selftrain.utils.io import read_json, write_json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", type=Path, required=True, help="Run dirs containing eval.json")
    ap.add_argument("--out_json", type=Path, required=True)
    args = ap.parse_args()

    out = {"runs": []}
    for run_dir in args.runs:
        eval_path = run_dir / "eval.json"
        if not eval_path.exists():
            raise SystemExit(f"Missing {eval_path}")
        data = read_json(eval_path)
        out["runs"].append({"run_dir": str(run_dir), **data})

    write_json(args.out_json, out)


if __name__ == "__main__":
    main()
