#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from coco_uncertainty_selftrain.utils.io import read_yaml


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.check_call(cmd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=Path, required=True, help="YAML listing ablation variants")
    ap.add_argument("--config", type=Path, required=True, help="Base config")
    ap.add_argument("--weights", type=Path, required=True, help="Teacher weights for pseudo-labeling")
    ap.add_argument("--labeled_split", type=Path, required=True)
    ap.add_argument("--unlabeled_split", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    suite = read_yaml(args.suite)
    variants = suite.get("variants", [])
    if not isinstance(variants, list) or not variants:
        raise SystemExit(f"No variants found in {args.suite}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for v in variants:
        name = v["name"]
        unc = v.get("uncertainty", "entropy")
        filt = v.get("filter", "fixed_thresh")
        score_thresh = float(v.get("score_thresh", 0.5))
        mc = int(v.get("mc_dropout", 0))
        no_weight = bool(v.get("no_uncertainty_weighting", False))
        no_cons = bool(v.get("no_consistency", False))
        topk = v.get("topk")
        adaptive_keep = v.get("adaptive_keep")

        pseudo_dir = args.output_dir / name / "pseudo"
        run_dir = args.output_dir / name / "run"

        pl_cmd = [
            sys.executable,
            "pseudo_label.py",
            "--config",
            str(args.config),
            "--weights",
            str(args.weights),
            "--unlabeled_split",
            str(args.unlabeled_split),
            "--out_dir",
            str(pseudo_dir),
            "--uncertainty",
            str(unc),
            "--mc_dropout",
            str(mc),
            "--score_thresh",
            str(score_thresh),
            "--filter",
            str(filt),
        ]
        if topk is not None:
            pl_cmd += ["--topk", str(int(topk))]
        if adaptive_keep is not None:
            pl_cmd += ["--adaptive_keep", str(float(adaptive_keep))]
        if no_weight:
            pl_cmd.append("--no_uncertainty_weighting")
        _run(pl_cmd)

        tr_cmd = [
            sys.executable,
            "train.py",
            "--config",
            str(args.config),
            "--output",
            str(run_dir),
            "--mode",
            "selftrain",
            "--labeled_split",
            str(args.labeled_split),
            "--pseudo_ann",
            str(pseudo_dir / "pseudo_instances.json"),
            "--init_weights",
            str(args.weights),
        ]
        if no_cons:
            tr_cmd.append("--no_consistency")
        _run(tr_cmd)

        ev_cmd = [
            sys.executable,
            "eval.py",
            "--config",
            str(args.config),
            "--weights",
            str(run_dir / "model.pt"),
            "--split",
            "val",
            "--out_dir",
            str(run_dir),
        ]
        _run(ev_cmd)


if __name__ == "__main__":
    main()
