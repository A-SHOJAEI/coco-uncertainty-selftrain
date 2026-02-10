#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path

import json
import torch
from torch.utils.data import DataLoader

# Local package without requiring editable install.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from coco_uncertainty_selftrain.data.coco_dataset import CocoInstanceSegmentation, collate_fn
from coco_uncertainty_selftrain.data.transforms import build_transforms
from coco_uncertainty_selftrain.models.maskrcnn import build_maskrcnn
from coco_uncertainty_selftrain.utils.io import mkdirp, read_json, read_yaml, write_json, write_yaml
from coco_uncertainty_selftrain.utils.meta import collect_meta
from coco_uncertainty_selftrain.utils.repro import ReproConfig, seed_everything, seed_worker


def _device_from_config(cfg: dict) -> torch.device:
    d = cfg.get("device", "cpu")
    if d == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(d)


def infinite_loader(dataloader: DataLoader):
    while True:
        for batch in dataloader:
            yield batch


def _grad_norm(model: torch.nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.sum(g * g).item())
    return float(total ** 0.5)


def _build_loader(cfg: dict, *, ann_file: Path, image_ids: list[int], augment: str, batch_size: int) -> DataLoader:
    ds = CocoInstanceSegmentation(
        coco_root=Path(cfg["data"]["coco_root"]),
        ann_file=ann_file,
        image_ids=image_ids,
        transforms=build_transforms(augment=augment),
    )
    g = torch.Generator()
    g.manual_seed(int(cfg.get("seed", 1337)))
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(cfg["data"].get("num_workers", 0)),
        collate_fn=collate_fn,
        worker_init_fn=seed_worker,
        generator=g,
    ), ds


def _sample_weight_from_target(target: dict) -> float:
    w = target.get("instance_weights")
    if w is None or w.numel() == 0:
        return 1.0
    return float(w.float().mean().item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--mode", choices=["baseline", "selftrain"], required=True)
    ap.add_argument("--labeled_split", type=Path, required=True)
    ap.add_argument("--pseudo_ann", type=Path, default=None, help="COCO-format pseudo instances JSON (for selftrain)")
    ap.add_argument("--init_weights", type=Path, default=None, help="Initialize model from a checkpoint (teacher->student)")
    ap.add_argument("--no_consistency", action="store_true", help="Ablation: pseudo-only, single augmentation")
    args = ap.parse_args()

    cfg = read_yaml(args.config)
    seed_everything(ReproConfig(seed=int(cfg.get("seed", 1337)), deterministic=True))

    run_dir = mkdirp(args.output)
    write_yaml(run_dir / "config.yaml", cfg)

    device = _device_from_config(cfg)
    meta = collect_meta()
    meta["device"] = str(device)
    meta["mode"] = args.mode
    meta["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(run_dir / "meta.json", meta)

    labeled_split = read_json(args.labeled_split)
    labeled_ids = [int(x) for x in labeled_split["image_ids"]]

    # train_ann/val_ann are relative to coco_root in configs, unless absolute.
    train_ann = Path(cfg["data"]["coco_root"]) / cfg["data"]["train_ann"]
    # If user supplied absolute ann paths, allow it.
    if Path(cfg["data"]["train_ann"]).is_absolute():
        train_ann = Path(cfg["data"]["train_ann"])

    batch_size = int(cfg["train"].get("batch_size", 1))
    max_steps = int(cfg["train"].get("max_steps", 100))

    # Baseline uses weak aug; self-train stage uses strong aug for student.
    if args.mode == "baseline":
        dl, _ds = _build_loader(cfg, ann_file=train_ann, image_ids=labeled_ids, augment="weak", batch_size=batch_size)
        loader = infinite_loader(dl)
        pseudo_scale = 0.0
        pseudo_only = False
    else:
        if args.pseudo_ann is None:
            raise SystemExit("--pseudo_ann is required for --mode selftrain")
        pseudo_only = bool(args.no_consistency)
        labeled_aug = "weak"
        pseudo_aug = "weak" if pseudo_only else ("strong" if cfg.get("selftrain", {}).get("use_strong_weak_consistency", True) else "weak")
        labeled_dl, _ = _build_loader(cfg, ann_file=train_ann, image_ids=labeled_ids, augment=labeled_aug, batch_size=1)

        pseudo_data = read_json(args.pseudo_ann)
        pseudo_ids = sorted({int(img["id"]) for img in pseudo_data.get("images", [])})
        if not pseudo_ids:
            raise SystemExit(
                f"Pseudo annotation file has 0 images: {args.pseudo_ann}. "
                "Lower the pseudo-label score threshold or change filtering."
            )
        pseudo_dl, _ = _build_loader(cfg, ann_file=args.pseudo_ann, image_ids=pseudo_ids, augment=pseudo_aug, batch_size=1)

        pseudo_scale = float(cfg.get("selftrain", {}).get("pseudo_loss_weight", 1.0))

        # Interleave labeled and pseudo batches deterministically: cycle both.
        def _mixed():
            labeled_loader = infinite_loader(labeled_dl)
            pseudo_loader = infinite_loader(pseudo_dl)
            if pseudo_only:
                while True:
                    yield ("pseudo", next(pseudo_loader))
            else:
                while True:
                    yield ("labeled", next(labeled_loader))
                    yield ("pseudo", next(pseudo_loader))

        loader = _mixed()

    num_classes = int(cfg["model"].get("num_classes", 3))
    model = build_maskrcnn(num_classes=num_classes, mc_dropout_p=float(cfg["model"].get("mc_dropout_p", 0.0)))
    if args.init_weights is not None:
        try:
            state = torch.load(args.init_weights, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(args.init_weights, map_location="cpu")
        model.load_state_dict(state)
    model.to(device)
    model.train()

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(
        params,
        lr=float(cfg["train"].get("lr", 0.005)),
        momentum=0.9,
        weight_decay=float(cfg["train"].get("weight_decay", 1e-4)),
    )

    amp_enabled = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    grad_clip = float(cfg["train"].get("grad_clip_norm", 0.0))

    log_path = run_dir / "train_log.jsonl"
    if log_path.exists():
        log_path.unlink()

    # Training loop
    step = 0
    t0 = time.time()
    while step < max_steps:
        if args.mode == "baseline":
            images, targets = next(loader)
            batch_kind = "labeled"
        else:
            batch_kind, (images, targets) = next(loader)

        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        # Per-sample weighting for pseudo-labels. We keep batch_size=1 in self-train to make this simple and robust.
        sample_weight = 1.0
        if batch_kind == "pseudo":
            sample_weight = _sample_weight_from_target(targets[0]) * pseudo_scale

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=amp_enabled):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values()) * float(sample_weight)

        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        gn = _grad_norm(model)
        scaler.step(opt)
        scaler.update()

        if step % 1 == 0:
            elapsed = time.time() - t0
            ld = {k: float(v.detach().cpu().item()) for k, v in loss_dict.items()}
            scale = float(scaler.get_scale()) if scaler.is_enabled() else None
            print(
                f"[step {step:04d}] kind={batch_kind} w={sample_weight:.3f} "
                f"loss={float(loss.detach().cpu()):.4f} grad_norm={gn:.3f} amp_scale={scale} {ld} ({elapsed:.1f}s)"
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "step": int(step),
                            "kind": batch_kind,
                            "sample_weight": float(sample_weight),
                            "loss": float(loss.detach().cpu().item()),
                            "losses": ld,
                            "grad_norm": float(gn),
                            "amp_scale": scale,
                            "elapsed_s": float(elapsed),
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        step += 1

    out_path = run_dir / "model.pt"
    torch.save(model.state_dict(), out_path)
    print(f"Saved model to {out_path}")


if __name__ == "__main__":
    main()
