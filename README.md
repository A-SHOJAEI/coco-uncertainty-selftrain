# COCO Uncertainty-Weighted Self-Training (Mask R-CNN)

This repository implements a teacher-student self-training pipeline for **label-efficient instance segmentation**. A teacher Mask R-CNN is trained on a labeled subset, used to generate pseudo-labels on unlabeled images, and a student is trained on labeled + pseudo-labeled data where **pseudo instances are weighted by an uncertainty signal**.

The implementation uses `torchvision`’s `maskrcnn_resnet50_fpn` (no Detectron2) and evaluates with COCO-native `pycocotools` / `COCOeval`.

## Problem Statement

Given COCO-format instance segmentation data with only a fraction labeled, improve student performance by:
1. training a supervised teacher on the labeled subset,
2. generating pseudo instance masks on the unlabeled subset, and
3. training a student where pseudo instances are **filtered** and **weighted** by uncertainty.

Core scripts:
- `train.py`: baseline or self-train (interleaves labeled/pseudo batches; pseudo loss scaled by per-image mean instance weight)
- `pseudo_label.py`: pseudo-label generation + uncertainty weights
- `eval.py`: COCOeval metrics (bbox + segm)
- `tools/summarize_results.py` + `tools/generate_report.py`: `artifacts/results.json` and `artifacts/report.md`

## Dataset Provenance

Two dataset modes exist in-tree:

1. **Smoke dataset (default; used for the committed results)**  
   Generated locally by `tools/create_smoke_coco.py` into `data/smoke_coco/` as a tiny COCO-format dataset:
   - Train: 8 images, 12 instances
   - Val: 4 images, 6 instances
   - Categories: `square` (id=1), `rectangle` (id=2)  
   Labeled/unlabeled split is created from the *train* images by `tools/make_splits.py` with `labeled_fraction=0.5`, `seed=1337`:
   - Labeled image ids: `[1, 3, 4, 8]` (`data/splits/labeled.json`)
   - Unlabeled image ids: `[2, 5, 6, 7]` (`data/splits/unlabeled.json`)

2. **COCO 2017 (optional; not downloaded or run by default)**  
   `tools/download_coco2017.py` downloads the official 2017 zips from `images.cocodataset.org` into `data/coco/` (train2017, val2017, annotations). COCO does not publish official checksums; the tool can still compute and record SHA256 for reproducibility.

## Methodology (What’s Implemented)

**Teacher model** (`train.py --mode baseline`):
- `torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)` (`src/coco_uncertainty_selftrain/models/maskrcnn.py`)
- Supervised training on labeled split only (weak aug = tensor + random horizontal flip; `src/coco_uncertainty_selftrain/data/transforms.py`)

**Pseudo-labeling** (`pseudo_label.py`):
- Runs teacher on unlabeled split with `augment="none"` to avoid misalignment.
- Exports COCO-format `data/pseudo/*/pseudo_instances.json` with per-annotation fields:
  - `score` (detector score)
  - `weight` (computed from uncertainty; consumed by dataset via `ann.get("weight", 1.0)`)
- Filtering modes:
  - `fixed_thresh`: keep detections with `score >= score_thresh`
  - `adaptive_class`: per-class score threshold set to keep a configured fraction (`--adaptive_keep`)
  - `topk_per_image`: keep top-K detections per image
- Uncertainty signals (see `src/coco_uncertainty_selftrain/pseudo/uncertainty.py`):
  - `entropy`, `margin`: **approximate** from top-1 score (torchvision does not expose full class probabilities)
  - `mask_mean_conf`: mean mask probability over predicted foreground pixels
  - `mc_dropout_var`: variance of matched detection scores across stochastic passes (dropout injected into ROI heads; `mc_dropout_p` in config)

**Student self-training** (`train.py --mode selftrain`):
- Interleaves labeled batches (weak aug) and pseudo batches (strong aug defaults to photometric jitter only; no geometry).
- Pseudo sample weight = `mean(instance_weights)` for that image, multiplied by `selftrain.pseudo_loss_weight`.

## Baselines and Ablations (In This Repo)

The `make all` smoke pipeline runs three variants (see `Makefile`):
- `baseline`: labeled-only supervised training (`outputs/baseline/`)
- `selftrain_entropy`: pseudo-labels + **entropy-derived weighting** (`outputs/selftrain_entropy/`)
- `ablation_no_weight`: pseudo-labels with `--no_uncertainty_weighting` (all pseudo instances weight=1) (`outputs/ablation_no_weight/`)

Additional ablation suite definitions are in `configs/ablations.yaml` and can be executed on COCO via `run_ablations.py`.

## Results (Exact Numbers From This Repo)

The authoritative sources are:
- Summary table: `artifacts/report.md` (section “Summary”)
- Raw run payloads: `artifacts/results.json` and per-run `outputs/*/eval.json`

Committed run context (from `artifacts/results.json` / `outputs/*/eval.json`):
- Config: `configs/smoke.yaml` (device `cpu`, `train.max_steps=2`, `eval.max_images=8`)
- Evaluation split: `val` (synthetic smoke val)
- Recorded at: `2026-02-10` on git `4339b34fe7df07c49402e52d72c88f41f34a8796` (Python `3.12.3`, torch `2.5.1+cu124`, torchvision `0.20.1+cu124`)

**Table: Smoke COCOeval metrics (max_images=8)** (mirrors `artifacts/report.md`):

| Run | segm AP | segm AP50 | segm AP75 | bbox AP |
|---|---:|---:|---:|---:|
| baseline | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| selftrain_entropy | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ablation_no_weight | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

Pseudo-labeling behavior on the smoke run (from `data/pseudo/main/pseudo_stats.json`):
- `score_thresh=0.05`, `filter=fixed_thresh` accepted 400/400 detections (effectively “accept all top detections”).

No figures are generated by this repository; the only rendered artifact is the markdown report `artifacts/report.md`.

## Repro Instructions

### End-to-end smoke run (matches `artifacts/results.json`)

```bash
make all
```

Key outputs:
- `data/smoke_coco/`: synthetic dataset
- `data/splits/labeled.json`, `data/splits/unlabeled.json`: split definition
- `data/pseudo/*/pseudo_instances.json`: pseudo annotations (+ `pseudo_stats.json`)
- `outputs/*/model.pt`: checkpoints
- `outputs/*/eval.json`: COCOeval metrics
- `artifacts/results.json`: aggregated metrics
- `artifacts/report.md`: markdown report

### Full COCO 2017 (download + 10% labeled baseline + self-training)

```bash
make setup
./.venv/bin/python tools/download_coco2017.py --out_dir data/coco
./.venv/bin/python tools/make_splits.py \
  --coco_root data/coco \
  --ann_file data/coco/annotations/instances_train2017.json \
  --labeled_fraction 0.1 --seed 1337 \
  --out_dir data/splits

./.venv/bin/python train.py \
  --config configs/coco_baseline_10pct.yaml \
  --mode baseline \
  --labeled_split data/splits/labeled.json \
  --output outputs/baseline_10pct

./.venv/bin/python pseudo_label.py \
  --config configs/coco_selftrain_uncertainty.yaml \
  --weights outputs/baseline_10pct/model.pt \
  --unlabeled_split data/splits/unlabeled.json \
  --out_dir data/pseudo/entropy \
  --uncertainty entropy --filter fixed_thresh --score_thresh 0.5

./.venv/bin/python train.py \
  --config configs/coco_selftrain_uncertainty.yaml \
  --mode selftrain \
  --labeled_split data/splits/labeled.json \
  --pseudo_ann data/pseudo/entropy/pseudo_instances.json \
  --init_weights outputs/baseline_10pct/model.pt \
  --output outputs/selftrain_entropy

./.venv/bin/python eval.py \
  --config configs/coco_selftrain_uncertainty.yaml \
  --weights outputs/selftrain_entropy/model.pt \
  --split val \
  --out_dir outputs/selftrain_entropy
```

## Limitations (Current State)

- The only recorded results in `artifacts/results.json` are from the **smoke** configuration (`max_steps=2`) and yield **0 AP** across runs; they are a pipeline sanity check, not a meaningful benchmark.
- Models are initialized with `weights=None` / `weights_backbone=None` (no ImageNet pretraining), which is atypical for COCO and makes convergence much harder.
- “Entropy” and “margin” uncertainty are computed from a **top-1 score approximation** because torchvision inference does not expose the full class distribution.
- The code calls the augmentation toggle “strong-weak consistency”, but there is **no explicit consistency loss** implemented; it is only differing augment pipelines for labeled vs pseudo batches.
- The smoke pseudo-label run uses a very low `score_thresh` (0.05) and accepts 100 detections per image, creating highly noisy pseudo labels.

## Next Research Steps (Concrete)

1. Run COCO with pretrained backbones (or load torchvision weights) and report real COCO numbers for baseline vs uncertainty-weighted self-training.
2. Replace top-1 entropy/margin approximations by extracting logits/probabilities (or use a framework that exposes them) and evaluate calibration-sensitive weighting.
3. Add an EMA teacher + explicit consistency losses (e.g., box/mask consistency under strong aug) rather than only augmentation differences.
4. Improve pseudo-label quality: per-class thresholds, NMS/duplicate suppression controls, class-balanced acceptance, and per-instance (not per-image mean) loss weighting.
5. Evaluate MC-dropout variance signal end-to-end (configs already support dropout injection and multiple passes).
