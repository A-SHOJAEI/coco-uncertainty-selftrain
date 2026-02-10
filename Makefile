PY ?= .venv/bin/python
PIP ?= .venv/bin/pip
SHELL := /bin/bash

VENV := .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

CONFIG ?= configs/smoke.yaml

.PHONY: setup data train eval report all clean

setup:
	@# venv bootstrap: host may lack ensurepip and system pip may be PEP668-managed
	@if [ -d .venv ] && [ ! -x .venv/bin/python ]; then rm -rf .venv; fi
	@if [ ! -d .venv ]; then python3 -m venv --without-pip .venv; fi
	@if [ ! -x .venv/bin/pip ]; then python3 -c "import pathlib,urllib.request; p=pathlib.Path('.venv/get-pip.py'); p.parent.mkdir(parents=True,exist_ok=True); urllib.request.urlretrieve('https://bootstrap.pypa.io/get-pip.py', p)"; .venv/bin/python .venv/get-pip.py; fi
	@bash scripts/setup_venv.sh
	@$(PIP) install -r requirements.txt

data:
	@$(PY) tools/create_smoke_coco.py --out_dir data/smoke_coco --seed 1337
	@$(PY) tools/make_splits.py --coco_root data/smoke_coco --ann_file data/smoke_coco/annotations/instances_train.json --labeled_fraction 0.5 --seed 1337 --out_dir data/splits

train:
	@$(PY) train.py --config $(CONFIG) --output outputs/baseline --mode baseline --labeled_split data/splits/labeled.json
	@$(PY) pseudo_label.py --config $(CONFIG) --weights outputs/baseline/model.pt --unlabeled_split data/splits/unlabeled.json --out_dir data/pseudo/main --uncertainty entropy --mc_dropout 0 --score_thresh 0.05 --filter fixed_thresh
	@$(PY) train.py --config $(CONFIG) --output outputs/selftrain_entropy --mode selftrain --labeled_split data/splits/labeled.json --pseudo_ann data/pseudo/main/pseudo_instances.json --init_weights outputs/baseline/model.pt
	@$(PY) pseudo_label.py --config $(CONFIG) --weights outputs/baseline/model.pt --unlabeled_split data/splits/unlabeled.json --out_dir data/pseudo/ablation_no_weight --uncertainty entropy --mc_dropout 0 --score_thresh 0.05 --filter fixed_thresh --no_uncertainty_weighting
	@$(PY) train.py --config $(CONFIG) --output outputs/ablation_no_weight --mode selftrain --labeled_split data/splits/labeled.json --pseudo_ann data/pseudo/ablation_no_weight/pseudo_instances.json --init_weights outputs/baseline/model.pt

eval:
	@mkdir -p artifacts
	@$(PY) eval.py --config $(CONFIG) --weights outputs/baseline/model.pt --split val --out_dir outputs/baseline
	@$(PY) eval.py --config $(CONFIG) --weights outputs/selftrain_entropy/model.pt --split val --out_dir outputs/selftrain_entropy
	@$(PY) eval.py --config $(CONFIG) --weights outputs/ablation_no_weight/model.pt --split val --out_dir outputs/ablation_no_weight
	@$(PY) tools/summarize_results.py --runs outputs/baseline outputs/selftrain_entropy outputs/ablation_no_weight --out_json artifacts/results.json
	@$(PY) tools/generate_report.py --results_json artifacts/results.json --out_md artifacts/report.md

report:
	@$(PY) tools/generate_report.py --results_json artifacts/results.json --out_md artifacts/report.md

all: setup data train eval report

clean:
	@rm -rf $(VENV) outputs artifacts data/smoke_coco data/splits data/pseudo
