from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class ReproConfig:
    seed: int
    deterministic: bool = True


def seed_everything(cfg: ReproConfig) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(cfg.seed))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    if cfg.deterministic:
        # cuDNN
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # cuBLAS (if CUDA is used)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int) -> None:
    # Dataloader workers: make RNG deterministic.
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

