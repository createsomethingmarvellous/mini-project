"""
Thin bridge to the official AASIST code (github.com/clovaai/aasist).

We reuse AASIST's own model, data loading, optimizer and EER code, and only
add what the project needs: files from more than one folder, resuming after
a Colab disconnect, and scoring any folder of test files.
"""

import json
from importlib import import_module
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

import config
from common import add_aasist_to_path

add_aasist_to_path()
from data_utils import Dataset_ASVspoof2019_devNeval, pad_random  # noqa: E402
from evaluation import compute_eer  # noqa: E402  (official EER code)

import soundfile as sf  # noqa: E402
from tqdm import tqdm  # noqa: E402


def load_aasist_config() -> dict:
    with open(config.AASIST_DIR / "config" / "AASIST.conf") as f:
        conf = json.load(f)
    conf["batch_size"] = config.BATCH_SIZE
    conf["num_epochs"] = config.NUM_EPOCHS
    return conf


def build_model(model_config: dict, device: str):
    module = import_module(f"models.{model_config['architecture']}")
    return module.Model(model_config).to(device)


class TrainSet(torch.utils.data.Dataset):
    """Same as AASIST's Dataset_ASVspoof2019_train, but each key has its own path."""

    def __init__(self, paths: dict, labels: dict):
        self.keys = list(paths)
        self.paths = paths
        self.labels = labels
        self.cut = 64600    # ~4 s, as in AASIST

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        key = self.keys[index]
        x, _ = sf.read(str(self.paths[key]))
        return Tensor(pad_random(x, self.cut)), self.labels[key]


def eval_loader(rows: list, base_dir: Path, batch_size: int = config.EVAL_BATCH_SIZE):
    """Scoring loader. Batch size doesn't change scores (eval mode), only speed."""
    keys = [r[1] for r in rows]
    dataset = Dataset_ASVspoof2019_devNeval(list_IDs=keys, base_dir=base_dir)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=False, num_workers=config.NUM_WORKERS,
                                       pin_memory=True, persistent_workers=False)


def write_scores(model, loader, rows: list, out_path: Path, device: str,
                 progress: bool = False) -> float:
    """Score every file; write AASIST-format lines (utt src key score); return EER %."""
    model.eval()
    scores = []
    batches = tqdm(loader, desc="scoring", unit="batch", mininterval=10) if progress else loader
    device_type = "cuda" if "cuda" in str(device) else "cpu"
    with torch.no_grad():
        for batch_x, _ in batches:
            try:
                try:
                    autocast_ctx = torch.amp.autocast(device_type=device_type)
                except (TypeError, AttributeError):
                    autocast_ctx = torch.cuda.amp.autocast()
                with autocast_ctx:
                    _, out = model(batch_x.to(device))
            except torch.cuda.OutOfMemoryError:
                raise SystemExit("CUDA out of memory while scoring: lower "
                                 "EVAL_BATCH_SIZE in scripts/config.py and rerun.")
            scores.extend(out[:, 1].cpu().float().numpy().ravel().tolist())
    assert len(scores) == len(rows)

    scores_arr = np.array(scores, dtype=np.float64)
    if np.isnan(scores_arr).any():
        nan_count = int(np.isnan(scores_arr).sum())
        valid_min = float(np.nanmin(scores_arr)) if not np.isnan(scores_arr).all() else 0.0
        scores_arr = np.nan_to_num(scores_arr, nan=valid_min)
        tqdm.write(f"⚠️ Notice: {nan_count}/{len(scores)} scores were NaN; safely regularized.")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row, score in zip(rows, scores_arr):
            f.write(f"{row[1]} {row[3]} {row[4]} {score}\n")
    return eer_from_rows(rows, scores_arr)


def eer_from_rows(rows: list, scores: np.ndarray) -> float:
    labels = np.array([r[4] for r in rows])
    bona = scores[labels == "bonafide"]
    spoof = scores[labels == "spoof"]
    if len(bona) == 0 or len(spoof) == 0:
        print(f"⚠️ Warning: protocol label mismatch. Bonafide: {len(bona)}, Spoof: {len(spoof)}")
        return 100.0
    return float(100 * compute_eer(bona, spoof)[0])
