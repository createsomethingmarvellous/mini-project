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


def eval_loader(rows: list, base_dir: Path, batch_size: int):
    keys = [r[1] for r in rows]
    dataset = Dataset_ASVspoof2019_devNeval(list_IDs=keys, base_dir=base_dir)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=False, num_workers=2,
                                       pin_memory=True)


def write_scores(model, loader, rows: list, out_path: Path, device: str) -> float:
    """Score every file; write AASIST-format lines (utt src key score); return EER %."""
    model.eval()
    scores = []
    with torch.no_grad():
        for batch_x, _ in loader:
            _, out = model(batch_x.to(device))
            scores.extend(out[:, 1].cpu().numpy().ravel().tolist())
    assert len(scores) == len(rows)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for row, score in zip(rows, scores):
            f.write(f"{row[1]} {row[3]} {row[4]} {score}\n")
    return eer_from_rows(rows, np.array(scores))


def eer_from_rows(rows: list, scores: np.ndarray) -> float:
    labels = np.array([r[4] for r in rows])
    # plain float: numpy scalars can't be reloaded from checkpoints by newer torch
    return float(100 * compute_eer(scores[labels == "bonafide"],
                                   scores[labels == "spoof"])[0])
