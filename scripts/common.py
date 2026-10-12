"""Small helpers shared by the scripts (paths, protocol files, audio I/O)."""

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf

import config

PROTOCOL_NAMES = {
    "train": "ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019.LA.cm.eval.trl.txt",
}


@lru_cache(maxsize=1)
def la_root() -> Path:
    """
    Folder that holds ASVspoof2019_LA_train/dev/eval and the protocols.
    Found by searching across ASVSPOOF_DIR, LOCAL_DATA, and any Kaggle input folders.
    """
    name = "ASVspoof2019_LA_cm_protocols"
    search_dirs = [config.LOCAL_DATA, config.ASVSPOOF_DIR, config.DATA_BASE, config.BASE]
    if Path("/kaggle/input").exists():
        search_dirs.extend([d for d in Path("/kaggle/input").glob("*") if d.is_dir()])
        search_dirs.append(Path("/kaggle/input"))
    for sdir in search_dirs:
        if not sdir.exists():
            continue
        if (sdir / name).is_dir():
            return sdir
        try:
            for hit in sdir.rglob(name):
                if hit.is_dir():
                    return hit.parent
        except Exception:
            pass
    sys.exit(f"ERROR: no ASVspoof2019_LA_cm_protocols folder under any candidate path. "
             f"Checked: {[str(d) for d in search_dirs if d.exists()]}. Finish T2 / unpack asvspoof first.")


def split_dir(split: str) -> Path:
    """Folder that contains flac/ for a split: train, dev or eval."""
    root = la_root()
    direct = root / f"ASVspoof2019_LA_{split}"
    if direct.exists():
        return direct
    try:
        for match in root.rglob(f"ASVspoof2019_LA_{split}"):
            if match.is_dir():
                return match
    except Exception:
        pass
    return direct


def protocol_path(split: str) -> Path:
    return la_root() / "ASVspoof2019_LA_cm_protocols" / PROTOCOL_NAMES[split]


def read_protocol(path: Path) -> list:
    """Each line: speaker utt_id - attack label  -> list of 5-item lists."""
    with open(path) as f:
        return [line.split() for line in f if line.strip()]


def write_protocol(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(" ".join(row) + "\n")


def eval_rows_for(condition: str) -> list:
    """Protocol rows for a test condition (full eval set or the 10k subset)."""
    if condition == "clean":
        return read_protocol(protocol_path("eval"))
    if not config.EVAL_SUBSET_FILE.exists():
        sys.exit(f"ERROR: {config.EVAL_SUBSET_FILE} missing. "
                 "Run scripts/make_eval_subset.py first.")
    return read_protocol(config.EVAL_SUBSET_FILE)


def condition_dir(condition: str) -> Path:
    """Folder containing flac/ for a test condition."""
    if condition in ("clean", "clean_subset"):
        return split_dir("eval")
    cand = config.EVAL_CONDITIONS_DIR / condition
    if (cand / "flac").exists() or (cand.is_dir() and list(cand.glob("*.flac"))):
        return cand
    if cand.exists():
        return cand
    if Path("/kaggle/input").exists():
        try:
            for d in Path("/kaggle/input").rglob(condition):
                if d.is_dir() and ((d / "flac").exists() or list(d.glob("*.flac")) or (d.parent.name == "eval")):
                    return d
        except Exception:
            pass
    return cand


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_audio(path: Path) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != config.SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=config.SAMPLE_RATE)
    return audio


def save_flac(path: Path, audio: np.ndarray) -> None:
    """Write 16 kHz FLAC (what AASIST reads). Scales down instead of clipping."""
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak * 0.99
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.flac")
    sf.write(str(tmp), audio, config.SAMPLE_RATE, format="FLAC")
    tmp.replace(path)   # no half-written files if Colab disconnects


def add_aasist_to_path() -> None:
    if not (config.AASIST_DIR / "evaluation.py").exists():
        sys.exit(f"ERROR: AASIST code not found at {config.AASIST_DIR}. Do T5.")
    sys.path.insert(0, str(config.AASIST_DIR))
