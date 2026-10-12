import os
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


def find_folder_fast(target_name: str, search_roots: list) -> Path | None:
    """
    Lightning-fast directory finder.
    Prunes audio folders ('flac', 'wav') so it never traverses 100,000+ clips.
    Recurses instantaneously to arbitrary depth in < 0.05 seconds.
    """
    for base in search_roots:
        p = Path(base)
        if not p.exists():
            continue
        if (p / target_name).is_dir():
            return p / target_name
        for root, dirs, _ in os.walk(p):
            for skip in ("flac", "wav", ".git", "__pycache__"):
                if skip in dirs:
                    dirs.remove(skip)
            if target_name in dirs:
                return Path(root) / target_name
    return None


@lru_cache(maxsize=1)
def la_root() -> Path:
    """
    Folder that holds ASVspoof2019_LA_train/dev/eval and the protocols.
    Auto-detects paths at any nesting depth in Kaggle/Drive/local in milliseconds.
    """
    roots = [config.LOCAL_DATA, config.ASVSPOOF_DIR, config.DATA_BASE, config.BASE]
    if Path("/kaggle/input").exists():
        roots.append(Path("/kaggle/input"))

    # 1. Primary: ASVspoof2019_LA_cm_protocols
    hit = find_folder_fast("ASVspoof2019_LA_cm_protocols", roots)
    if hit is not None:
        return hit.parent

    # 2. Fallback: ASVspoof2019_LA_asv_protocols (same parent directory!)
    hit = find_folder_fast("ASVspoof2019_LA_asv_protocols", roots)
    if hit is not None:
        return hit.parent

    # 3. Fallback: ASVspoof2019_LA_train (same parent directory!)
    hit = find_folder_fast("ASVspoof2019_LA_train", roots)
    if hit is not None:
        return hit.parent

    sys.exit(
        f"ERROR: no ASVspoof protocols folder found. "
        f"Checked: {[str(d) for d in roots if Path(d).exists()]}. "
        f"Run Step 2 to unpack asvspoof first."
    )


def split_dir(split: str) -> Path:
    """Folder that contains flac/ for a split: train, dev or eval."""
    root = la_root()
    direct = root / f"ASVspoof2019_LA_{split}"
    if direct.exists():
        return direct
    # Fast: check one level deeper (handles nested layout)
    for child in root.iterdir():
        if child.is_dir() and child.name == f"ASVspoof2019_LA_{split}":
            return child
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
    if cand.exists():
        return cand
    if Path("/kaggle/input").exists():
        hit = find_folder_fast(condition, [Path("/kaggle/input")])
        if hit is not None:
            return hit
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
