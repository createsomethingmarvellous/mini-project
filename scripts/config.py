"""
Central configuration for the whole project.

Every script imports from here so that parameters live in ONE place.
If you change a value here, it changes everywhere - do not hardcode
these numbers inside individual scripts.

Decisions referenced below (B9, B10, ...) are recorded in DECISIONS.md.
"""

import os
from pathlib import Path

# ----------------------------------------------------------------------
# PATHS
# Two places:
#   BASE      = Google Drive (permanent): code, archives, checkpoints, scores
#   DATA_BASE = where audio is read/written. Drive is very slow with many
#               small files, so once scripts/fast_data.py has unpacked the
#               archives to Colab's local disk, everything uses that copy.
# For a local dry run, set MP_BASE (and optionally MP_DATA).
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ENVIRONMENT AUTO-DETECTION (Kaggle vs Colab vs Local)
# ----------------------------------------------------------------------
IS_KAGGLE = "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle/working").exists()
IS_COLAB = "COLAB_GPU" in os.environ or Path("/content").exists()

if IS_KAGGLE:
    ENV_NAME = "Kaggle (30 GB RAM + 4 CPU Cores)"
    BASE = Path(os.environ.get("MP_BASE", "/kaggle/working/mini_project"))
    LOCAL_DATA = Path("/kaggle/fast") if Path("/kaggle/fast").exists() else Path("/kaggle/working/fast")
    NUM_WORKERS = min(8, max(4, os.cpu_count() or 4))  # 4 CPU workers for 2x parallel audio loading
    BATCH_SIZE = 32  # 32 batch size with Kaggle high RAM/GPU
    EVAL_BATCH_SIZE = 64
    ENHANCEMENT_BATCH_SIZE = 64
elif IS_COLAB:
    ENV_NAME = "Google Colab"
    BASE = Path(os.environ.get("MP_BASE", "/content/drive/MyDrive/mini_project"))
    LOCAL_DATA = Path("/content/fast")
    NUM_WORKERS = max(2, min(8, os.cpu_count() or 2))
    BATCH_SIZE = 24
    EVAL_BATCH_SIZE = 48
    ENHANCEMENT_BATCH_SIZE = 32
else:
    ENV_NAME = "Local PC / Other"
    BASE = Path(os.environ.get("MP_BASE", Path(__file__).resolve().parent.parent / "data"))
    LOCAL_DATA = BASE / "fast"
    NUM_WORKERS = max(2, min(8, os.cpu_count() or 2))
    BATCH_SIZE = 24
    EVAL_BATCH_SIZE = 48
    ENHANCEMENT_BATCH_SIZE = 32

# Helper to find first existing directory from a list of candidates
def _first_existing(candidates, default):
    for c in candidates:
        if c.exists():
            return c
    return default

# Intelligent path resolution across Kaggle / Colab / Drive
KAG_INPUTS = list(Path("/kaggle/input").glob("*")) if Path("/kaggle/input").exists() else []
KAG_SUBDIRS = [d for d in Path("/kaggle/input").glob("*/*") if d.is_dir()] if Path("/kaggle/input").exists() else []
ALL_KAG_DIRS = KAG_INPUTS + KAG_SUBDIRS

# Archive directory (check Drive, local working dir, or any Kaggle input dataset folder)
ARCHIVE_DIR = _first_existing(
    [BASE / "archives"] + [k / "archives" for k in ALL_KAG_DIRS] + [k for k in ALL_KAG_DIRS if list(k.glob("*.tar")) or "archive" in k.name.lower()],
    BASE / "archives"
)

# Audio data base directory
if os.environ.get("MP_DATA"):
    DATA_BASE = Path(os.environ["MP_DATA"])
elif (LOCAL_DATA / "asvspoof2019").exists() or (LOCAL_DATA / "ASVspoof2019").exists():
    DATA_BASE = LOCAL_DATA
else:
    DATA_BASE = BASE

# ── Fast Kaggle input detection: pruned directory walk (< 0.05s total) ───────
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


def find_file_fast(target_file: str, search_roots: list) -> Path | None:
    """
    Lightning-fast file finder.
    Prunes audio folders ('flac', 'wav') so it never traverses 100,000+ clips.
    Recurses instantaneously to arbitrary depth in < 0.05 seconds.
    """
    for base in search_roots:
        p = Path(base)
        if not p.exists():
            continue
        if (p / target_file).is_file():
            return p / target_file
        for root, dirs, files in os.walk(p):
            for skip in ("flac", "wav", ".git", "__pycache__"):
                if skip in dirs:
                    dirs.remove(skip)
            if target_file in files:
                return Path(root) / target_file
    return None

_kag_roots = [Path("/kaggle/input")] if Path("/kaggle/input").exists() else []

_asv_proto = (
    find_folder_fast("ASVspoof2019_LA_cm_protocols", _kag_roots) or
    find_folder_fast("ASVspoof2019_LA_asv_protocols", _kag_roots) or
    find_folder_fast("ASVspoof2019_LA_train", _kag_roots)
)
_asv_found = [_asv_proto.parent] if _asv_proto else []

ASVSPOOF_DIR = _first_existing(
    _asv_found +
    [LOCAL_DATA / "asvspoof2019", LOCAL_DATA / "ASVspoof2019",
     DATA_BASE / "asvspoof2019", DATA_BASE / "ASVspoof2019"] +
    [k / "asvspoof2019" for k in ALL_KAG_DIRS] +
    [k / "ASVspoof2019" for k in ALL_KAG_DIRS] +
    [k for k in ALL_KAG_DIRS if "asvspoof" in k.name.lower() and not k.name.endswith(".tar")],
    LOCAL_DATA / "asvspoof2019"
)

# Noise (MS-SNSD noise_train) and RIRS (simulated_rirs)
_noise_train_found = find_folder_fast("noise_train", _kag_roots)
_rirs_found = find_folder_fast("simulated_rirs", _kag_roots)

MSSNSD_DIR = _first_existing(
    ([_noise_train_found.parent] if _noise_train_found else []) +
    [LOCAL_DATA / "MS-SNSD", DATA_BASE / "MS-SNSD"] + [k / "MS-SNSD" for k in ALL_KAG_DIRS] +
    [k for k in ALL_KAG_DIRS if "ms-snsd" in k.name.lower() or "mssnsd" in k.name.lower() or "noise" in k.name.lower()],
    LOCAL_DATA / "MS-SNSD"
)

RIRS_DIR = _first_existing(
    ([_rirs_found.parent.parent] if _rirs_found else []) +
    [LOCAL_DATA / "RIRS_NOISES", DATA_BASE / "RIRS_NOISES"] + [k / "RIRS_NOISES" for k in ALL_KAG_DIRS] +
    [k for k in ALL_KAG_DIRS if "rirs" in k.name.lower()],
    LOCAL_DATA / "RIRS_NOISES"
)

REPO_DIR = Path(__file__).resolve().parent.parent

AASIST_DIR = _first_existing(
    [BASE / "aasist", REPO_DIR / "aasist"] + [k / "aasist" for k in ALL_KAG_DIRS] + [k for k in ALL_KAG_DIRS if ("aasist" in k.name.lower() and (k / "main.py").exists())],
    BASE / "aasist"
)

MODEL_A_WEIGHTS = AASIST_DIR / "models" / "weights" / "AASIST.pth"

# Train and test noise paths
NOISE_TRAIN_DIR = _noise_train_found if _noise_train_found else MSSNSD_DIR / "noise_train"
NOISE_TEST_DIR = (NOISE_TRAIN_DIR.parent / "noise_test") if _noise_train_found else MSSNSD_DIR / "noise_test"
RIR_DIR = _rirs_found if _rirs_found else RIRS_DIR / "simulated_rirs"

# Generated data (noisy / enhanced / train_aug)
GENERATED_DIR = LOCAL_DATA / "generated" if (LOCAL_DATA / "generated").exists() else DATA_BASE / "generated"
EVAL_CONDITIONS_DIR = GENERATED_DIR / "eval"

# train_aug: find protocol_aug.txt directly via fast pruned search
_proto_aug_hit = find_file_fast("protocol_aug.txt", _kag_roots)
if _proto_aug_hit is not None:
    _train_aug_found = _proto_aug_hit.parent
else:
    _train_aug_found = find_folder_fast("train_aug", _kag_roots)
    if _train_aug_found and (_train_aug_found / "generated" / "train_aug").exists():
        _train_aug_found = _train_aug_found / "generated" / "train_aug"

_train_aug_list = [_train_aug_found] if _train_aug_found else []

TRAIN_AUG_DIR = _first_existing(
    _train_aug_list +
    [LOCAL_DATA / "generated" / "train_aug", DATA_BASE / "generated" / "train_aug", BASE / "generated" / "train_aug"] +
    [k / "generated" / "train_aug" for k in ALL_KAG_DIRS] +
    [k / "train_aug" for k in ALL_KAG_DIRS] +
    [k for k in ALL_KAG_DIRS if "train_aug" in k.name.lower() and not k.name.endswith(".tar")],
    LOCAL_DATA / "generated" / "train_aug"
)

# Checkpoints (auto-search Drive, Kaggle working dir, and Kaggle input datasets)
CHECKPOINT_DIR = _first_existing(
    [BASE / "checkpoints"] + [k / "checkpoints" for k in ALL_KAG_DIRS] + [k for k in ALL_KAG_DIRS if "checkpoint" in k.name.lower()],
    BASE / "checkpoints"
)

SCORES_DIR = BASE / "scores"
RESULTS_DIR = REPO_DIR / "results"

# Fixed eval subset shared by the whole team (commit this file once made)
EVAL_SUBSET_FILE = REPO_DIR / "configs" / "eval_subset_10k.txt"

# ----------------------------------------------------------------------
# AUDIO
# ----------------------------------------------------------------------
SAMPLE_RATE = 16000     # ASVspoof and AASIST both expect 16 kHz

# ----------------------------------------------------------------------
# DATA SEED
# One fixed seed for everything that builds DATA (subset choice, noise
# draws). Training seeds (SEEDS below) only change model training, so all
# three Model B runs see exactly the same augmented data.
# ----------------------------------------------------------------------
DATA_SEED = 42

# ----------------------------------------------------------------------
# TEST SETS  (decisions B10, E28)
# Clean test uses the FULL eval set (compare with published 0.83%).
# Noisy / enhanced tests use a fixed 10k subset at three fixed SNRs.
# ----------------------------------------------------------------------
EVAL_SUBSET_SIZE = 10_000
EVAL_SNRS_DB = [0, 10, 20]      # same levels as Paper 1
REVERB_PROBABILITY = 0.7        # fraction of test clips that also get room echo

# ----------------------------------------------------------------------
# TRAINING DATA  (decision B9)
# 10% of the training files, 3 degraded copies each (+30% data):
#   _n  = background noise only (random SNR in the range below)
#   _r  = room echo only
#   _nr = room echo + background noise
# A lighter recipe inspired by Ali et al. (Paper 1), who made 9 copies.
# ----------------------------------------------------------------------
AUGMENT_FRACTION = 0.10
AUG_COPIES = ["n", "r", "nr"]
TRAIN_SNR_MIN_DB = 0
TRAIN_SNR_MAX_DB = 20

# Batch size parameters are auto-configured above based on environment (IS_KAGGLE vs IS_COLAB)

# E29: AASIST's default is 100. Time one epoch first (train_model.py prints
# it). If 100 x 3 seeds does not fit, lower this AND train the clean
# control model (tag C) with the same value.
NUM_EPOCHS = 100

# Training progress is saved this often (minutes), even mid-epoch, so a
# Colab disconnect loses at most this much work.
SAVE_EVERY_MINUTES = 10

# B15: fixed seeds. Do not change once T14 has started.
SEEDS = [42, 123, 2024]

# ----------------------------------------------------------------------
# EVALUATION  (decision B16)
# ----------------------------------------------------------------------
EER_TOLERANCE_PP = 0.5
AASIST_PUBLISHED_EER = 0.83

# ----------------------------------------------------------------------
# SPEECH ENHANCEMENT MODELS  (decision B11)
# MetricGAN+ is required. The second enhancer is SEGAN+ if its weights
# can be obtained within one session; otherwise SepFormer (listed here).
# ----------------------------------------------------------------------
ENHANCEMENT_MODELS = {
    "metricgan": "speechbrain/metricgan-plus-voicebank",
    "sepformer": "speechbrain/sepformer-wham16k-enhancement",
}


def eval_conditions():
    """Every test condition name, in table order."""
    names = ["clean", "clean_subset"]
    names += [f"noisy_snr{s}" for s in EVAL_SNRS_DB]
    for enh in ENHANCEMENT_MODELS:
        names += [f"{enh}_snr{s}" for s in EVAL_SNRS_DB]
    return names


if __name__ == "__main__":
    print("Project configuration")
    print(f"  Environment:       {ENV_NAME}")
    print(f"  Base path:         {BASE}")
    print(f"  Audio read from:   {DATA_BASE}"
          + ("   (fast local disk)" if DATA_BASE == LOCAL_DATA else "   (Drive - slow)"))
    print(f"  CPU workers:       {NUM_WORKERS}")
    print(f"  Eval subset:       {EVAL_SUBSET_SIZE} files, SNRs {EVAL_SNRS_DB} dB")
    print(f"  Train augment:     {AUGMENT_FRACTION:.0%} x {len(AUG_COPIES)} copies")
    print(f"  Batch / epochs:    {BATCH_SIZE} / {NUM_EPOCHS}")
    print(f"  Seeds:             {SEEDS}")
    print(f"  Enhancers:         {list(ENHANCEMENT_MODELS)}")
    print(f"  Conditions:        {eval_conditions()}")
