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

# Archive directory (check Drive, local working dir, or Kaggle input datasets)
ARCHIVE_DIR = _first_existing(
    [BASE / "archives"] + [k / "archives" for k in KAG_INPUTS] + [k for k in KAG_INPUTS if "archive" in k.name.lower()],
    BASE / "archives"
)

# Audio data base directory
if os.environ.get("MP_DATA"):
    DATA_BASE = Path(os.environ["MP_DATA"])
elif (LOCAL_DATA / "asvspoof2019").exists() or (LOCAL_DATA / "ASVspoof2019").exists():
    DATA_BASE = LOCAL_DATA
else:
    DATA_BASE = BASE

# Downloaded inputs (auto-finds ASVspoof, MS-SNSD, RIRS across Kaggle inputs & Drive)
ASVSPOOF_DIR = _first_existing(
    [DATA_BASE / "asvspoof2019", DATA_BASE / "ASVspoof2019"] +
    [k / "asvspoof2019" for k in KAG_INPUTS] + [k / "ASVspoof2019" for k in KAG_INPUTS] +
    [k for k in KAG_INPUTS if "asvspoof" in k.name.lower()],
    DATA_BASE / "asvspoof2019"
)

MSSNSD_DIR = _first_existing(
    [DATA_BASE / "MS-SNSD"] + [k / "MS-SNSD" for k in KAG_INPUTS] + [k for k in KAG_INPUTS if "ms-snsd" in k.name.lower()],
    DATA_BASE / "MS-SNSD"
)

RIRS_DIR = _first_existing(
    [DATA_BASE / "RIRS_NOISES"] + [k / "RIRS_NOISES" for k in KAG_INPUTS] + [k for k in KAG_INPUTS if "rirs" in k.name.lower()],
    DATA_BASE / "RIRS_NOISES"
)

REPO_DIR = Path(__file__).resolve().parent.parent

AASIST_DIR = _first_existing(
    [BASE / "aasist", REPO_DIR / "aasist"] + [k / "aasist" for k in KAG_INPUTS] + [k for k in KAG_INPUTS if "aasist" in k.name.lower()],
    BASE / "aasist"
)

MODEL_A_WEIGHTS = AASIST_DIR / "models" / "weights" / "AASIST.pth"

# Train and test noise paths
NOISE_TRAIN_DIR = MSSNSD_DIR / "noise_train"
NOISE_TEST_DIR = MSSNSD_DIR / "noise_test"
RIR_DIR = RIRS_DIR / "simulated_rirs"

# Generated data (noisy / enhanced / train_aug)
GENERATED_DIR = DATA_BASE / "generated"
EVAL_CONDITIONS_DIR = GENERATED_DIR / "eval"
TRAIN_AUG_DIR = GENERATED_DIR / "train_aug"

# Checkpoints (auto-search Drive, Kaggle working dir, and Kaggle input datasets)
CHECKPOINT_DIR = _first_existing(
    [BASE / "checkpoints"] + [k / "checkpoints" for k in KAG_INPUTS] + [k for k in KAG_INPUTS if "checkpoint" in k.name.lower()],
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
