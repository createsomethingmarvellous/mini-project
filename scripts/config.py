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
BASE = Path(os.environ.get("MP_BASE", "/content/drive/MyDrive/mini_project"))
LOCAL_DATA = Path("/content/fast")
ARCHIVE_DIR = BASE / "archives"                 # one .tar per dataset (fast_data.py)

if os.environ.get("MP_DATA"):
    DATA_BASE = Path(os.environ["MP_DATA"])
elif (LOCAL_DATA / "asvspoof2019").exists():
    DATA_BASE = LOCAL_DATA                      # unpacked this session: fast
else:
    DATA_BASE = BASE                            # fallback: files directly on Drive (slow)

# Downloaded inputs (tasks T2-T5)
ASVSPOOF_DIR = DATA_BASE / "asvspoof2019"   # searched automatically for the LA folder
MSSNSD_DIR = DATA_BASE / "MS-SNSD"
RIRS_DIR = DATA_BASE / "RIRS_NOISES"
AASIST_DIR = BASE / "aasist"
MODEL_A_WEIGHTS = AASIST_DIR / "models" / "weights" / "AASIST.pth"

# Train and test noise are kept apart, so Model B never hears a test noise.
NOISE_TRAIN_DIR = MSSNSD_DIR / "noise_train"
NOISE_TEST_DIR = MSSNSD_DIR / "noise_test"
RIR_DIR = RIRS_DIR / "simulated_rirs"

# Generated data (pack to Drive with fast_data.py after building)
GENERATED_DIR = DATA_BASE / "generated"
EVAL_CONDITIONS_DIR = GENERATED_DIR / "eval"    # eval/<condition>/flac/*.flac
TRAIN_AUG_DIR = GENERATED_DIR / "train_aug"     # flac/*.flac + protocol_aug.txt

# Outputs
CHECKPOINT_DIR = BASE / "checkpoints"           # checkpoints/<tag>/...
SCORES_DIR = BASE / "scores"                    # scores/<tag>/<condition>.txt
REPO_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_DIR / "results"              # small summary tables (committed)

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

# ----------------------------------------------------------------------
# TRAINING  (decisions B12, B15, E29)
# ----------------------------------------------------------------------
# Increase batch size to 22 for ~14.5 GB GPU VRAM utilization on Colab T4
BATCH_SIZE = 22

# Scoring only (no training): bigger batches use more of the GPU and give
# identical scores, because AASIST scores each file independently in eval mode.
# Batch 16 used ~3.7 GB on a T4; 48 uses roughly 10-11 GB of its 15 GB.
# If you see "CUDA out of memory" while scoring, lower this (e.g. 32).
EVAL_BATCH_SIZE = 48

# Speech enhancement inference batch size (uses ~12-13.5 GB VRAM on T4 GPU)
ENHANCEMENT_BATCH_SIZE = 32

# CPU processes for loading audio and building noisy sets.
NUM_WORKERS = max(2, min(8, os.cpu_count() or 2))

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
    print(f"  Base path (Drive): {BASE}")
    print(f"  Audio read from:   {DATA_BASE}"
          + ("   (fast local disk)" if DATA_BASE == LOCAL_DATA else "   (Drive - slow)"))
    print(f"  CPU workers:       {NUM_WORKERS}")
    print(f"  Eval subset:       {EVAL_SUBSET_SIZE} files, SNRs {EVAL_SNRS_DB} dB")
    print(f"  Train augment:     {AUGMENT_FRACTION:.0%} x {len(AUG_COPIES)} copies")
    print(f"  Batch / epochs:    {BATCH_SIZE} / {NUM_EPOCHS}")
    print(f"  Seeds:             {SEEDS}")
    print(f"  Enhancers:         {list(ENHANCEMENT_MODELS)}")
    print(f"  Conditions:        {eval_conditions()}")
