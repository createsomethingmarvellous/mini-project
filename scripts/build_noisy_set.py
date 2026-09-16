"""
Tasks T6 and T9 - build noisy audio sets.

T6 (--mode eval):  the 10k eval subset at each fixed SNR (0, 10, 20 dB),
                   using TEST noises. 70% of clips also get room echo.
                   -> generated/eval/noisy_snr{X}/flac/
T9 (--mode train): 10% of the training files, 3 copies each
                   (noise / echo / echo+noise), using TRAIN noises.
                   -> generated/train_aug/flac/ + protocol_aug.txt

Each file gets its own fixed random seed, so:
  - re-running gives identical audio (reproducible);
  - the same clip gets the same noise sample and echo at every SNR,
    so the SNR levels differ ONLY in loudness of the noise;
  - finished files are skipped, so a Colab disconnect loses nothing.

Usage (in Colab) - always try --limit 20 first:
    !python scripts/build_noisy_set.py --mode eval --limit 20
    !python scripts/build_noisy_set.py --mode eval
    !python scripts/build_noisy_set.py --mode train
"""

import argparse
import random
import sys
from pathlib import Path

from audiomentations import AddBackgroundNoise, ApplyImpulseResponse, Compose
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
import config
from common import (eval_rows_for, load_audio, protocol_path, read_protocol,
                    save_flac, seed_everything, split_dir, write_protocol)


def noise(noise_dir, snr_min, snr_max, p=1.0):
    return AddBackgroundNoise(sounds_path=str(noise_dir),
                              min_snr_db=snr_min, max_snr_db=snr_max, p=p)


def reverb(p):
    return ApplyImpulseResponse(ir_path=str(config.RIR_DIR), p=p)


def check_inputs(*paths):
    for path in paths:
        if not path.exists():
            sys.exit(f"ERROR: {path} not found. Fix scripts/config.py or finish T2-T4.")


def process(jobs, desc):
    """jobs: list of (seed, src_path, out_path, augmenter)."""
    failed = skipped = 0
    for seed, src, out, augment in tqdm(jobs, desc=desc):
        if out.exists():
            skipped += 1
            continue
        try:
            seed_everything(seed)
            audio = load_audio(src)
            save_flac(out, augment(samples=audio, sample_rate=config.SAMPLE_RATE))
        except Exception as e:
            failed += 1
            if failed <= 5:
                print(f"\n  failed on {src.name}: {e}")
    print(f"{desc}: {len(jobs) - failed - skipped} written, "
          f"{skipped} already existed, {failed} failed")
    if failed:
        print("WARNING: investigate failures before continuing.")


def build_eval(limit):
    check_inputs(config.NOISE_TEST_DIR, config.RIR_DIR, split_dir("eval"))
    rows = eval_rows_for("noisy")[:limit]
    src_dir = split_dir("eval") / "flac"

    for snr in config.EVAL_SNRS_DB:
        # Echo first (speaker in a room), then background noise on top.
        augment = Compose([reverb(config.REVERB_PROBABILITY),
                           noise(config.NOISE_TEST_DIR, snr, snr)])
        out_dir = config.EVAL_CONDITIONS_DIR / f"noisy_snr{snr}" / "flac"
        jobs = [(config.DATA_SEED + i, src_dir / f"{r[1]}.flac",
                 out_dir / f"{r[1]}.flac", augment)
                for i, r in enumerate(rows)]
        process(jobs, f"noisy_snr{snr}")


def build_train(limit):
    check_inputs(config.NOISE_TRAIN_DIR, config.RIR_DIR, split_dir("train"))
    rows = read_protocol(protocol_path("train"))
    n = int(len(rows) * config.AUGMENT_FRACTION)
    picked = random.Random(config.DATA_SEED).sample(rows, n)[:limit]
    print(f"Augmenting {len(picked)} of {len(rows)} training files "
          f"x {len(config.AUG_COPIES)} copies")

    lo, hi = config.TRAIN_SNR_MIN_DB, config.TRAIN_SNR_MAX_DB
    augmenters = {
        "n": Compose([noise(config.NOISE_TRAIN_DIR, lo, hi)]),
        "r": Compose([reverb(1.0)]),
        "nr": Compose([reverb(1.0), noise(config.NOISE_TRAIN_DIR, lo, hi)]),
    }

    src_dir = split_dir("train") / "flac"
    out_dir = config.TRAIN_AUG_DIR / "flac"
    jobs, new_rows = [], []
    for i, row in enumerate(picked):
        for c, copy in enumerate(config.AUG_COPIES):
            key = f"{row[1]}_{copy}"
            seed = config.DATA_SEED + i * len(config.AUG_COPIES) + c
            jobs.append((seed, src_dir / f"{row[1]}.flac",
                         out_dir / f"{key}.flac", augmenters[copy]))
            new_rows.append([row[0], key, row[2], row[3], row[4]])

    # Protocol first, so the training script knows every expected file.
    write_protocol(config.TRAIN_AUG_DIR / "protocol_aug.txt", new_rows)
    process(jobs, "train_aug")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["eval", "train"], required=True)
    parser.add_argument("--limit", type=int, default=None,
                        help="process only the first N source files (test run)")
    args = parser.parse_args()

    if args.mode == "eval":
        build_eval(args.limit)
    else:
        build_train(args.limit)


if __name__ == "__main__":
    main()
