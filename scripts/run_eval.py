"""
Tasks T7, T8, T11, T13 - score a model on test conditions.

Writes scores/<model tag>/<condition>.txt (AASIST format) and prints EER.
Conditions whose audio is not built yet are skipped with a message.

Models:
    --model A                 pretrained AASIST.pth (Model A)
    --model B --seed 42       checkpoints/B_seed42/swa.pth
    --model C --seed 42       checkpoints/C_seed42/swa.pth (clean control)

Usage (in Colab):
    !python scripts/run_eval.py --model A --conditions clean      # T7 sanity check
    !python scripts/run_eval.py --model A                         # every condition
    !python scripts/run_eval.py --model B --seed 42
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
import config
from common import condition_dir, eval_rows_for
import aasist_bridge as ab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["A", "B", "C"], required=True)
    parser.add_argument("--seed", type=int, choices=config.SEEDS)
    parser.add_argument("--conditions", nargs="+", default=config.eval_conditions(),
                        help=f"default: all of {config.eval_conditions()}")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.model == "A":
        tag, weights = "A", config.MODEL_A_WEIGHTS
    else:
        if args.seed is None:
            sys.exit("--seed is required for models B and C")
        tag = f"{args.model}_seed{args.seed}"
        candidates = [
            config.CHECKPOINT_DIR / tag,
            config.BASE / "checkpoints" / tag,
            Path("/kaggle/working/mini_project/checkpoints") / tag,
            Path("/kaggle/working/checkpoints") / tag,
            Path("./checkpoints") / tag,
        ]
        target_dir = next((c for c in candidates if c.is_dir()), config.CHECKPOINT_DIR / tag)
        weights = target_dir / "swa.pth"
        if not weights.exists():
            best_dev = target_dir / "best_dev.pth"
            if best_dev.exists():
                print(f"ℹ️ Notice: swa.pth not found. Using best dev weights: {best_dev}")
                weights = best_dev
            else:
                last_ckpt = target_dir / "last_checkpoint.pth"
                if last_ckpt.exists():
                    print(f"ℹ️ Notice: swa.pth / best_dev.pth not found. Using last_checkpoint: {last_ckpt}")
                    weights = last_ckpt
                else:
                    sys.exit(f"ERROR: weights not found: {weights} (or {best_dev})")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = ab.load_aasist_config()
    model = ab.build_model(conf["model_config"], device)
    try:
        model_weights = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        model_weights = torch.load(weights, map_location=device)
    if isinstance(model_weights, dict) and "model" in model_weights:
        model_weights = model_weights.get("state", {}).get("swa") or model_weights["model"]
    clean_weights = {(k[7:] if k.startswith("module.") else k): v for k, v in model_weights.items()}
    model.load_state_dict(clean_weights)
    print(f"Model {tag} loaded from {weights}")
    print(f"Reading audio from {config.DATA_BASE}"
          + ("" if config.DATA_BASE == config.LOCAL_DATA else
             "  (Drive - slow! run: python scripts/fast_data.py unpack)"))

    for cond in args.conditions:
        out = config.SCORES_DIR / tag / f"{cond}.txt"
        if out.exists() and not args.overwrite:
            print(f"  {cond:<16} already scored - skipping ({out})")
            continue
        rows = eval_rows_for(cond)
        base = condition_dir(cond)
        flac_dir = base / "flac"
        have = set(os.listdir(flac_dir)) if flac_dir.exists() else set()  # one Drive listing, fast
        missing = [r[1] for r in rows if f"{r[1]}.flac" not in have]
        if missing:
            print(f"  {cond:<16} SKIPPED - {len(missing)} of {len(rows)} files not built yet")
            continue
        loader = ab.eval_loader(rows, base)
        print(f"  {cond:<16} scoring {len(rows)} files ...", flush=True)
        eer = ab.write_scores(model, loader, rows, out, device, progress=True)
        print(f"  {cond:<16} EER = {eer:6.3f}%   ({len(rows)} files)")

        if tag == "A" and cond == "clean":
            gap = abs(eer - config.AASIST_PUBLISHED_EER)
            verdict = "OK" if gap <= config.EER_TOLERANCE_PP else "NOT OK - debug before continuing"
            print(f"  T7 sanity check: published {config.AASIST_PUBLISHED_EER}%, "
                  f"gap {gap:.2f} pp -> {verdict}")


if __name__ == "__main__":
    main()
