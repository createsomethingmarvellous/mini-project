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
        weights = config.CHECKPOINT_DIR / tag / "swa.pth"
    if not weights.exists():
        sys.exit(f"ERROR: weights not found: {weights}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    conf = ab.load_aasist_config()
    model = ab.build_model(conf["model_config"], device)
    model.load_state_dict(torch.load(weights, map_location=device))
    print(f"Model {tag} loaded from {weights}")

    for cond in args.conditions:
        out = config.SCORES_DIR / tag / f"{cond}.txt"
        if out.exists() and not args.overwrite:
            print(f"  {cond:<16} already scored - skipping ({out})")
            continue
        rows = eval_rows_for(cond)
        base = condition_dir(cond)
        missing = [r[1] for r in rows if not (base / "flac" / f"{r[1]}.flac").exists()]
        if missing:
            print(f"  {cond:<16} SKIPPED - {len(missing)} of {len(rows)} files not built yet")
            continue
        loader = ab.eval_loader(rows, base, conf["batch_size"])
        eer = ab.write_scores(model, loader, rows, out, device)
        print(f"  {cond:<16} EER = {eer:6.3f}%   ({len(rows)} files)")

        if tag == "A" and cond == "clean":
            gap = abs(eer - config.AASIST_PUBLISHED_EER)
            verdict = "OK" if gap <= config.EER_TOLERANCE_PP else "NOT OK - debug before continuing"
            print(f"  T7 sanity check: published {config.AASIST_PUBLISHED_EER}%, "
                  f"gap {gap:.2f} pp -> {verdict}")


if __name__ == "__main__":
    main()
