"""
Bundle one trained run, its score files, and generated summary tables for review.

Usage:
    python scripts/bundle_checkpoints.py --tag B --seed 123
"""

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config


def bundle_checkpoints(output_zip: Path = None, tag: str = None, seed: int = None) -> Path:
    if (tag is None) != (seed is None):
        raise ValueError("Specify both tag and seed, or neither.")

    if tag is not None:
        run_name = f"{tag}_seed{seed}"
        candidates = [
            config.CHECKPOINT_DIR / run_name,
            config.BASE / "checkpoints" / run_name,
            Path("/kaggle/working/mini_project/checkpoints") / run_name,
            Path("/kaggle/working/checkpoints") / run_name,
            Path("./checkpoints") / run_name,
            Path("./data/checkpoints") / run_name,
        ]
        checkpoint_dir = next((path for path in candidates if path.is_dir()), None)
        if checkpoint_dir is None and Path("/kaggle/working").exists():
            for p in Path("/kaggle/working").rglob(run_name):
                if p.is_dir():
                    checkpoint_dir = p
                    break
        if checkpoint_dir is None:
            raise FileNotFoundError(f"Checkpoint directory not found for {run_name}. Searched: {[str(c) for c in candidates]}")
        checkpoint_files = sorted(
            path for path in checkpoint_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".txt", ".json"}
        )
    else:
        run_name = "all_runs"
        candidates = [
            config.CHECKPOINT_DIR,
            config.BASE / "checkpoints",
            Path("/kaggle/working/mini_project/checkpoints"),
            Path("/kaggle/working/checkpoints"),
            Path("./checkpoints"),
            Path("./data/checkpoints"),
        ]
        checkpoint_dir = next(
            (path for path in candidates
             if path.is_dir() and any(path.rglob("*.pth"))),
            None,
        )
        if checkpoint_dir is None:
            raise FileNotFoundError(f"No checkpoint directory found among: {candidates}")
        checkpoint_files = sorted(
            path for path in checkpoint_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".pth", ".pt", ".txt", ".json"}
        )

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint or training log files found in {checkpoint_dir}")

    swa_file = checkpoint_dir / "swa.pth"
    best_dev_file = checkpoint_dir / "best_dev.pth"
    last_ckpt_file = checkpoint_dir / "last_checkpoint.pth"

    if not swa_file.is_file():
        if best_dev_file.is_file():
            import shutil
            shutil.copyfile(best_dev_file, swa_file)
            print(f"ℹ️ Training in progress / paused: copied best_dev.pth to swa.pth for review and evaluation.")
            if swa_file not in checkpoint_files:
                checkpoint_files.append(swa_file)
                checkpoint_files.sort()
        elif last_ckpt_file.is_file():
            try:
                import torch
                ckpt = torch.load(last_ckpt_file, map_location="cpu", weights_only=False)
                state = ckpt.get("state", {})
                weights_to_save = state.get("swa") or ckpt.get("model")
                if weights_to_save:
                    clean_w = {(k[7:] if k.startswith("module.") else k): v for k, v in weights_to_save.items()}
                    torch.save(clean_w, swa_file)
                    print(f"ℹ️ Training in progress / paused: synthesized swa.pth from last_checkpoint.pth.")
                    if swa_file not in checkpoint_files:
                        checkpoint_files.append(swa_file)
                        checkpoint_files.sort()
            except Exception as e:
                print(f"⚠️ Notice: swa.pth could not be auto-generated: {e}")
        else:
            print(f"⚠️ Notice: swa.pth and best_dev.pth not found in {checkpoint_dir}.")

    score_root = config.SCORES_DIR
    score_files = sorted(score_root.glob("*/*.txt")) if score_root.exists() else []
    run_score_files = (
        [path for path in score_files if path.parent.name == run_name]
        if tag is not None else score_files
    )
    expected_conditions = config.eval_conditions()
    scored_conditions = {path.stem for path in run_score_files}
    missing_conditions = (
        [name for name in expected_conditions if name not in scored_conditions]
        if tag is not None else []
    )

    results_root = config.RESULTS_DIR
    summary_files = (
        sorted(path for path in results_root.iterdir()
               if path.is_file() and path.suffix.lower() in {".md", ".csv"})
        if results_root.exists() else []
    )
    expected_tables = {
        "main_table.md", "main_table.csv",
        "per_snr_table.md", "per_snr_table.csv",
        "per_attack_table.md", "per_attack_table.csv",
    }
    missing_tables = sorted(expected_tables - {path.name for path in summary_files})

    if output_zip is None:
        output_root = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path.cwd()
        output_name = f"{run_name}_final_review.zip" if tag is not None else "checkpoints_bundle.zip"
        output_zip = output_root / output_name
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "model": tag or "all",
        "seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint_files": [path.name for path in checkpoint_files],
        "score_files": [
            path.relative_to(score_root).as_posix() for path in score_files
        ],
        "expected_evaluation_conditions": expected_conditions,
        "missing_evaluation_conditions_for_this_run": missing_conditions,
        "summary_files": [path.name for path in summary_files],
        "missing_summary_tables": missing_tables,
        "evaluation_complete_for_this_run": not missing_conditions if tag is not None else None,
        "summary_tables_complete": not missing_tables and not missing_conditions,
    }

    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in checkpoint_files:
            archive.write(path, arcname=f"checkpoints/{run_name}/{path.name}")
        for path in score_files:
            archive.write(path, arcname=f"scores/{path.relative_to(score_root).as_posix()}")
        for path in summary_files:
            archive.write(path, arcname=f"results/{path.name}")
        archive.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")

    if missing_conditions:
        print("WARNING: evaluation is incomplete for this seed; missing conditions:")
        for condition in missing_conditions:
            print(f"  - {condition}")
    if missing_tables:
        print("WARNING: summary tables missing:")
        for table in missing_tables:
            print(f"  - {table}")
    elif missing_conditions:
        print("WARNING: summary table files exist, but they may be incomplete because this seed is missing evaluation conditions.")

    print(f"Review bundle created: {output_zip}")
    print(f"  Checkpoint/training files: {len(checkpoint_files)}")
    print(f"  Score files (all runs):   {len(score_files)}")
    print(f"  Summary tables:           {len(summary_files)}")
    print(f"  Evaluation complete:      {manifest['evaluation_complete_for_this_run']}")
    print(f"  Summary tables complete:  {manifest['summary_tables_complete']}")
    return output_zip


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", choices=["B", "C"])
    parser.add_argument("--seed", type=int, choices=config.SEEDS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if (args.tag is None) != (args.seed is None):
        parser.error("--tag and --seed must be supplied together.")
    bundle_checkpoints(args.output, args.tag, args.seed)


if __name__ == "__main__":
    main()
