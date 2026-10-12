"""
Utility script to verify, inspect, and bundle trained model checkpoints
into a downloadable zip file on Kaggle or Colab.

Usage:
    python scripts/bundle_checkpoints.py
"""

import os
import sys
import zipfile
from pathlib import Path

# Add scripts directory to path for config
sys.path.insert(0, str(Path(__file__).parent))
try:
    import config
    default_ckpt_dir = config.CHECKPOINT_DIR
except Exception:
    default_ckpt_dir = Path("checkpoints")

def bundle_checkpoints(output_zip: Path = None):
    # Candidate locations for checkpoints directory
    candidates = [
        default_ckpt_dir,
        Path("/kaggle/working/mini_project/checkpoints"),
        Path("/kaggle/working/checkpoints"),
        Path("./checkpoints"),
        Path("./data/checkpoints"),
    ]
    
    ckpt_dir = None
    for cand in candidates:
        if cand.exists() and any(cand.rglob("*.pth")) or any(cand.rglob("*.pt")):
            ckpt_dir = cand
            break
            
    if ckpt_dir is None:
        for cand in candidates:
            if cand.exists():
                ckpt_dir = cand
                break
                
    if ckpt_dir is None or not ckpt_dir.exists():
        print(f"[!] Checkpoint directory not found among candidates: {[str(c) for c in candidates]}")
        return None

    # Search for all checkpoint files (.pth, .pt, .txt, .json)
    files_to_bundle = []
    for ext in ["*.pth", "*.pt", "*.txt", "*.json"]:
        files_to_bundle.extend(list(ckpt_dir.rglob(ext)))

    if not files_to_bundle:
        print(f"[!] No checkpoint files (.pth / .pt / .txt) found under: {ckpt_dir}")
        print("    Make sure training has completed at least one epoch or save interval.")
        return None

    # Output zip location
    if output_zip is None:
        if Path("/kaggle/working").exists():
            output_zip = Path("/kaggle/working/checkpoints_bundle.zip")
        else:
            output_zip = Path("./checkpoints_bundle.zip")

    output_zip.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("           CHECKPOINT VERIFICATION & BUNDLING")
    print("=" * 65)
    print(f"Scanning directory: {ckpt_dir}\n")

    total_bytes = 0
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(set(files_to_bundle)):
            rel_name = f.relative_to(ckpt_dir).as_posix()
            size_mb = f.stat().st_size / 1e6
            total_bytes += f.stat().st_size
            zf.write(f, arcname=f"checkpoints/{rel_name}")
            print(f"  + Packed: {rel_name:<35} ({size_mb:6.2f} MB)")

    zip_mb = output_zip.stat().st_size / 1e6
    print("-" * 65)
    print(f"Total files packed : {len(files_to_bundle)}")
    print(f"Uncompressed size  : {total_bytes / 1e6:.2f} MB")
    print(f"Compressed zip size: {zip_mb:.2f} MB")
    print(f"Destination        : {output_zip.resolve()}")
    print("=" * 65)
    print("[SUCCESS] All checkpoints verified and bundled successfully!")
    print("\nHow to download:")
    print("1. In Kaggle UI right sidebar: Check 'Output' -> 'checkpoints_bundle.zip' -> Download")
    print("2. Or click the FileLink rendered in the notebook cell below.")
    
    return output_zip

if __name__ == "__main__":
    bundle_checkpoints()
