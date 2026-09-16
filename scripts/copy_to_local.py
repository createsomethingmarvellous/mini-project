"""
Speed-up for training: copy the training data from Drive to Colab's local disk.

Reading ~30k small files from Google Drive every epoch is very slow.
This copies only what training needs (train + dev audio, protocols, and
the augmented copies) to /content/fast, keeping the same folder layout.
Afterwards set MP_DATA=/content/fast (notebook 03 does this).

Colab's local disk is wiped when the session ends, so run this once at
the start of every training session. Files already copied are skipped.

Usage (in Colab, BEFORE setting MP_DATA):
    !python scripts/copy_to_local.py
"""

import os
import shutil
import sys
from pathlib import Path

if os.environ.get("MP_DATA"):
    sys.exit("Unset MP_DATA before copying (the source must be Drive).")

sys.path.insert(0, str(Path(__file__).parent))
import config
from common import la_root

DEST = Path("/content/fast")


def copy_tree(src: Path, dst: Path):
    if not src.exists():
        print(f"  skip (not found): {src}")
        return
    print(f"  {src} -> {dst}")
    shutil.copytree(src, dst, dirs_exist_ok=True,
                    copy_function=lambda s, d: None if Path(d).exists()
                    else shutil.copy2(s, d))


def main():
    root = la_root()
    rel = root.relative_to(config.BASE)
    for name in ["ASVspoof2019_LA_cm_protocols",
                 "ASVspoof2019_LA_train", "ASVspoof2019_LA_dev"]:
        copy_tree(root / name, DEST / rel / name)
    copy_tree(config.TRAIN_AUG_DIR, DEST / "generated" / "train_aug")
    print(f"Done. Now set MP_DATA={DEST}")


if __name__ == "__main__":
    main()
