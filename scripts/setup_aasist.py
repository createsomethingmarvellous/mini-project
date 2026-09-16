"""
Task T5 helper - make the official AASIST code run on current Colab.

AASIST's evaluation.py uses `np.float`, which was removed in NumPy 1.24,
so it crashes on today's Colab. This replaces it with plain `float`
(same behaviour). Safe to run more than once.

Usage (in Colab, after cloning AASIST):
    !python scripts/setup_aasist.py
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config


def main():
    target = config.AASIST_DIR / "evaluation.py"
    if not target.exists():
        sys.exit(f"ERROR: {target} not found - clone AASIST first (T5).")
    text = target.read_text()
    fixed, n = re.subn(r"\bnp\.float\b(?!\d)", "float", text)
    if n:
        target.write_text(fixed)
    print(f"evaluation.py: replaced {n} np.float occurrence(s)")

    weights = config.MODEL_A_WEIGHTS
    print(f"Model A weights: {'found' if weights.exists() else 'MISSING'} ({weights})")


if __name__ == "__main__":
    main()
