"""
Task T6a - choose the fixed 10k eval subset used for every noisy test.

Keeps the same mix of bona fide speech and each attack (A07-A19) as the
full eval set. Run ONCE, then commit configs/eval_subset_10k.txt so the
whole team uses identical files.

Usage (in Colab):
    !python scripts/make_eval_subset.py
"""

import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import config
from common import protocol_path, read_protocol, write_protocol


def main():
    out = config.EVAL_SUBSET_FILE
    if out.exists():
        sys.exit(f"{out} already exists - delete it only if the team agrees.")

    rows = read_protocol(protocol_path("eval"))
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row[3]].append(i)        # column 4: '-' (bona fide) or A07..A19

    # proportional share per group, rounded so the total is exactly the target
    shares = {s: len(groups[s]) * config.EVAL_SUBSET_SIZE / len(rows) for s in groups}
    counts = {s: int(v) for s, v in shares.items()}
    leftover = config.EVAL_SUBSET_SIZE - sum(counts.values())
    for s in sorted(shares, key=lambda s: shares[s] - counts[s], reverse=True)[:leftover]:
        counts[s] += 1

    rng = random.Random(config.DATA_SEED)
    chosen = []
    for src in sorted(groups):
        chosen += rng.sample(groups[src], counts[src])

    subset = [rows[i] for i in sorted(chosen)]
    write_protocol(out, subset)

    print(f"Wrote {len(subset)} of {len(rows)} eval files to {out}")
    for src in sorted(groups):
        print(f"  {src:>4}: {sum(r[3] == src for r in subset)}")
    print("Now commit this file so everyone uses the same subset.")


if __name__ == "__main__":
    main()
