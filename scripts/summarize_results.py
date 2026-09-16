"""
Tasks T15 / T16 - turn score files into the final tables.

Reads scores/<tag>/<condition>.txt and writes to results/:
  main_table.md / .csv       models x (clean, noisy, each enhancer); noisy and
                             enhanced columns are the average over the SNRs
  per_snr_table.md / .csv    each model x processing at each SNR
  per_attack_table.md / .csv EER for each attack type (A07-A19)
Seeds of the same model (B_seed42, B_seed123, ...) are reported as mean ± std.

Usage (in Colab, or locally after copying the scores folder):
    !python scripts/summarize_results.py
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import config
from common import add_aasist_to_path

add_aasist_to_path()
from evaluation import compute_eer  # noqa: E402  (official EER code)

ATTACKS = [f"A{i:02d}" for i in range(7, 20)]
PROCESSING = ["noisy"] + list(config.ENHANCEMENT_MODELS)


def load_scores(path: Path):
    src, key, score = [], [], []
    with open(path) as f:
        for line in f:
            _, s, k, v = line.split()
            src.append(s)
            key.append(k)
            score.append(float(v))
    return np.array(src), np.array(key), np.array(score)


def eer(bona, spoof):
    if len(bona) == 0 or len(spoof) == 0:
        return np.nan
    return 100 * compute_eer(bona, spoof)[0]


def fmt(values):
    """One number, or mean ± std across seeds."""
    values = [v for v in values if not np.isnan(v)]
    if not values:
        return "–"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f} (n={len(values)})"


def write_table(name, header, rows):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS_DIR / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows([header] + rows)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    (config.RESULTS_DIR / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n## {name}\n" + "\n".join(lines))


def main():
    # pooled[tag][cond] = EER ; attack[tag][cond][Axx] = EER
    pooled = defaultdict(dict)
    attack = defaultdict(lambda: defaultdict(dict))
    files = sorted(config.SCORES_DIR.glob("*/*.txt"))
    if not files:
        sys.exit(f"No score files under {config.SCORES_DIR}. Run run_eval.py first.")
    for path in files:
        tag, cond = path.parent.name, path.stem
        src, key, score = load_scores(path)
        bona = score[key == "bonafide"]
        pooled[tag][cond] = eer(bona, score[key == "spoof"])
        for a in ATTACKS:
            attack[tag][cond][a] = eer(bona, score[src == a])

    # group seeds: "B_seed42" -> "B"
    groups = defaultdict(list)
    for tag in pooled:
        groups[re.sub(r"_seed\d+$", "", tag)].append(tag)
    names = {"A": "Model A (pretrained, clean)",
             "B": "Model B (noise-augmented)",
             "C": "Model C (clean control)"}

    def snr_avg(tag, proc):
        vals = [pooled[tag].get(f"{proc}_snr{s}", np.nan) for s in config.EVAL_SNRS_DB]
        return np.nan if any(np.isnan(vals)) else float(np.mean(vals))

    # ---- main table ---------------------------------------------------------
    header = ["Model", "Clean (full)"] + [f"{p} (avg {len(config.EVAL_SNRS_DB)} SNRs)"
                                          for p in PROCESSING]
    rows = []
    for g in sorted(groups):
        tags = groups[g]
        row = [names.get(g, g), fmt([pooled[t].get("clean", np.nan) for t in tags])]
        row += [fmt([snr_avg(t, p) for t in tags]) for p in PROCESSING]
        rows.append(row)
    write_table("main_table", header, rows)

    # ---- per-SNR table ------------------------------------------------------
    header = ["Model", "Processing", "Clean subset"] + [f"{s} dB" for s in config.EVAL_SNRS_DB]
    rows = []
    for g in sorted(groups):
        tags = groups[g]
        for p in PROCESSING:
            row = [names.get(g, g), p,
                   fmt([pooled[t].get("clean_subset", np.nan) for t in tags])]
            row += [fmt([pooled[t].get(f"{p}_snr{s}", np.nan) for t in tags])
                    for s in config.EVAL_SNRS_DB]
            rows.append(row)
    write_table("per_snr_table", header, rows)

    # ---- per-attack table (clean subset + SNR-averaged conditions) ----------
    def attack_avg(tag, proc, a):
        if proc == "clean_subset":
            return attack[tag]["clean_subset"].get(a, np.nan)
        vals = [attack[tag][f"{proc}_snr{s}"].get(a, np.nan) for s in config.EVAL_SNRS_DB]
        return np.nan if any(np.isnan(vals)) else float(np.mean(vals))

    columns = [(g, p) for g in sorted(groups) for p in ["clean_subset"] + PROCESSING]
    header = ["Attack"] + [f"{g} {p}" for g, p in columns]
    rows = [[a] + [fmt([attack_avg(t, p, a) for t in groups[g]]) for g, p in columns]
            for a in ATTACKS]
    write_table("per_attack_table", header, rows)
    print(f"\nTables written to {config.RESULTS_DIR}")


if __name__ == "__main__":
    main()
