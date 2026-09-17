# T7 · Sanity check (Model A, pretrained AASIST)

Run on 17 Sep 2026 in Colab (Tesla T4), `scripts/run_eval.py --model A`.

| Test set | Files | EER |
|---|---|---|
| Full ASVspoof 2019 LA eval (clean) | 71,237 | **0.830%** |
| Fixed 10k subset (clean) | 10,000 | 0.783% |

Published AASIST EER: 0.83%. Gap: 0.00 percentage points (tolerance 0.5, decision B16) -> **pipeline validated**.
Raw scores: `MyDrive/mini_project/scores/A/clean.txt`, `clean_subset.txt`.
