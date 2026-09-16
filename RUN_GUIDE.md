# Run Guide: everything, start to finish

Repo: https://github.com/createsomethingmarvellous/mini-project
Everything runs in **Google Colab**. Data lives in Google Drive at `MyDrive/mini_project/`.
Steps run **one after another**, one person at a time. Tick each step on the task board in `GUIDE.md` when it's done.

---

## Part 0 · One-time preparation (Anup)

| # | Do this | Where |
|---|---|---|
| 0.1 | **Kaggle token:** kaggle.com → Settings → API → *Create New Token*. This downloads `kaggle.json`; keep it private. | kaggle.com |
| 0.2 | **GitHub token** (lets Colab pull and push the repo): GitHub → Settings → Developer settings → Personal access tokens → **Fine-grained** → *Generate new token*. Pick only `mini-project`, set **Contents: Read and write**, and an expiry after Dec 2026. Copy it. | github.com |
| 0.3 | **Save the token in Colab:** open any Colab notebook → **🔑 key icon** (left bar) → add `GITHUB_USER` = `createsomethingmarvellous` and `GITHUB_TOKEN` = your token → switch on *Notebook access* for both. | Colab |

Never paste a token into a notebook cell, a chat or a file.

---

## Part 1 · How to open a notebook

- **First time (before the repo is on Drive):** colab.research.google.com → File → Open notebook → **GitHub** tab → `createsomethingmarvellous/mini-project` → `notebooks/01_setup.ipynb`.
- **After that:** Google Drive → `mini_project/repo/notebooks/` → double-click the notebook → **Open with Google Colab**.
- **Every time:** Runtime → Change runtime type → **GPU** → Save. Then run the **first code cell** before anything else.

---

## Part 2 · The steps

### Step 1 · Setup: `01_setup.ipynb` (Anup, once, ~1–2 h)
Run every cell from top to bottom.
- **Drive:** allow access when asked.
- **`kaggle.json`:** upload it when asked.
- **Dataset download:** the ASVspoof download is large, so wait for it.

✅ **Done when** the last cell shows `OK` on every line.
It also creates and commits `configs/eval_subset_10k.txt` (the 10,000 test files everyone uses).

### Step 2 · Sanity check: `04_evaluate.ipynb`, T7 cell only (~20–40 min)
Run the first two cells, then:
```
!python scripts/run_eval.py --model A --conditions clean
```
✅ **Must print `OK`** (EER near 0.83%). If it says `NOT OK`, **stop** and fix before going on.

### Step 3 · Build the data: `02_build_data.ipynb` (Anup, several hours)
Run the cells in order. Each builder runs as a **20-file test first**, then in full.
1. Noisy test audio (T6): 3 × 10,000 files. Listen to the sample cell once.
2. Noisy training audio (T9): about 7,600 files.
3. Cleaned audio with MetricGAN+ (T12), then with SepFormer.

✅ **Done when** each count cell shows 10,000 (test sets) and "files built = expected" (training set).
💡 If Colab disconnects, run the same cell again: finished files are skipped.

### Step 4 · Train Model B: `03_train.ipynb` (longest step)
1. Run all cells with `SEED = 42`.
2. **After the first epoch, read the log line** (`... min training, ~X h left`) and tell the team.
   - If the hours left are more than your schedule allows, lower `NUM_EPOCHS` in `scripts/config.py`, delete `checkpoints/B_seed42/`, and restart. Record the choice in `DECISIONS.md` (E29).
3. If Colab disconnects, **run all cells again**. It continues from the last save (every 10 minutes).

✅ **Done when** the notebook prints `Done. Final model: .../swa.pth`.

### Step 5 · Test: `04_evaluate.ipynb` (1–3 h)
Run all cells. This scores Model A and Model B (seed 42) on clean, noisy and cleaned audio, then builds the first tables.
✅ **Done when** the tables appear at the bottom.

### Step 6 · Repeat for reliability
1. `03_train.ipynb` with `SEED = 123` → wait for `Done`.
2. `03_train.ipynb` with `SEED = 2024` → wait for `Done`.
3. **Only if you lowered `NUM_EPOCHS`:** also train the clean control, with `TAG = 'C'` for seeds 42, 123 and 2024.
4. `04_evaluate.ipynb` → run all cells again (already-scored parts are skipped).

✅ **Done when** the tables show Model B as `mean ± std (n=3)`.

### Step 7 · Results and write-up
- The tables are in `results/` (`main_table`, `per_snr_table`, `per_attack_table`) and are committed automatically.
- Write the results section (T16) using `GUIDE.md` → T16 and `docs/literature_survey.md`.

---

## Part 3 · Teammates taking a turn (once, ~5 min)
1. Anup shares the Drive folder `mini_project` with them as **Editor**.
2. They go to *Shared with me* → right-click `mini_project` → Organize → **Add shortcut** → My Drive. Keep the name `mini_project`.
3. They open notebooks from Drive (Part 1) and run the first cell. "No GitHub secrets found" is fine.

They don't need `kaggle.json`, a GitHub token, or notebook 01.
Steps 1 and 3 stay with Anup (big files count against the creator's Drive).
Only Anup commits from Colab.

---

## Part 4 · If something goes wrong
| Problem | Fix |
|---|---|
| "No GPU" | Runtime → Change runtime type → GPU, then rerun the first cell |
| "CUDA out of memory" in training | In `scripts/config.py` set `BATCH_SIZE = 12`, delete that seed's checkpoint folder, restart |
| Colab disconnected | Rerun the same cells; builders skip finished files and training resumes |
| "Repo not found" | Run notebook 01 first (or add the Drive shortcut, Part 3) |
| "No ASVspoof2019_LA_cm_protocols folder" | Step 1's download or unzip didn't finish; rerun those cells |
| Clone fails with `403` / "Write access to repository not granted" | The token can't access this repo. Edit it on GitHub (Repository access: `mini-project`; Contents: **Read and write**), update the `GITHUB_TOKEN` secret if the token changed, then rerun the cell |
| `git push` failed in a cell | Check the two Colab secrets (Part 0.3); results are still saved on Drive |
| Any other error | Copy the full red error text and ask for help; don't change settings mid-way |

**Never change `scripts/config.py` in the middle of the experiment** without telling the team and noting it in `DECISIONS.md`.
