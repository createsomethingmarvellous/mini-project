# Project Guide

This file has two jobs: give **complete context** to any human teammate *or* any AI assistant helping them, and provide a **task board** where whoever is free can pick up unblocked work without needing to ask anyone else what to do first.

**How to use this file:**
- New to the project, or picking up any task? Read the **Context** section once — it never needs re-reading, it applies to every task.
- Then go to the **Task Board**, find a `🟢 Ready` task with no owner, claim it, and jump to its detail section.
- Getting help from an AI assistant? Paste it the **Context** section plus your one task's detail section. That's the complete brief — no other explanation needed.

Last updated: this file's content should always match what's actually in the repo. If anything below looks out of date compared to the code, trust the code and fix this file.

---

## Context (read once, applies to every task)

### The problem, in plain terms
AI voice cloning is used in impersonation fraud (fake calls pretending to be a bank, boss, or family member). Detectors exist to catch cloned voices, but many detectors lean on a shortcut: cloned audio often sounds unnaturally "clean" (no background noise). Attackers defeat this by gluing realistic background noise or room echo back onto the cloned audio, hiding the shortcut cue.

### What we're actually testing
1. How much does this noise-masking trick actually hurt a real, published detector?
2. Two existing, separately-published fixes: (a) train the detector on noise-augmented data, (b) clean noisy audio with a speech-enhancement tool before testing.
3. **Our question:** does noise-augmented training make a detector less dependent on the artifacts that independent test-time enhancement destroys?

### What is already known (we build on these, we do not compete with them)
- **Paper 2 (Anacin et al., 2026)** — noise masking badly degrades AASIST on ASVspoof 2019; test-time enhancement partly recovers it. They explicitly left training data untouched.
- **Paper 1 (Ali et al., ASVspoof 2024 Workshop)**: trains AASIST on **ASVspoof 5** augmented with laundering attacks (9 degraded copies of a random 10% subset: noise, reverb, recompression, resampling, low-pass). It reports no non-augmented baseline.
- **Trachu et al. (Interspeech 2025)**: uses SE to extract the noise-plus-artifact residual and amplifies it. This shows that what enhancement removes carries detection-relevant signal.
- Full survey with 33 checked references: `docs/literature_survey.md` (see §9 there for design implications).
- **TL-SEJ (2024)** — already combines augmentation + enhancement, but via JOINT optimization, and notes that independent SE + anti-spoofing causes inconsistency artifacts.

### How we position this work: building on and improving prior work
We are not claiming the idea was never done: TL-SEJ already combined augmentation and enhancement by training them jointly. We build on that line of work and aim to **improve** robustness in the *independent* setup, where the enhancer and detector are not trained together. That's what real deployments actually use, since most people apply off-the-shelf enhancers. Our goal is to **publish the results** (DECISIONS.md C19), backed by a proper **literature survey** (task T18). In writing, frame our contribution as extending and improving existing work, and cite the prior work it builds on.

### The exact experiment design
Two versions of the detector, tested against three versions of audio:

| | Clean Test | Noisy Test | Enhanced Test |
|---|---|---|---|
| **Model A** (clean-trained, pretrained checkpoint) | ① reference | ② damage from attack | ③ Paper 2's fix |
| **Model B** (trained on noise-augmented data) | ④ sanity check | ⑤ Paper 1's fix | ⑥ **our new result** |

Each cell is one EER (Equal Error Rate) number; Model B's cells are mean ± std over 3 training seeds. With two enhancers, the Enhanced column is reported once per enhancer. Noisy and enhanced cells are the average over 0, 10 and 20 dB, and are also reported per SNR and per attack type (see Task Details).

### Tools and datasets already decided — do not re-litigate these
- **Detector:** AASIST (`github.com/clovaai/aasist`) — pretrained checkpoint already exists at `models/weights/AASIST.pth`, so Model A needs **no training**, only Model B does.
- **Main dataset:** ASVspoof 2019 LA — via Kaggle (`awsaf49/asvpoof-2019-dataset`), no registration needed.
- **Background noise:** MS-SNSD (`github.com/microsoft/MS-SNSD`), free, no registration.
- **Room echo/reverb:** RIRS_NOISES (`openslr.org/resources/28`), free, no registration.
- **Noise/reverb mixing:** use the `audiomentations` Python library (`AddBackgroundNoise`, `ApplyImpulseResponse`) — don't hand-write this.
- **Test-time cleanup:** pretrained enhancers only, never trained by us: SpeechBrain's `metricgan-plus-voicebank`, plus SEGAN+ if its weights can be obtained, otherwise SpeechBrain's `sepformer-wham16k-enhancement` (decision B11).
- **Scoring:** EER is computed with AASIST's bundled official scoring code (`evaluation.compute_eer`); never hand-write it.
- **Noise/reverb parameters to match the literature:** SNR levels 0–20 dB, RT60 reverberation of 0.3–0.9s.

### Why this specific design (in case anyone — human or AI — is tempted to change it)
Two real 2024–2026 papers each tried one fix in isolation (train-time augmentation OR test-time enhancement) on AASIST/ASVspoof. TL-SEJ combined them, but only with joint training. Our study extends this to the independent, off-the-shelf combination on the same benchmark — **don't quietly simplify the design down to just reproducing one paper**, and don't add scope beyond the 6-cell grid without discussing it with the team first, since the synopsis already submitted to college describes exactly this design.

### Deadline
Target: **15 November 2026**. Internal buffer target: **5 November 2026** — treat this as the real deadline.

### Where things live
```
scripts/      <- all real logic goes here as .py files (not inside notebooks)
notebooks/    <- thin notebooks that just call scripts/ — keeps Git merges clean
configs/      <- AASIST config files
results/      <- small result files: tables, plots, logs (never datasets/checkpoints)
docs/         <- synopsis, report, and other submission documents
```
Datasets and model checkpoints are **never** committed to Git — they go in each person's own Google Drive (see `.gitignore`). Large files (trained `.pth` checkpoints) that need to be shared between teammates should be shared via a link to the team's shared Google Drive folder, not through Git.

---

## Task Board

Status legend: `🟢 Ready` (unblocked, anyone can start) · `🟡 In Progress` · `🔒 Blocked` (waiting on a dependency) · `✅ Done`

Claim a task by editing the **Owner** column and status, committing just that change, and pushing. Do this before you start, so two people don't duplicate the same task.

**Everything runs in Google Colab** through the four notebooks in `notebooks/`. Each notebook cell just calls a script in `scripts/`.

| ID | Task | Notebook | Status | Owner | Depends on |
|----|------|----------|--------|-------|------------|
| T0 | Git & GitHub team setup | — | ✅ Done | Anup | none |
| T1 | Colab environment setup | 01 | ✅ Done | Anup | T0 |
| T2 | Download ASVspoof 2019 LA | 01 | ✅ Done | Anup | T1 |
| T3 | Download MS-SNSD (noise) | 01 | ✅ Done | Anup | T1 |
| T4 | Download RIRS_NOISES (reverb) | 01 | ✅ Done | Anup | T1 |
| T5 | Get AASIST code + pretrained checkpoint (Model A) | 01 | ✅ Done | Anup | T1 |
| T6a | Choose the fixed 10k test subset (once, then commit) | 01 | ✅ Done | Anup | T2 |
| T6 | Build the noisy test sets (0/10/20 dB) | 02 | 🟢 Ready | | T3, T4, T6a |
| T7 | Sanity-check Model A on the full clean test set | 04 | 🟢 Ready | | T2, T5 |
| T8 | Test Model A on noisy sets — cells ① ② | 04 | 🔒 Blocked | | T6, T7 |
| T9 | Build the noise-augmented training set | 02 | 🟢 Ready | | T2, T3, T4 |
| T10 | Train Model B (seed 42) — time the first epoch! | 03 | 🔒 Blocked | | T9 |
| T11 | Test Model B on clean + noisy — cells ④ ⑤ | 04 | 🔒 Blocked | | T10, T6 |
| T12 | Build the enhanced test sets (MetricGAN+, SepFormer; try SEGAN+) | 02 | 🔒 Blocked | | T6 |
| T13 | Test Models A & B on enhanced sets — cells ③ ⑥ | 04 | 🔒 Blocked | | T12, T10 |
| T14 | Reliability pass — train + test seeds 123 and 2024 | 03, 04 | 🔒 Blocked | | T11, T13 |
| T15 | Build the result tables | 04 | 🔒 Blocked | | T14 |
| T16 | Write the results & analysis section | — | 🔒 Blocked | | T15 |
| **T17** | **⭐ OPTIONAL: add SEMamba as a third enhancer (timeboxed)** | 02 | 🔒 Blocked | | T12, T13 |
| T18 | Literature survey (related work for the paper) | — | 🟡 In Progress (first draft in `docs/literature_survey.md`) | | none |

**Unblocking rule:** once every task in a "Depends on" list is `✅ Done`, flip that task to `🟢 Ready`. Whoever notices first should update it; don't wait for someone else.

**Run order (sequential, one Colab run at a time):**
T0 → T1–T5 → T6a → **T7** → T6 → T9 → T12 → T10 → T8/T11/T13 → T14 (seed 123, then 2024) → T15 → T16.
Different people can take turns, but only one person runs a notebook at any moment. When you finish a step, mark it ✅ on the board and tell the next person. T18 (literature survey) needs no Colab, so it can be done at any time.

### For teammates: how to take a turn (about 5 minutes, once)
Anup runs notebooks 01 and 02, because they create several GB of data, which counts against the Drive of whoever creates it. Teammates can take any later step (training, evaluation). To get access:
1. **Anup shares** the Drive folder `mini_project` with you as **Editor**.
2. **You add it to your Drive:** Shared with me → right-click `mini_project` → Organize → **Add shortcut** → My Drive. The shortcut must be named exactly `mini_project`, directly in My Drive.
3. **Open notebooks from Drive:** `mini_project/repo/notebooks/` → double-click → **Open with Google Colab**. Run the first cell as usual. "No GitHub secrets found" is fine; you don't need a token or `kaggle.json`.

Rules:
- Only Anup commits from Colab; teammates update this board from their own PC clone of the repo.
- Don't edit `scripts/config.py` in the shared folder without telling the team.

---

## Task Details

### The final experiment in one place (agreed plan; see DECISIONS.md B9–B12, E28–E30)
- **Models:** A = pretrained AASIST (no training). B = AASIST trained on clean + augmented data, 3 seeds (42, 123, 2024). C = clean control, trained **only if** epochs had to be cut below 100.
- **Training data:** a random 10% of training files × 3 copies (`_n` noise, `_r` echo, `_nr` both), MS-SNSD **train** noises, SNR 0–20 dB. That's +30% data.
- **Test data:**
  - clean = the full eval set (71,237 files);
  - noisy = a fixed 10k subset at **0, 10 and 20 dB**, MS-SNSD **test** noises, echo on 70% of clips;
  - enhanced = the noisy sets after MetricGAN+ and after the second enhancer (SEGAN+ if obtainable, otherwise SepFormer).
- **Model selection:** dev set only; the final model is `swa.pth`. The eval set is never used to choose a checkpoint.
- **Reported:**
  - main table (noisy/enhanced columns = average over the 3 SNRs);
  - per-SNR table;
  - per-attack table;
  - Model B values as mean ± std over its 3 seeds.

### Where things are stored (Google Drive: `MyDrive/mini_project/`)
```
repo/                       <- this Git repo (cloned in notebook 01)
asvspoof2019/  MS-SNSD/  RIRS_NOISES/  aasist/     <- downloads (T2-T5)
generated/eval/<condition>/flac/                   <- noisy_snr0 ... sepformer_snr20 (T6, T12)
generated/train_aug/flac/ + protocol_aug.txt       <- augmented copies (T9)
checkpoints/B_seed42/  (last_checkpoint.pth, best_dev.pth, swa.pth, log.txt)   <- T10/T14
scores/<model>/<condition>.txt                     <- T7-T14
```
Only `repo/` goes to GitHub. Tables land in `repo/results/`.

### T0 — Git & GitHub Team Setup
1. One person runs `git init` in the project folder, creates a **private** GitHub repo, pushes, and adds all teammates as collaborators.
2. Everyone clones it and pushes one test commit (e.g. claiming a task on this board).
**Done when:** everyone has pushed at least one commit. `.gitignore` already blocks datasets, audio, checkpoints and `kaggle.json`.

### T1–T5 — Setup (notebook `01_setup`)
Run notebook 01 top to bottom **once**, in the Drive account that will hold the shared data (it needs about 20 GB; see the note at the top of the notebook). It:
- mounts Drive and clones the repo;
- installs packages and checks for a GPU;
- downloads only the LA part of ASVspoof 2019, only the two MS-SNSD noise folders, and only the simulated RIRs;
- clones AASIST and applies a one-line fix for current NumPy (`scripts/setup_aasist.py`).

**Done when:** the "Final check" cell shows OK for everything.

### T6a — Fixed test subset (notebook 01, last section)
`scripts/make_eval_subset.py` picks 10,000 eval files with the same bona fide / attack mix as the full set, then commits `configs/eval_subset_10k.txt`. **Run once for the whole team**; never regenerate it after T6 has started.

### T6 — Noisy test sets (notebook 02)
`scripts/build_noisy_set.py --mode eval` (test with `--limit 20` first).
- Each file uses its own fixed random seed, so the same clip gets the same noise and echo at every SNR; only the loudness of the noise changes.
- Already-built files are skipped, so rerun after a disconnect.

**Done when:** `generated/eval/noisy_snr0`, `noisy_snr10` and `noisy_snr20` each hold 10,000 files. Listen to the sample cell once.

### T7 — Sanity check (notebook 04)
`scripts/run_eval.py --model A --conditions clean`. It prints the EER and compares it with the published 0.83%.
**Done when:** the result is within 0.5 points (**OK**). If it's **NOT OK**, stop and debug: nothing else is trustworthy until this passes.

### T8 — Model A on noisy sets (notebook 04)
`scripts/run_eval.py --model A` scores every condition that exists and skips the rest.
**Done when:** `scores/A/noisy_snr*.txt` and `clean_subset.txt` exist.

### T9 — Noise-augmented training set (notebook 02)
`scripts/build_noisy_set.py --mode train` (test with `--limit 20` first).
**Done when:** the count cell shows as many files as lines in `protocol_aug.txt` (about 7,600).

### T10 — Train Model B (notebook 03)
`scripts/train_model.py --tag B --seed 42`.
- Uses AASIST's own recipe, with batch size 16.
- Copies the training data to Colab's local disk first, for speed.
- Saves its progress every 10 minutes, even mid-epoch (`SAVE_EVERY_MINUTES`). **After a disconnect, rerun the notebook and it continues from the last save.**
- **Time the first epoch.** The log shows minutes per epoch and hours left. If 100 epochs × 3 seeds won't fit before the buffer deadline, lower `NUM_EPOCHS` in `scripts/config.py`, restart this seed, and plan to train model C with the same setting (decision E29).

**Done when:** `checkpoints/B_seed42/swa.pth` exists.

### T11 / T13 — Test Model B, and all enhanced sets (notebook 04)
Run the Model A and Model B cells again once the enhanced sets exist; already-scored conditions are skipped.
**Done when:** `scores/A/` and `scores/B_seed42/` contain every condition. **Cell ⑥ (Model B on enhanced audio) is the core result, so double-check it.**

### T12 — Enhanced test sets (notebook 02)
`scripts/build_enhanced_set.py --model metricgan`, then `--model sepformer`.
**SEGAN+ attempt (one session maximum):**
- try `github.com/santi-pdp/segan_pytorch`;
- email Paper 2's authors asking which SEGAN weights they used;
- if it works, add it to `ENHANCEMENT_MODELS` in `config.py` and write a loader. Otherwise SepFormer stays as the second enhancer (decision B11).

**Done when:** each `metricgan_snr*` and `sepformer_snr*` folder holds 10,000 files.

### T14 — Reliability pass (notebooks 03 + 04)
Train seeds 123 and 2024 (one per session is realistic), then rerun the Model B cell in notebook 04. If epochs were cut, also train and score model C (`--tag C --clean`) with the same epochs.
**Done when:** `scores/B_seed123/` and `scores/B_seed2024/` are complete.

### T15 — Result tables (notebook 04)
`scripts/summarize_results.py` writes `results/main_table`, `per_snr_table` and `per_attack_table` (`.md` + `.csv`) using AASIST's official EER code, and the notebook commits them.
**Done when:** all three tables are complete, with mean ± std for Model B.

### T16 — Write the Results & Analysis Section
**Depends on:** T15.
Interpret the tables:
- ① vs ② shows the attack damage;
- ② vs ⑤ tests augmented training;
- ② vs ③ tests enhancement (compare with Paper 2's SEGAN vs MetricGAN+ inversion);
- ⑤ vs ⑥ is our core question.

Then use the per-SNR table (where does enhancement help or hurt?) and the per-attack table (which attacks does noise hide best?) to explain *why*. Be honest about whichever outcome came out; all outcomes are valid findings (DECISIONS.md C20). Budget 5–10 hours for this analysis.
**Done when:** a draft results section exists in `docs/`.

### T17 — ⭐ OPTIONAL: Add SEMamba as a Third Enhancer
**Only attempt this if the core experiment (T1–T15) is on track and time genuinely permits. This is a bonus, never a blocker.**
**Depends on:** T12, T13 (the core enhancement pipeline must already work with SEGAN/MetricGAN+ first).

**Why it's worth doing:** Paper 2 found that *higher*-quality enhancement hurt detection more. SEMamba is higher quality still (PESQ 3.69, state of the art), so it should hurt detection even more. Confirming that turns a two-point observation into a three-point trend — a stronger, more general claim, and a second potential positive finding in case the core ⑤ vs ⑥ comparison comes back null.

**⏱️ HARD TIMEBOX — ONE SESSION (3 hours):**
Mamba requires custom CUDA kernels (`mamba-ssm`) that are known to be awkward to install on Colab. **If it is not running by the end of one session, stop and abandon this task.** Do not spend a second session on it. Note it as future work in the paper and move on — the core experiment does not need it.

**Steps:**
1. Try installing: `!pip install mamba-ssm causal-conv1d` (start the timer here).
2. Clone the official implementation: `github.com/RoyChao19477/SEMamba` — pretrained weights are provided, so **no training is required**.
3. Run SEMamba over the same noisy test sets from T6, writing `generated/eval/semamba_snr{0,10,20}/flac/` (then add `semamba` to `ENHANCEMENT_MODELS`).
4. Evaluate both Model A and Model B on it, exactly as in T13 — two more cells.
5. Add both numbers to the results table and check whether they fit the predicted quality-vs-detectability trend.

**Done when:** either the two extra cells are recorded, **or** the timebox expired and the attempt was abandoned and noted. Both outcomes close this task — abandoning it is a valid, planned result, not a failure.

### T18 — Literature Survey
**What:** A proper related-work survey for the paper (DECISIONS.md D26). Budget 10–15 hours. It can run alongside everything else, but should be well underway before T16.
**Steps:**
1. Read the four anchor papers in the Context section in full. (Paper 1's augmentation recipe is already checked: see DECISIONS.md B9.)
2. Search Google Scholar, IEEE Xplore, arXiv and the ISCA Archive (Interspeech / ASVspoof workshop proceedings). Use the search terms and themes listed in `docs/literature_survey.md`.
3. Follow each anchor paper's references (backward) and "cited by" list (forward).
4. Add each relevant paper to the table in `docs/literature_survey.md`. Only add papers you have actually opened; never add a citation from memory or from an AI summary without checking it.
5. Write a short summary paragraph per theme, ending with how our work builds on or improves that theme.
6. Keep a BibTeX file at `docs/references.bib`.
**Done when:** every theme has at least 4–5 verified papers with summaries, the gap statement at the end is written, and `references.bib` matches the table.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "CUDA out of memory" during training | Lower `batch_size` in the config (try 12 or 16 instead of 24) |
| Colab disconnects mid-training | Always checkpoint to Google Drive, not just local Colab disk |
| Audio won't load / format errors | Confirm 16kHz sample rate; resample with `librosa.load(path, sr=16000)` if not |
| Kaggle download fails | Re-check `kaggle.json` is correctly placed in `~/.kaggle/` |
| Two people edited the Task Board table and it conflicts in Git | Pull, manually merge the table rows (keep both people's status changes), commit |
| `mamba-ssm` won't install / CUDA kernel build fails (T17) | This is the expected failure mode — abandon T17 when the timebox expires. Do not debug past one session. |
