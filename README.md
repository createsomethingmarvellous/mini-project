# Robustness of Audio Deepfake Detection Against Background-Noise Masking

**5th Semester B.E. Mini Project — BGS College of Engineering and Technology (BGSCET)**

## What this project is about

AI voice cloning is increasingly used in impersonation fraud. Detectors that catch cloned/fake voices can be fooled when attackers glue realistic background noise or room echo back onto the fake audio. This project:

1. Measures how much that trick actually breaks a real, published detector (AASIST).
2. Tests two known fixes — training on noisy examples, and cleaning up noisy audio before testing — separately.
3. Tests **combining both fixes together**, using off-the-shelf tools that are not tuned to each other. Earlier work (TL-SEJ, 2024) combined the two by training them jointly. We build on that and ask how the combination behaves in the simpler setup most real systems use, aiming to improve robustness and publish the results.
4. Includes a **literature survey** of the anti-spoofing, augmentation and speech-enhancement research this work builds on (task T18 in `GUIDE.md`).

Full background, the research papers this builds on, and the complete task breakdown live in **[`GUIDE.md`](./GUIDE.md)** — read that before doing anything else.

## Team

- Anup N Raykar — 1MP24AI010
- Amith Anadh Gowndker — 1MP24AI009
- Kushaal K Gowda — 1MP24AI028
- Guru Charan Gowda — 1MP24AI017

## Quick Start — for whoever is free and picking up work

1. Open **[`GUIDE.md`](./GUIDE.md)** and read the **Context** section once — it's written so you (or an AI helping you) only need to read it a single time, and it applies to every task after that.
2. Look at the **Task Board** in `GUIDE.md`. Find a task marked `🟢 Ready` with no name in the "Owner" column.
3. Claim it: edit the table, add your name, set status to `🟡 In Progress`, commit and push (see the Git workflow below).
4. Open that task's detail section further down in `GUIDE.md` and follow its steps.
5. **Using an AI assistant (Claude, ChatGPT, etc.)?** Paste the "Context" section of `GUIDE.md` plus your specific task's detail section into your chat. That's everything it needs — you don't need to re-explain the project.
6. When finished, check the task's "Done when" checklist, mark it `✅ Done` on the board, push your branch, and open a Pull Request for someone else to glance over before merging.

## Repo Structure

```
.
├── README.md          <- you are here
├── GUIDE.md          <- full context + task board + step-by-step detail for every task
├── DECISIONS.md      <- decisions log (the adopted experiment plan)
├── notebooks/        <- 01_setup, 02_build_data, 03_train, 04_evaluate (run in Colab)
├── scripts/          <- all real logic; config.py holds every parameter
├── configs/          <- eval_subset_10k.txt (the fixed test subset)
├── results/          <- result tables (small files only — see .gitignore)
└── docs/             <- literature survey, synopsis, report
```

**Everything runs in Google Colab.** Data, generated audio and checkpoints live on Google Drive (`MyDrive/mini_project/`), never in Git. See `START_HERE.txt` for the run order.

## Git Workflow (short version — full detail in `GUIDE.md`)

```bash
git pull origin main
git checkout -b <task-id>-<short-description>   # e.g. T6-noisy-test-set
# ...do the work...
git add .
git commit -m "data: build noisy test set (T6)"
git push origin <task-id>-<short-description>
# then open a Pull Request on GitHub to merge into main
```

## Deadline

Target submission: **15 November 2026**
Internal buffer target (aim to functionally finish by): **5 November 2026**
