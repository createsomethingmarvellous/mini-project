# Project Context for Claude Code

## What this project is

Testing whether an audio deepfake detector (AASIST) can be defeated by adding
realistic background noise/reverb onto cloned speech, and comparing two known
defences: train-time noise augmentation and test-time speech enhancement.

**Read `GUIDE.md` for the full context and task board. Read `DECISIONS.md`
for decisions.** Do not re-derive the plan - it is already settled there.

## Critical framing rule

**We build on and improve prior work; we do not claim it was never done.**
Prior work (TL-SEJ, 2024) already combines both defences via joint
optimization. We extend it to the *independent, non-jointly-optimized* case,
aiming to improve robustness and publish the results with a full literature
survey (task T18, `docs/literature_survey.md`). Core question:

> Does noise-augmented training make a detector less dependent on the artifacts
> that independent test-time enhancement destroys?

Never write "first", "novel", or "nobody has done this". If you draft text for
the paper or report, frame it as building on prior work, not competing with it.

## The experiment (details: GUIDE.md "The final experiment in one place")

|                                        | Clean | Noisy | Enhanced (x2 enhancers) |
|----------------------------------------|-------|-------|-------------------------|
| Model A (pretrained AASIST.pth)        | (1)   | (2)   | (3)                     |
| Model B (noise-augmented, 3 seeds)     | (4)   | (5)   | (6)                     |

- Training data: 10% of train files x 3 copies (noise / echo / both), MS-SNSD train noises.
- Clean test = full eval set; noisy/enhanced = fixed 10k subset at 0/10/20 dB.
- Enhancers: MetricGAN+, plus SEGAN+ if obtainable, else SepFormer.
- Model C (clean control) only if epochs are cut below 100.
- Metric is EER (lower = better). Cell (6) vs (5) is the core question.

## Key facts that are already decided

- **Model A needs NO training** - pretrained `AASIST.pth` from clovaai/aasist.
- **Model B is the only real training job.** `scripts/train_model.py` follows
  AASIST's recipe but selects models on the DEV set only (never eval) and
  resumes after disconnects. Final model = `swa.pth`.
- **Never train the speech enhancers** - pretrained weights only.
- **Never hand-write EER scoring** - use AASIST's `evaluation.compute_eer`.
- **Never hand-write noise mixing** - use `audiomentations`.
- All parameters live in `scripts/config.py`.

## Repo layout

```
scripts/     config.py, common.py, aasist_bridge.py + one script per task
notebooks/   01_setup, 02_build_data, 03_train, 04_evaluate (thin Colab cells)
configs/     eval_subset_10k.txt (made once in T6a, then committed)
results/     summary tables written by summarize_results.py
docs/        literature survey, synopsis, report
```

## Hard constraints

- **Everything runs in Google Colab** via the notebooks. Do not install
  packages or run training/data jobs on the local PC; local checks are
  limited to syntax checks (`python -m py_compile`).
- **Datasets, generated audio, checkpoints and raw scores are NEVER committed.**
  They live on Google Drive under `MyDrive/mini_project/`. `.gitignore` covers this.
- **`kaggle.json` must never be committed.**
- Batch size 16. Fallback ladder on OOM: 16 -> 12 -> AASIST-L.
- Audio is 16 kHz FLAC, in AASIST's `<dir>/flac/<utt_id>.flac` layout.

## Working style for this repo

- Team project, 3-4 people. Tasks are claimed on the board in `GUIDE.md`.
- Test any builder with `--limit 20` before running on thousands of files.
- Builders skip finished files and training resumes, so reruns are safe.

## Known unknowns - do not invent answers to these

- The ASVspoof folder is found automatically (`common.la_root()`), but the
  Kaggle mirror's layout has not been seen yet.
- `NUM_EPOCHS` stays 100 until the first epoch is timed (decision E29).
- Whether SEGAN+ weights can be obtained (decision B11).
- Nothing in this repo has been executed against real data yet.
