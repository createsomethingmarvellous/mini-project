# Open Decisions Log

Every open question the team needs to actually settle, with a recommendation for each. Recommendations are defaults to save time — override any of them with a quick team discussion if someone disagrees, and record the final call in the "Decided" column.

**How to use this:** as each decision gets made, fill in the "Decided" column (who/what was chosen, and the date) so this becomes a record, not just a wishlist. Don't delete resolved rows — a decided decision is still useful history if a question resurfaces later.

---

## A. Team & Process Decisions

### A1. Task-claiming mechanism
**Options:** keep the Markdown table in `GUIDE.md`, or move to GitHub Issues/Projects.
**Recommendation: keep the Markdown table for now.** Switching tools adds setup overhead that isn't worth it for a 3–4 person team over ~2 months. Only move to Issues/Projects if table-merge conflicts actually start happening repeatedly — solve the problem when it's real, not preemptively.
**Decided:** _____

### A2. Check-in cadence
**Options:** no fixed cadence ("whoever is free"), a lightweight async check-in every 2–3 days, or a scheduled weekly call.
**Recommendation: a short async update every 2–3 days** (a one-line message: what you did, what's blocked), plus **one longer sync per week** to re-plan. This matches the existing "alternate day" work rhythm without adding meeting overhead.
**Decided:** _____

### A3. Pull Request review rule
**Options:** self-merge everything, or require review for some/all tasks.
**Recommendation: self-merge for setup/data-prep tasks (T0–T6, T9, T12)**, since mistakes there are cheap to redo. **Require at least one other teammate to skim the diff before merging anything touching T10 (training) or T14–T16 (final results)** — mistakes there are expensive to redo.
**Decided:** _____

### A4. Shared Drive folder ownership and naming
**Options:** ad hoc sharing, or a defined owner + naming convention.
**Recommendation:** whoever creates the GitHub repo also creates the Drive folder. Name checkpoint files by task and seed, e.g. `T10_ModelB_seed42.pth` — self-documenting names avoid needing a separate log file.
**Decided:** _____

### A5. Repo visibility
**Options:** public or private GitHub repo.
**Recommendation: private during development.** Avoids any plagiarism-policy concerns before submission. Can always be made public afterward if the team wants to showcase it — the reverse (un-exposing a public repo) isn't really possible.
**Decided:** _____

### A6. License (if ever made public)
**Options:** none, or an open license like MIT.
**Recommendation:** doesn't matter while private — decide this only if/when you make the repo public. Default to MIT at that point if you want others to be able to reference it.
**Decided:** _____

### A7. Who's actually doing hands-on work
**Options:** assume all 4 listed names are equally active, or explicitly clarify.
**Recommendation:** explicitly state in the README who's doing hands-on technical tasks vs. who's contributing to documentation/report work. The task board shouldn't silently imply obligations on someone who said they're doing a lighter role.
**Decided:** _____

### A8. Exam blackout weeks
**Options:** leave it vague, or nail down actual dates.
**Recommendation: do this immediately — it's a 10-minute task with zero dependencies**, and every timeline estimate made so far assumes it gets done. This is the single easiest decision on this list to close out; there's no reason it should still be open.
**Decided:** _____

---

## B. Technical / Methodology Decisions

### B9. Noise-augmentation percentage for training (T9) — ⚠️ CHANGED by C19
**Options:** an exact match to Paper 1's number (not fully confirmed), or a stated approximation.
**Recommendation (revised for publication): go and verify Paper 1's actual figure before starting T9.** For coursework, "approximately 10%, following Paper 1" would pass. For a paper, a reviewer can check the source and will notice if your stated reproduction doesn't match — and an unverified claim about someone else's method is the kind of thing that gets a submission desk-rejected or demanded in revision. This is now a required 30-minute reading task, not an optional nicety.
**Finding (16 Sep 2026, from reading arXiv:2410.01108):** Paper 1 randomly selected 10% of the **ASVspoof 5** train set and made **9 augmented copies** of it: 1 reverb (RT60 0.3/0.6/0.9 s), 5 additive noises (babble, Volvo, white, cafe, street) at SNR 0/10/20 dB, 1 MP3 recompression, 1 resampling, and 1 low-pass filter. So the augmented data was about 90% of the original training size, not 10%. Our current T9 (one noisy copy of 10%) is lighter. Options: match the ratio with multiple noise/reverb copies, or state it as a lighter recipe "inspired by" Paper 1. See `docs/literature_survey.md` §9.
**Decided (adopted 16 Sep 2026):** 10% of training files × **3 copies** (noise only / echo only / echo + noise), about +30% data. Train noises only (MS-SNSD `noise_train`); SNR 0–20 dB. In the paper: "a lighter recipe inspired by Ali et al.". Implemented in `scripts/config.py` (`AUGMENT_FRACTION`, `AUG_COPIES`).

### B10. SNR/RT60 sampling strategy (T6, T9)
**Options:** fixed discrete steps (matching the papers' exact values), or randomized continuous sampling within the range.
**Recommendation: randomized continuous sampling** (0–20 dB SNR, 0.3–0.9s RT60) — this is `audiomentations`' natural behavior (just set min/max), requires no extra bucketing code, and arguably simulates realistic variability better than fixed steps.
**Decided (adopted 16 Sep 2026):** training uses random continuous SNR (0–20 dB). Testing uses **fixed levels 0, 10 and 20 dB** (the same levels as Paper 1), so results can be reported per level. Echo: RIRS `simulated_rirs`, applied to 70% of test clips and to the `_r`/`_nr` training copies.

### B11. Which enhancement model to actually use (T12) — ⚠️ CHANGED by C19
**Options:** MetricGAN+ only, SEGAN only, or both.
**Recommendation (revised for publication): run both SEGAN and MetricGAN+ — this is now required, not optional.** Paper 2's counterintuitive finding (the higher-quality enhancer hurts detection more) is central to the question you're extending; a paper that tests only one enhancer invites the obvious reviewer question "what about the other one?" Running both also gives you a 9-cell grid instead of 6, which is a meaningfully stronger contribution for roughly one extra evaluation pass (no extra training). **A third enhancer (SEMamba) is an optional extension — see B18 and task T17.**
**Decided (adopted 16 Sep 2026):** MetricGAN+ (required). The second enhancer is SEGAN+ from `santi-pdp/segan_pytorch` **if** it works within one session (in parallel, email Paper 2's authors for their SEGAN weights); otherwise SpeechBrain SepFormer (`sepformer-wham16k-enhancement`), with the reason stated in the paper. No public SEGAN exists on Hugging Face/SpeechBrain.

### B12. Training batch size fallback (T10)
**Options:** try batch size 24 first (as AASIST's default) and reduce only on error, or start conservatively.
**Recommendation: start at batch size 16, not 24.** AASIST's own documentation says 24 needs ~16GB — exactly free Colab's T4 ceiling, leaving no margin for anything else running. Starting lower avoids repeated crash-and-retry cycles that waste real session time.
**Decided (adopted 16 Sep 2026):** batch size 16.

### B13. AASIST vs. AASIST-L switch trigger
**Options:** vague "if needed," or a defined fallback ladder.
**Recommendation: a fixed ladder — batch 24 → 16 → 12 → switch to AASIST-L.** Removes any mid-project guessing about when it's "bad enough" to switch.
**Decided (adopted 16 Sep 2026):** ladder 16 → 12 → AASIST-L (we start at 16, per B12).

### B14. Which EER/t-DCF scoring implementation to use
**Options:** the official ASVspoof organizer toolkit, or a separate public repo's implementation.
**Recommendation:** use whatever scoring code is already bundled with the AASIST repo itself, since it's guaranteed to match AASIST's output format. Only fall back to a separately-sourced script if AASIST's own doesn't include one.
**Decided (adopted 16 Sep 2026):** AASIST's bundled `evaluation.py` (`compute_eer`), with its `np.float` line fixed for current NumPy by `scripts/setup_aasist.py`.

### B15. Random seed values (T14)
**Options:** whatever seed happens to be used per run, or fixed values decided in advance.
**Recommendation: fix three specific seeds now — e.g. 42, 123, 2024** — and write them into the config before the first training run. Needed for anyone (including you, later) to reproduce the results.
**Decided (adopted 16 Sep 2026):** training seeds 42, 123, 2024. Data-building seed fixed at 42 (`DATA_SEED`), so all three runs see identical augmented data.

### B16. Numeric tolerance for the T7 sanity check
**Options:** a vague "close to published," or a specific number.
**Recommendation: within 0.5 percentage points of EER** counts as validated. If it's off by several points, something in the pipeline is broken and needs debugging before moving on — don't proceed on a shaky foundation.
**Decided (adopted 16 Sep 2026):** within 0.5 percentage points of the published 0.83% EER (checked automatically by `run_eval.py`).

### B17. Whether to pursue ASVspoof 2021 at all — ⚠️ CHANGED by C19
**Options:** include it as a second test set, or drop it.
**Recommendation (revised for publication): keep it as a stretch goal, and start the registration now.** Single-dataset results are a standard reviewer criticism ("does this generalize, or is it an artifact of ASVspoof 2019?"). Registration takes ~48 hours of waiting, so submitting the form costs you nothing today and preserves the option. **But do not let it block the core 6/9-cell experiment** — if the timeline tightens, drop it and state the single-dataset scope as an explicit limitation in the paper. An honest limitation is far better than a rushed, half-run second dataset.
**Decided:** _____

### B18. Paper 2's stated future-work items (ASVspoof 5, Mamba-based enhancement) — ✅ DECIDED (split)
**DECIDED: split the two items — SEMamba is an optional extension, ASVspoof 5 stays out of scope.**

**SEMamba (Mamba-based enhancement) → OPTIONAL, timeboxed (see task T17).** It slots into the existing grid as just one more enhancement model — same detector, same datasets, same Model B, no extra training, since pretrained weights are public. Paper 2's own logic makes a clean prediction here: since they found *higher* enhancement quality hurt detection more, and SEMamba is higher quality still (PESQ 3.69 vs. the others), it should hurt detection even more. Confirming that turns a two-point oddity into a three-point trend — a stronger, more general claim, and a second potential positive finding that hedges against the combination question (⑤ vs ⑥) coming back null.
**Hard condition: timebox the `mamba-ssm` installation to ONE session (3 hours).** Mamba needs custom CUDA kernels that are known to be awkward on Colab. If it isn't running by the end of that session, abandon it immediately and proceed with SEGAN + MetricGAN+ only. The installation risk is the *only* reason this was originally droppable — the timebox neutralises it.

**ASVspoof 5 → still out of scope.** Unlike SEMamba (one extra inference pass), a new dataset means a fresh download, re-running every cell, and a substantial timeline hit. List it as future work in the paper.
**Decided:** SEMamba optional + timeboxed; ASVspoof 5 out — recorded _____ (date)

---

## C. Scope & Outcome Decisions

### C19. Internal submission only, or attempt external publication? ✅ DECIDED
**DECIDED: Yes — the team is aiming for external publication.**
This raises the bar on several other decisions below (B9, B11, B15, B16, B17 are all affected), and adds new decisions C22–C26. Note the two goals have **different deadlines**: the college submission is fixed at 15 Nov 2026, while a paper submission depends entirely on the target venue's own CFP deadline — which may fall well before *or* after November. Treat them as two separate tracks sharing the same experiments.
**Decided:** Aiming for publication — recorded _____ (date)

### C20. Agreeing in advance that a null result is acceptable
**Options:** treat any outcome as valid, or leave this unstated and risk pressure to reshape the experiment later.
**Recommendation: write this agreement down now, before anyone sees real numbers.** If cell ⑤ and cell ⑥ turn out statistically similar (combining both fixes doesn't beat training-time augmentation alone), that is reported honestly as a legitimate finding — not treated as a failure requiring a redo. Any new idea sparked by the results becomes a "future work" note, not a silent, unplanned change to the dataset/model/hyperparameters after the fact. This is a scientific-integrity guardrail, and it only works if it's agreed to *before* the results exist, when nobody yet has a stake in a particular outcome.
**Decided:** _____

### C21. Final deliverable format
**Options:** just the 2-page synopsis, or a fuller results write-up too.
**Recommendation:** this genuinely depends on your specific department/guide's requirements, which I can't determine for you — **check with your project guide directly.** As a working default, plan for both: the synopsis (already done) plus a fuller write-up appended to `docs/` before submission. Assign one person to final assembly/formatting in the last week, ideally someone other than whoever did the technical work, so a fresh pair of eyes catches inconsistencies.
**Decided:** _____

---

## D. Publication-Specific Decisions (new — triggered by C19)

### D22. Target venue
**Options:** an IEEE student/regional conference, a national conference, a workshop, or a journal.
**Recommendation: target a student-friendly IEEE conference or workshop, not a top-tier venue or journal.** A single-detector, single-dataset reproduction-plus-extension study is a realistic fit for a student conference track; it is not competitive at a major speech venue (Interspeech/ICASSP-tier), where reviewers expect multiple architectures and datasets. **Pick the venue early**, because its CFP deadline and page limit drive your entire schedule — you cannot plan backwards from an unknown date. Note: some conferences charge registration/publication fees, so check cost before committing.
**Decided:** _____

### D23. Authorship order and faculty co-author
**Options:** decide now, or leave it until the paper is written.
**Recommendation: decide now, before results exist** — the same logic as C20. Order should reflect actual contribution, and **talk to your project guide early about whether they want to be a co-author**; many institutions expect or require faculty involvement for a student paper, and they can also advise on venue choice. Leaving this until the end is how teams end up in awkward disputes.
**Decided:** _____

### D24. Statistical significance testing
**Options:** report raw averaged EER numbers, or add significance testing.
**Recommendation: add it — this is close to mandatory for publication.** The core claim in cell ⑤ vs. ⑥ is a comparison between two conditions; with only 3 seeds, a small numeric difference could easily be noise. Report mean ± standard deviation at minimum, and ideally a simple significance test. **This is exactly what turns "our number was slightly lower" into a defensible claim** — and without it, a reviewer can dismiss your central finding outright.
**Decided:** _____

### D25. Reproducibility package
**Options:** keep the repo private permanently, or release code on publication.
**Recommendation: stay private during development (per A5), then make the repo public at submission or acceptance.** Reviewers increasingly expect a code link, and since your whole project is itself built on other people's public code, releasing yours is both consistent and good practice. Clean up the repo (clear README, exact dependency versions, seed values documented) before making it public.
**Decided:** _____

### D26. Literature review depth
**Options:** cite just the two source papers, or do a proper related-work survey.
**Recommendation: a proper related-work section is now required.** Coursework tolerates citing two papers; a submission needs to show you know the field — meaning a genuine search covering the broader anti-spoofing literature, other detectors besides AASIST, and other augmentation/enhancement approaches. **Budget real time for this (10–15 hours)** — it's a substantial task that didn't exist in the coursework-only plan, and it's often what separates an accepted student paper from a rejected one.
**Decided:** _____

### D27. Ethics / responsible disclosure statement
**Options:** omit it, or include a short statement.
**Recommendation: include a brief statement.** Your paper describes, in effect, how to make voice-clone attacks harder to detect. That's normal and accepted in security research (you can't build defences without studying attacks), but venues increasingly expect an explicit sentence or two noting that the work is defensive in intent, uses only public benchmark data, and generates no new attack tooling. Cheap to write, and its absence is a easy thing for a reviewer to flag.
**Decided:** _____

---

## E. Experiment Design Details (added 16 Sep 2026, adopted)

### E28. Test-set design
**Decided (adopted 16 Sep 2026):**
- Clean test = the **full** eval set (71,237 files), for comparison with the published baseline.
- Noisy and enhanced tests = a **fixed 10k subset** with the same bona fide / attack mix (`configs/eval_subset_10k.txt`, made once and committed) at 0, 10 and 20 dB.
- The main-table value for a noisy or enhanced column is the average over the three levels.

**Why:** running 5 levels × 2 enhancers on all 71k files is not feasible on free Colab. A fixed subset keeps every comparison like-for-like.

### E29. Epoch budget and fair comparison
**Decided (adopted 16 Sep 2026):** AASIST's default is 100 epochs. **Time the first epoch** of Model B.
- If 100 epochs × 3 seeds fits the schedule, train Model B for 100 epochs, the same as Model A.
- If not, lower `NUM_EPOCHS` **and** train a clean control model C with the same epoch count, so ② vs ⑤ isn't confounded by training length.

**Decided value:** _____ epochs (fill in after timing).

### E30. Model selection must not use the test set
**Decided (adopted 16 Sep 2026):** AASIST's `main.py` saves `best.pth` based on **eval-set** results, which leaks test data. Our `scripts/train_model.py` selects on the **dev set only**, and the reported model is the SWA average (`swa.pth`), following AASIST's own recipe otherwise.

### E31. Result tables
**Decided (adopted 16 Sep 2026):** report three tables:
- main;
- per SNR;
- per attack type (A07–A19).

Model B values are given as mean ± std over its 3 seeds. Audio-quality scores (PESQ) and score-distribution plots are optional extras if time allows.

---

## Summary: what changed because of the publication decision

| Decision | Was | Now |
|---|---|---|
| B9 (augmentation %) | Approximate, note it | **Verify Paper 1's exact figure first** |
| B11 (enhancement model) | SEGAN, MetricGAN+ optional | **Both required — 9-cell grid** |
| B18 (Paper 2 future work) | Both dropped | **SEMamba = optional, timeboxed (T17); ASVspoof 5 still out** |
| B17 (ASVspoof 2021) | Dropped | **Stretch goal — register now, don't let it block** |
| C19 | Undecided | **Publishing — two separate deadline tracks** |
| — | — | **New: D22–D27 (venue, authorship, statistics, code release, literature review, ethics)** |

**Honest timeline warning:** the additions above (a full literature review, a second enhancement model, significance testing, and possibly a second dataset) add roughly **25–40 hours** on top of the original estimate, which already sat near the top of your available budget. The college deadline of 15 Nov is still achievable for the *experiments*, but the *paper* will very likely need to be written after that — which is fine, as long as the team plans for it rather than being surprised by it. **The single most useful thing you can do right now is pick the target venue (D22)**, because its deadline determines whether this timeline is comfortable or impossible.
