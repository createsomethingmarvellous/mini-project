# Literature Survey: Noise Robustness of Audio Deepfake Detection

**Project:** Robustness of Audio Deepfake Detection Against Background-Noise Masking (task T18)
**Status:** First draft, compiled 16 Sep 2026.

**How this was checked:** each paper below was checked against a primary record: arXiv metadata, ISCA Archive, IEEE/ACM listing, J-STAGE, or the publisher page. Where only an arXiv version was checked, the venue is marked "(arXiv)". Before submitting the paper, each entry still needs a final pass against the publisher version (page numbers, volume), and the team should read the full text of every paper it cites.

---

## 1. Introduction

Text-to-speech (TTS) and voice-conversion (VC) systems now produce convincing cloned voices. Spoofing countermeasures (CMs) such as AASIST [7] detect them with low error on the ASVspoof 2019 LA benchmark [1, 2]. However, that benchmark is recorded in clean conditions. A growing body of work shows that CM performance drops sharply when additive noise or reverberation is present [12, 13, 14, 16], which an attacker can exploit by "laundering" a cloned voice through background noise or room acoustics.

Two families of defence have been studied:
1. **Train-time data augmentation**: expose the CM to degraded audio during training [9, 14, 17–23].
2. **Test-time speech enhancement (SE)**: clean the input before it reaches the CM [28–31].

This survey organises the literature into six themes and ends with the research gap our project addresses.

---

## 2. Theme A: Spoofing countermeasure architectures

- **RawNet2** [8] (Tak et al., ICASSP 2021) showed that end-to-end models working directly on raw waveforms are competitive for logical-access (LA) spoofing detection.
- **AASIST** [7] (Jung et al., ICASSP 2022) uses graph attention networks with a heterogeneous stacking layer to model spectral and temporal artifacts jointly, and gives state-of-the-art results on ASVspoof 2019 LA. Its public code and pretrained checkpoint make it the standard baseline, and it is the detector used in our study and in both anchor papers [20, 31].
- **Self-supervised (SSL) front-ends**: Tak et al. [9] (Odyssey 2022) paired a wav2vec 2.0 front-end with an AASIST back-end and data augmentation, greatly improving generalisation. Schäfer et al. [21] found that heavy augmentation helps non-SSL front-ends but can *hurt* SSL-based ones. This is a relevant caveat, since our Model B uses the non-SSL AASIST.
- **Surveys**: Yi et al. [10] and Pham et al. [11] give broad overviews of datasets, features, classifiers and open problems. Both name robustness to real-world conditions as unresolved.

**Relevance to us:** AASIST is a well-motivated, widely reproduced choice. Testing only one (non-SSL) architecture is a limitation we should state; SSL front-ends are future work.

---

## 3. Theme B: Benchmarks and evaluation

- **ASVspoof 2019** [1, 2] is the primary LA benchmark (TTS/VC attacks, clean recordings) and our main dataset. Evaluation uses EER and t-DCF.
- **ASVspoof 2021** [3] added channel and codec variability (LA) and a deepfake (DF) track, moving "towards the wild". This is our stretch-goal second dataset (decision B17).
- **ASVspoof 5** [4] (2024) uses crowdsourced data from diverse acoustic conditions, codecs and adversarial attacks. It is out of our scope (B18), but is the dataset Paper 1 [20] used.
- **ADD 2022** [5] included a low-quality track with noisy/degraded fake audio.
- **In-the-Wild** [6] (Müller et al., Interspeech 2022) showed that models strong on ASVspoof degrade badly on real-world deepfakes, which motivates robustness research.

**Relevance to us:** ASVspoof 2019 LA allows direct comparison with Paper 2 [31] and with the published AASIST baseline. Single-dataset scope is a known limitation.

---

## 4. Theme C: Robustness of CMs to noise and laundering

- **Hanilçi et al.** [12] (Speech Communication, 2016) were among the first to analyse synthetic-speech detectors under additive noise, showing large degradation and a strong dependence on front-end features.
- **Chen et al.** [14] (Odyssey 2020) evaluated on a noisy version of ASVspoof 2019 LA built from public noises. They used large-margin cosine loss, frequency masking and noise/reverb augmentation to improve generalisation.
- **Ali et al.** [13] (ACM IH&MMSec 2024) built an *ASVspoof Laundering Database* (~1,388 hours derived from the ASVspoof 2019 eval set) and tested seven state-of-the-art CMs. These systems perform poorly under aggressive laundering, *especially reverberation and additive noise*. This is the clearest statement of the threat our project studies.
- **Fan et al.** [15] (arXiv 2023) proposed dual-branch knowledge distillation for noise-robust synthetic speech detection.
- **Sen et al.** [16] (arXiv 2025) combined a survey with SNR-controlled benchmarks (35 dB down to −5 dB, MS-SNSD noise, ASVspoof 2021 DF) for SSL backbones (WavLM, wav2vec 2.0, MMS). They report that multi-condition fine-tuning reduces EER by 10–15 percentage points at 10–0 dB SNR. They do **not** evaluate SE preprocessing.

**Relevance to us:** the degradation (our cell ②) is well established; our contribution is not showing that noise hurts, but analysing how the two defences interact.

---

## 5. Theme D: Train-time data augmentation

- **Signal companding** [17] (Das et al., ICASSP 2021): a-law/µ-law companding as augmentation, on ASVspoof 2019 LA.
- **RawBoost** [18] (Tak et al., ICASSP 2022): raw-waveform augmentation simulating linear/non-linear convolutive noise and impulsive/stationary additive noise. It is now a standard recipe.
- **Cohen et al.** [19] (Speech Communication, 2022): a systematic study of augmentation for channel variability, compression and bandwidth on ASVspoof 2021 LA/DF.
- **Ali et al., "Augmentation through Laundering Attacks"** [20] (ASVspoof 2024 Workshop), **our Paper 1**: randomly selected **10% of the ASVspoof 5 training set** (18,235 files) and created **9 degraded copies of that subset**:
  - reverberation (RT60 ∈ {0.3, 0.6, 0.9} s);
  - five additive noises (babble, Volvo, white, cafe, street) at SNR ∈ {0, 10, 20} dB, one copy per noise type;
  - MP3 recompression;
  - resampling;
  - 8 kHz low-pass filtering.

  This gave 164,115 augmented files, added to the original set (327,461 files total), which was used to train AASIST. They report pooled EER 25.32% / minDCF 0.662 on ASVspoof 5 eval, but **no non-augmented baseline** in the same paper.
- **Schäfer et al.** [21] (ASVspoof 2024): augmentation helps AASIST-style models but can hurt SSL front-ends.
- **Frequency Feature Masking** [22] (Han et al., 2025): masking frequency bands during training; reported large gains on ADD 2022 and ASVspoof 2019 LA.
- **Truong et al.** [23] (ICASSP 2026): gradients from original and augmented inputs conflict in ~25% of iterations with RawBoost. Their dual-path gradient-alignment training improves In-the-Wild EER by 18.69% (relative).

**Relevance to us:** augmentation is an established defence. Paper 1 is the closest recipe, but see §9 for how it differs from our current T9 plan.

---

## 6. Theme E: Speech enhancement models

- **SEGAN** [24] (Pascual et al., Interspeech 2017): the first GAN-based end-to-end waveform enhancer.
- **MetricGAN+** [25] (Fu et al., Interspeech 2021): trains the generator to directly optimise perceptual metrics (PESQ). A pretrained VoiceBank model is released through **SpeechBrain** [27].
- **SEMamba** [26] (Chao et al., IEEE SLT 2024): Mamba (state-space) based SE with state-of-the-art quality, and the candidate for our optional task T17.

---

## 7. Theme F: Speech enhancement combined with spoofing detection

- **Wang et al.** [28] (Interspeech 2023): a U-Net SE front-end *jointly trained* with an anti-spoofing back-end. They found joint training more effective than a cascaded (independent) pipeline.
- **TL-SEJ** [29] (Wang et al., arXiv 2024; extended as "SECM-Joint" in IEICE Trans. Inf. & Syst., 2025) adds:
  - on-the-fly data augmentation;
  - an ASR-pretrained Conformer back-end;
  - joint optimisation with the SE front-end.

  They argue that *independent* SE + CM creates unknown extra artifacts because the two tasks are inconsistent. Reported gains: 2.7–15.8% accuracy over baseline in noise, and 0.7–5.8% over data augmentation alone.
- **Trachu et al.** [30] (Interspeech 2025) reverse the usual use of SE. They add noise, use SE to *extract* the noise-plus-artifact residual, amplify it and re-insert it. Spoofed speech shows larger artifact magnitude, giving relative gains of up to 44.44% (ASVspoof 2019) and 26.34% (ASVspoof 2021). This implies that **the part SE removes carries detection-relevant information**.
- **Anacin et al.** [31] (arXiv, March 2026), **our Paper 2**:
  - **Setup:** ASVspoof 2019 LA test set corrupted with **babble and cafeteria** noise at **0, 5, 10, 15, 20 dB** (additive only, **no reverberation**); training data untouched; SEGAN and SpeechBrain MetricGAN+ as test-time enhancers.
  - **Main finding:** MetricGAN+ gives the highest PESQ/SRMR but the **worst** EER, while SEGAN has the lowest quality scores but the **best** EER. For example, with cafeteria noise at 0 dB, EER was 42.58% noisy, 14.03% after SEGAN and 40.40% after MetricGAN+.
  - **Detail:** at **20 dB**, both enhancers *increase* EER over the unenhanced noisy audio (cafeteria: 2.80 → 5.89 / 5.47).
  - **Future work named:** ASVspoof 5 and Mamba-based SE.

---

## 8. Research gap and our contribution

| Work | Train-time augmentation | Test-time SE | SE and CM trained together? | Dataset |
|---|---|---|---|---|
| Ali et al. [20] | ✔ | ✘ | — | ASVspoof 5 |
| Anacin et al. [31] | ✘ (explicitly untouched) | ✔ SEGAN, MetricGAN+ | No (independent) | ASVspoof 2019 LA |
| Wang et al. [28] | — | ✔ U-Net | **Yes (joint)** | ASVspoof 2019 LA (noisy) |
| TL-SEJ [29] | ✔ | ✔ U-Net | **Yes (joint)** | ASVspoof 2019 LA (noisy/reverb) |
| Trachu et al. [30] | — | SE as artifact *amplifier* | No | ASVspoof 2019 / 2021 |
| Sen et al. [16] | ✔ (multi-condition fine-tuning) | ✘ | — | ASVspoof 2021 DF |
| **Ours** | ✔ | ✔ SEGAN, MetricGAN+ (+ SEMamba, optional) | **No (independent, off-the-shelf)** | ASVspoof 2019 LA |

**Gap:** combining augmentation and enhancement has been studied only with *joint* optimisation [28, 29], which requires retraining the enhancer. Studies of *independent*, off-the-shelf enhancers [31] kept the detector's training data clean. Nobody has yet measured what happens when a noise-augmented detector receives independently enhanced audio. That is the common real-world setup, because practitioners use pretrained enhancers as-is.

**Our contribution (building on [20, 28–31]):** we extend Paper 2's experiment with a noise-augmented AASIST (following Paper 1's idea), keeping the enhancer independent. We ask:
- Does augmented training reduce the detector's dependence on the artifacts that independent SE removes (as suggested by [30])?
- Does Paper 2's quality-vs-detection inversion persist, shrink or reverse for an augmented detector?

A null result (cells ⑤ ≈ ⑥) is still informative. It would suggest that the joint optimisation of [28, 29] is necessary rather than optional.

---

## 9. Implications for our experimental design (please review as a team)

These come directly from reading the anchor papers:

1. **B9 (augmentation fraction), now answered.** Paper 1's "10%" is the size of the *subset selected*, but it made **9 copies** of that subset. The augmented data was ~90% of the original training size, not 10%. It also used **ASVspoof 5**, not 2019, and included codec/resampling/filter attacks we don't use. Our current T9 plan (one noisy copy of 10%) is a much lighter recipe. The team should decide either to match Paper 1's ratio (e.g. several noise/reverb copies of the 10% subset) or to state clearly that we use a different, lighter recipe "inspired by" [20].
2. **Paper 1 has no before/after comparison.** It does not show that augmentation improves AASIST against a non-augmented baseline, so we should not cite it as proof that augmentation helps. Our cells ② vs ⑤ provide that comparison on ASVspoof 2019 ourselves, which is a useful contribution in its own right.
3. **Paper 2 used no reverberation, and different noise.** Our noisy set adds reverb (70% of clips) and uses MS-SNSD. So our cell ② / ③ numbers **will not be directly comparable** to Paper 2's table. Consider also reporting a noise-only condition at fixed SNRs {0, 5, 10, 15, 20} dB to reproduce Paper 2 first (a stronger sanity check), then the noise + reverb condition.
4. **Report per-SNR results, not only a pooled EER.** Paper 2's most interesting behaviour (enhancement hurting at 20 dB) is only visible per SNR. That argues for fixed SNR steps (decision B10) at least for the eval set.
5. **SEGAN source.** Paper 2 used "pretrained SEGAN weights" without naming the checkpoint. We need a verifiable public SEGAN checkpoint. The current `config.py` entry points to SepFormer, which is a different model.
6. **Stated limitations to include:** a single non-SSL detector (see [9, 21]), a single dataset, and simulated rather than real-world noise.

---

## References

[1] X. Wang, J. Yamagishi, M. Todisco, H. Delgado, A. Nautsch, et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," *Computer Speech & Language*, 2020. arXiv:1911.01601.

[2] A. Nautsch, X. Wang, N. Evans, T. Kinnunen, V. Vestman, et al., "ASVspoof 2019: Spoofing countermeasures for the detection of synthesized, converted and replayed speech," *IEEE Trans. Biometrics, Behavior, and Identity Science*, 2021. arXiv:2102.05889.

[3] X. Liu, X. Wang, M. Sahidullah, J. Patino, H. Delgado, et al., "ASVspoof 2021: Towards spoofed and deepfake speech detection in the wild," *IEEE/ACM Trans. Audio, Speech, and Language Processing*. arXiv:2210.02437.

[4] X. Wang, H. Delgado, H. Tak, J.-w. Jung, H.-j. Shim, et al., "ASVspoof 5: Crowdsourced speech data, deepfakes, and adversarial attacks at scale," *ASVspoof 2024 Workshop*. arXiv:2408.08739.

[5] J. Yi, R. Fu, J. Tao, S. Nie, H. Ma, et al., "ADD 2022: The first audio deep synthesis detection challenge," *ICASSP 2022*. arXiv:2202.08433.

[6] N. M. Müller, P. Czempin, F. Dieckmann, A. Froghyar, K. Böttinger, "Does audio deepfake detection generalize?," *Interspeech 2022*. arXiv:2203.16263.

[7] J.-w. Jung, H.-S. Heo, H. Tak, H.-j. Shim, J. S. Chung, et al., "AASIST: Audio anti-spoofing using integrated spectro-temporal graph attention networks," *ICASSP 2022*. arXiv:2110.01200.

[8] H. Tak, J. Patino, M. Todisco, A. Nautsch, N. Evans, A. Larcher, "End-to-end anti-spoofing with RawNet2," *ICASSP 2021*. arXiv:2011.01108.

[9] H. Tak, M. Todisco, X. Wang, J.-w. Jung, J. Yamagishi, N. Evans, "Automatic speaker verification spoofing and deepfake detection using wav2vec 2.0 and data augmentation," *Odyssey 2022*. arXiv:2202.12233.

[10] J. Yi, C. Wang, J. Tao, X. Zhang, C. Y. Zhang, et al., "Audio deepfake detection: A survey," arXiv:2308.14970, 2023.

[11] L. Pham, P. Lam, D. Tran, H. Tang, T. Nguyen, et al., "A comprehensive survey with critical analysis for deepfake speech detection," *Computer Science Review* (to appear). arXiv:2409.15180.

[12] C. Hanilçi, T. Kinnunen, M. Sahidullah, A. Sizov, "Spoofing detection goes noisy: An analysis of synthetic speech detection in the presence of additive noise," *Speech Communication*, vol. 85, pp. 83–97, 2016. arXiv:1603.03947.

[13] H. Ali, S. Subramani, S. Sudhir, R. Varahamurthy, H. Malik, "Is audio spoof detection robust to laundering attacks?," *Proc. ACM Workshop on Information Hiding and Multimedia Security (IH&MMSec)*, 2024. doi:10.1145/3658664.3659656. arXiv:2408.14712.

[14] T. Chen, A. Kumar, P. Nagarsheth, G. Sivaraman, E. Khoury, "Generalization of audio deepfake detection," *Odyssey 2020*, pp. 132–137.

[15] C. Fan, M. Ding, J. Tao, R. Fu, J. Yi, et al., "Dual-branch knowledge distillation for noise-robust synthetic speech detection," arXiv:2310.08869, 2023.

[16] U. Sen, A. Luqman, A. Chattopadhyay, "Toward noise-aware audio deepfake detection: Survey, SNR-benchmarks, and practical recipes," arXiv:2512.13744, 2025.

[17] R. K. Das, J. Yang, H. Li, "Data augmentation with signal companding for detection of logical access attacks," *ICASSP 2021*, pp. 6349–6353. arXiv:2102.06332.

[18] H. Tak, M. Kamble, J. Patino, M. Todisco, N. Evans, "RawBoost: A raw data boosting and augmentation method applied to automatic speaker verification anti-spoofing," *ICASSP 2022*. arXiv:2111.04433.

[19] A. Cohen, I. Rimon, E. Aflalo, H. H. Permuter, "A study on data augmentation in voice anti-spoofing," *Speech Communication*, 2022. doi:10.1016/j.specom.2022.04.005. arXiv:2110.10491.

[20] H. Ali, S. Subramani, H. Malik, "Augmentation through laundering attacks for audio spoof detection," *ASVspoof 2024 Workshop*. arXiv:2410.01108. **(Paper 1)**

[21] K. Schäfer, J.-E. Choi, M. Neu, "Robust audio deepfake detection: Exploring front-/back-end combinations and data augmentation strategies for the ASVspoof5 Challenge," *ASVspoof 2024 Workshop*, pp. 56–63. doi:10.21437/ASVspoof.2024-9.

[22] S. Han, J. Seo, S. Choi, T. Kang, S. Chung, S. Lee, S. Park, S. Oh, I.-Y. Kwak, "Enhancing voice spoofing detection in noisy environments using frequency feature masking augmentation," *Engineering Science and Technology, an International Journal*, vol. 63, 101972, 2025. doi:10.1016/j.jestch.2025.101972.

[23] D.-T. Truong, T. Liu, J. Li, R. Tao, K. A. Lee, E. S. Chng, "Addressing gradient misalignment in data-augmented training for robust speech deepfake detection," *ICASSP 2026*. arXiv:2509.20682.

[24] S. Pascual, A. Bonafonte, J. Serrà, "SEGAN: Speech enhancement generative adversarial network," *Interspeech 2017*. arXiv:1703.09452.

[25] S.-W. Fu, C. Yu, T.-A. Hsieh, P. Plantinga, M. Ravanelli, et al., "MetricGAN+: An improved version of MetricGAN for speech enhancement," *Interspeech 2021*. arXiv:2104.03538.

[26] R. Chao, W.-H. Cheng, M. La Quatra, S. M. Siniscalchi, C.-H. H. Yang, et al., "An investigation of incorporating Mamba for speech enhancement," *IEEE SLT 2024*. arXiv:2405.06573.

[27] M. Ravanelli, T. Parcollet, P. Plantinga, A. Rouhe, S. Cornell, et al., "SpeechBrain: A general-purpose speech toolkit," arXiv:2106.04624, 2021.

[28] X. Wang, B. Zeng, H. Suo, Y. Wan, M. Li, "Robust audio anti-spoofing countermeasure with joint training of front-end and back-end models," *Interspeech 2023*, pp. 4004–4008. doi:10.21437/Interspeech.2023-1166.

[29] Y. Wang, X. Wang, H. Nishizaki, M. Li, "Enhancing anti-spoofing countermeasures robustness through joint optimization and transfer learning," arXiv:2407.20111, 2024. Journal version: Y. Wang, X. Wang, C. S. Leow, Q. Zhang, M. Li, H. Nishizaki, "Enhancing the robustness of speech anti-spoofing countermeasures through joint optimization and transfer learning," *IEICE Trans. Inf. & Syst.*, vol. E108-D, no. 12, pp. 1594–1604, 2025. doi:10.1587/transinf.2025EDP7044. **(TL-SEJ)**

[30] T. Trachu, T. Lertpetchpun, E. Chuangsuwanich, "Amplifying artifacts with speech enhancement in voice anti-spoofing," *Interspeech 2025*. arXiv:2506.11542.

[31] A. Anacin, S. Kshirsagar, A. R. Avila, "Investigating the impact of speech enhancement on audio deepfake detection in noisy environments," arXiv:2603.14767, 2026. **(Paper 2)** *Check the first author's name format: arXiv lists it as "Anacin, Angela".*

### Datasets and tools
[32] C. K. A. Reddy, E. Beyrami, J. Pool, R. Cutler, S. Srinivasan, et al., "A scalable noisy speech dataset and online subjective test framework," *Interspeech 2019*. arXiv:1909.08050. **(MS-SNSD)**

[33] T. Ko, V. Peddinti, D. Povey, M. L. Seltzer, S. Khudanpur, "A study on data augmentation of reverberant speech for robust speech recognition," *ICASSP 2017*, pp. 5220–5224. **(RIRS_NOISES)**

---

## Next steps for T18
- [ ] Team reads the full texts of [13], [20], [28]–[31] (the papers our argument depends on most).
- [ ] Settle the design points in §9 (B9, B10, SEGAN source) and record them in `DECISIONS.md`.
- [ ] Convert references to BibTeX (`docs/references.bib`) using each paper's official citation export.
- [ ] Before submission, re-search for anything published after Sep 2026 on "speech enhancement + augmentation + spoofing detection".
