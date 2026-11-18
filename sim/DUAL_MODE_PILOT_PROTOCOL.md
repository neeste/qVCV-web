# Dual-Mode `qVCV` Assessment Study Protocol

**Protocol Title:** Within-Subject Crossover Evaluation of Real-Time Adaptive-SNR (`50` Trials) vs. Standard Fixed-SNR (`100` Trials) Consonant-in-Noise Testing  
**Software Version:** Dual-Mode `qVCV` (`index.html`, `qvcv_adaptive.js`, `app.js`, `Stimuli/ltass_noise.wav`)  
**Total Session Duration:** `~25 minutes` per participant (`~14 minutes` active `qVCV` testing)

---

## 1. Objectives & Hypotheses

1. **Test–Retest Precision Efficiency (Primary):** Determine whether a `50`-trial (`~2 min`) **Real-Time Adaptive-SNR** `qVCV` run achieves equal or lower test–retest error (`MAD`, `ICC`) for Speech-in-Noise Loss (`dB`) than a `100`-trial (`~4 min`) **Standard Fixed-SNR** run—especially in Normal and Slight-HL ears (`PTA ≤ 25 dB HL`), where Fixed-SNR suffers a `>96%` ceiling effect (`predicted efficiency gain: 2.36× overall, 4.05× for PTA ≤ 25 dB HL`).
2. **Diagnostic Confusion Yield (Primary):** Verify that Adaptive-SNR equalizes intelligibility near **`54–58%` correct** (`~20–23` consonant confusions per 50 trials) across all hearing levels, yielding repeatable Miller–Nicely **Voicing**, **Manner**, and **Place** feature transmission estimates in listeners who make `<4` errors on Fixed-SNR.
3. **Empirical Low-SNR Psychometric Calibration (Secondary):** Verify the empirical human psychometric slope ($k$) and consonant intercepts ($\tau_{0,c}$) at negative SNRs (`-18` to `0 dB SNR`) under live Web Audio LTASS mixing against the 98-listener cohort model ($k = 0.3356\text{ dB}^{-1}$).

---

## 2. Participants (`N = 20` Adults)

Recruit `N = 20` adult native/fluent English speakers (`18–80 years`) across two strata (`10` per stratum):

| Stratum | Sample Size | Audiometric Criteria (Test Ear) | Recruitment Note |
| :--- | :---: | :--- | :--- |
| **Stratum 1: Normal / Near-Normal** | `n = 10` | 4-freq `PTA (0.5, 1, 2, 4 kHz) ≤ 20 dB HL` and all thresholds `0.25–8 kHz ≤ 25 dB HL` | Enrich with `~4–5` listeners reporting difficulty understanding speech in noise ("hidden" deficit candidates) |
| **Stratum 2: Slight-to-Moderate SNHL** | `n = 10` | `PTA` `21–55 dB HL`, sensorineural (air–bone gap `≤ 10 dB`), tympanometry Type A | Span Slight (`21–25`), Mild (`26–40`), and Moderate (`41–55 dB HL`) |

*Exclusion criteria:* Conductive/mixed loss in test ear, fluctuating hearing/active Ménière's, or inability to complete closed-set touchscreen/mouse responses. Clear-speech Practice score `< 70%` after one repeat is flagged separately in sensitivity analysis.

---

## 3. Equipment & Acoustic Calibration

1. **Booth & Transducer:** Sound-treated booth; desktop/laptop hosting `qVCV` locally (`python3 -m http.server 8080`), driving calibrated stereo insert earphones (Etymotic ER-3A or RadioEar IP30) via an external USB audio interface or clinical audiometer external-input routing.
2. **60 dB SPL Coupler Check (Daily / Pre-Study):**
   - On the `qVCV` Setup screen, select `Test Ear` (**Left** or **Right**), select **Tone (1 kHz, 60 dB SPL)** (`0.01 RMS`), click **Play Calibration Audio**, and adjust hardware output until a 2-cc (`HA-2`) coupler sound-level meter reads **`60.0 dB SPL (±0.5 dB)`** (`52.5 dB HL` for ER-3A).
   - Verify zero audible signal in the non-test earphone (`ChannelMergerNode` isolation check).

---

## 4. Counterbalanced Within-Subject Crossover Design

Each participant completes **one 10-trial clear-speech Practice block (`60 dB SNR`)** followed by **four Graded blocks** (`300` graded trials total = `~12–14 min` active testing) in ABAB / BABA counterbalanced order:

- **Odd Participant IDs (`P01, P03, ...` — Sequence ABAB):**
  1. `Practice (10 trials, 60 dB SNR)` + **`Adaptive-SNR (50 trials)` — Run 1** (`~2.5 min`)
  2. **`Standard Fixed-SNR (100 trials)` — Run 1** (`~4.0 min`)
  3. *2-minute seated rest break*
  4. **`Adaptive-SNR (50 trials)` — Run 2 (Retest)** (`~2.0 min`)
  5. **`Standard Fixed-SNR (100 trials)` — Run 2 (Retest)** (`~4.0 min`)

- **Even Participant IDs (`P02, P04, ...` — Sequence BABA):**
  1. `Practice (10 trials, 60 dB SNR)` + **`Standard Fixed-SNR (100 trials)` — Run 1** (`~4.5 min`)
  2. **`Adaptive-SNR (50 trials)` — Run 1** (`~2.0 min`)
  3. *2-minute seated rest break*
  4. **`Standard Fixed-SNR (100 trials)` — Run 2 (Retest)** (`~4.0 min`)
  5. **`Adaptive-SNR (50 trials)` — Run 2 (Retest)** (`~2.0 min`)

*(Optional abbreviated clinical-visit variant: Omit `Standard Fixed-SNR Run 2` if booth time is capped at `<10 min`, comparing `Adaptive-50` Test–Retest `MAD` against both `Standard-100 Run 1` agreement and the 98-listener archival `Standard-100` Test–Retest benchmark `MAD = 1.04 dB`.)*

---

## 5. Step-by-Step Booth Operator Checklist

1. **Baseline Audiometry:** Measure air-conduction thresholds at `250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000 Hz` (`dB HL`). *(Optional: expand `Optional: Pure-Tone Audiogram` on `qVCV` Setup and enter the 10 thresholds—note the online Adaptive-SNR tracker remains 100% audiogram-blind during testing and uses the audiogram only for post-test Plomp Audibility vs. Clarity reporting.)*
2. **Block 1:**
   - Enter `Participant ID` (e.g., `P01_R1`), select `Test Ear` (`Left` or `Right`), `Condition = Unaided`, `Test Mode` (`Adaptive-SNR 50` or `Standard 100` per sequence), keep **`Include 10-trial clear-speech Practice Phase` checked**, and click **Step 2: Ready to Begin Test**.
   - At *Practice Complete*: if Practice `< 8/10` (`<80%`), re-verify insert seal and instructions and click **Repeat Practice** once; then click **Step 3: Begin Graded Test**.
   - At *Test Complete*: click **Download CSV Results**, then click **New Run (Keep Setup)** (which preserves ID/Ear/Audiogram and automatically unchecks Practice for Blocks 2–4).
3. **Blocks 2–4:**
   - Update `Participant ID` run suffix if desired (e.g., `P01_R1` vs `P01_R2`; note Adaptive CSV filenames also auto-append `ADP50_HHMM`), toggle `Test Mode` per sequence, confirm Practice is unchecked, run the block, and click **Download CSV Results**.

---

## 6. Pre-Specified Statistical Analysis & Success Criteria

All exported CSVs contain `Trial,Phase,Stimulus,Target,Response,Target_SNR_dB,RT_ms,Running_SNR_Loss_dB,Running_SE_dB,Final_SNR_Loss_dB,Mode,Condition,Ear`.

1. **Test–Retest Repeatability (Primary):**
   - Compute within-subject absolute test–retest difference `|Run 1 − Run 2|` on `Final_SNR_Loss_dB` (and Plomp `Suprathreshold_Deficit_dB` using baseline audiograms) for `Adaptive-50` vs. `Standard-100`.
   - **Success Criterion:** `Adaptive-50` (`50` trials) achieves non-inferior or lower mean/median `|Run 1 − Run 2|` (`MAD ≤ 1.15 dB`) than `Standard-100` (`100` trials) across all `N = 20`, and significantly lower `MAD` (paired Wilcoxon $p < 0.05$) or higher `ICC(2,1)` within Stratum 1 (`PTA ≤ 20 dB HL`).
2. **Ceiling Elimination & Confusion-Matrix Reliability (Primary):**
   - Compare graded `% Correct` and total error count per session (`Adaptive-50` expected IQR: `50–62%` correct, `19–25` errors/run across *both* strata vs. `Standard-100` Stratum 1 median `≥96%` correct, `≤4` errors/run).
   - Compare Run 1 vs. Run 2 reproducibility (`RMSE` / correlation) of Miller–Nicely **Voicing**, **Manner**, and **Place** `%` information transmitted.
3. **Cross-Mode Convergent Validity & Psychometric Model Check (Secondary):**
   - Evaluate Deming regression and Pearson/Spearman correlation between `Adaptive-50` `Final_SNR_Loss_dB` and `Standard-100` `Final_SNR_Loss_dB` ($r \ge 0.85$ expected across combined cohort).
   - Fit a mixed-effects logistic psychometric curve to all pooled `Adaptive-50` trials (`1,000` trials across `20` listeners) to verify empirical slope $\hat{k}$ against `0.3356 dB⁻¹` and check whether any consonant's NH 50%-logit intercept $\tau_{0,c}$ warrants fine-tuning (`> 1.5 dB` offset).
4. **Administration Time & Listening Effort (Secondary):**
   - Compare total block duration (`s`) and median post-stimulus reaction time (`RT_ms`) across modes and SNRs.
