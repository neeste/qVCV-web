# Adaptive qVCV Implementation Roadmap (3-Phase Plan)

**Status:** **Phases 1 & 2 Implemented & Verified (November 18, 2026)** — Ready for Phase 3 Human Crossover Pilot (`index.html`, `qvcv_adaptive.js`, `app.js`, `Stimuli/ltass_noise.wav`, `sim/test_js_adaptive_parity.mjs`).  
**Context:** Derived from human-calibrated tracking and multi-SNR simulations across 98 listeners (`32 Normal`, `23 Slight`, `26 Mild`, `17 Moderate`; 19,600 unaided trials + 60-dB-SNR Practice block + 10-frequency audiograms) in `sim/`.

---

## Executive Summary & Design Decisions

1. **Keep Current Fixed-SNR Mode Intact for Backward Compatibility:**
   - The current `5 × 20 = 100`-trial Fixed-SNR protocol (`B:+18, D:+4, G:+3, K:0, N:+15, S:0, SH:0, T:0, V:+18, Z:0 dB`) is already near-optimal for binary NH-vs-HL screening ($A_{\text{ROC}} = 0.886$ vs. theoretical single-SNR ceiling $0.896$) and directly supports the published `mnr_vcv_pta` (`PPTA`), `Residual_Loss`, and hearing-aid `Benefit` norms (Hajicek, Harris, & Neely, 2026).
2. **Skip Static 5-SNR `.wav` File Banks:**
   - A static 5-SNR stimulus set improves binary screening $A_{\text{ROC}}$ by only `+0.004` (`0.896 -> 0.900`), requires `100` pre-baked `.wav` files, invalidates existing `PPTA` regression weights, and still spends ~3 of every 5 trials per consonant away from any individual listener's steepest psychometric slope (`69` trials needed vs. `42` with adaptive SNR).
3. **Do Not Adapt Consonant Selection — Adapt *Only* SNR Over Balanced 10-Consonant Blocks:**
   - Adaptive consonant selection yields negligible efficiency gain over balanced 10-consonant randomized blocks while degrading uniform consonant coverage. Keeping balanced 10-consonant blocks (with no boundary repeats and alternating token parity) while adapting SNR trial-by-trial captures the full **`2.36×` all-cohort efficiency gain** (**`4.05×` gain** in `Normal + Slight` listeners, `PTA ≤ 25 dB HL`).

---

## Phase 1: Dual-Mode Architecture & Real-Time Web Audio Mixing

### Goals
Add an **Adaptive Suprathreshold Mode** alongside the existing **Standard Screening Mode** without expanding the audio asset footprint or requiring a backend server.

### Technical Steps
1. **Mode Selector on Step 1 (Configuration Screen in `index.html` / `app.js`):**
   - `Standard Screening (Fixed SNR, 100 trials)` — exact current behavior.
   - `Adaptive Suprathreshold (50 trials default; optional 100-trial high-precision)` — trial-by-trial SNR selection.
   - Optional collapsible pure-tone audiogram input (`0.25–8 kHz` or 4-frequency `0.5, 1, 2, 4 kHz`) to enable post-test partitioning of total speech-in-noise shift into **Audibility Loss** vs. **Pure Suprathreshold Deficit ($D_i$)**.
2. **On-the-Fly Web Audio Speech + LTASS Noise Mixing (`21` `.wav` files total):**
   - Reuse the 20 existing calibrated clear VCV tokens in `Stimuli/practice/VCV_a*a_*_60SNR.wav` (`0.01 RMS = 60 dB SPL`).
   - Extract/store a single calibrated speech-shaped noise file (`Stimuli/ltass_noise.wav`, e.g., 5–10 s seamlessly loopable/sliceable LTASS waveform matching the exact spectrum of `Stimuli/test/`).
   - On each adaptive trial, select a random start offset in `ltass_noise.wav`, scale noise amplitude by $10^{-\text{SNR}_{\text{dB}} / 20}$ relative to the 0-dB-SNR reference, sum in a `Float32Array` `AudioBuffer`, and apply a headroom limiter guard (even at `-18 dB SNR`, RMS remains `>15 dB` below `1.0` full scale).
   - **Benefits:** Continuous `1-dB` SNR resolution (`-18` to `+30 dB`), zero additional `.wav` storage bloat, and elimination of frozen-noise learning across repetitions.

---

## Phase 2: Audiogram-Blind Hierarchical Adaptive SNR Engine & Clinical Readout

### Online Tracking Algorithm (`~40` lines of client-side JS, `<0.5 ms`/trial)
1. **Schedule:** 5 randomized 10-consonant blocks (`50` trials total, `~2 min`), preventing consecutive consonant repeats across block boundaries and alternating Token 1 / Token 2 across blocks.
2. **State Model (100% Audiogram-Blind Online):**
   - Model behavioral threshold shift relative to NH 50%-logit reference $\tau_{0,c} = -\alpha_{0,c}/k$ as:
     $$\theta_{i,c} = G_i + \delta_{i,c}, \qquad G_i \sim \mathcal{N}(4.0,\, 6.5^2), \quad \delta_{i,c} \sim \mathcal{N}(0,\, 4.5^2)$$
     with calibrated psychometric slope $k = 0.3356\text{ dB}^{-1}$, chance floor $\gamma = 1/11$, familiarized ceiling $c_{\max} = 0.996$, and token half-differences from `sim/human_data/human_listeners_98.json`.
   - Because cross-consonant coupling ($G_i$) shares information across all 10 VCVs, the first `10–15` trials warm-start the SNR neighborhood for every consonant.
3. **Trial SNR Selection:**
   - Evaluate 3-point Gauss-Hermite quadrature expected Fisher information over candidate SNRs `[-18..+30 dB]` (`1-dB` step) under $\mathcal{N}(\hat{\theta}_{i,c}, \hat{\sigma}_{i,c}^2)$ and play the argmax SNR (targeting `~56%` correct).
4. **Posterior Update:**
   - Run 2 damped Newton-Raphson MAP steps + closed-form Sherman-Morrison posterior variance update after each trial.

### Results Screen & CSV Enhancements
1. **CSV Columns:** Append `Target_SNR_dB`, `Running_Global_Deficit_dB`, and `Running_Global_SE_dB` to trial exports.
2. **Clinical Summary Metrics:**
   - **Global Speech-in-Noise / Suprathreshold Deficit (`dB SNR ± SE`)** relative to the 32-listener NH reference norm (plus Low/Mid Voiced-Nasal vs. High-Frequency Sibilant/Stop sub-scores).
   - **If Audiogram Provided:** Exact Plomp band-masking decomposition into **Audibility Component (`dB`)** and **Residual Suprathreshold Coding Deficit $D_i$ (`dB`)** via `decode_hierarchical_suprathreshold_map` (`sim/exp_suprathreshold_efficiency.py`).
   - **Intelligibility-Equalized Confusion Matrix & Phonetic Feature Transmission:** Report Miller-Nicely / Empirical-Bayes transmission of **Voicing**, **Manner**, and **Place** at `~55–60%` correct (`~20` errors in 50 trials even for `Normal` listeners, vs. `<4` errors in 100 trials under Fixed-SNR ceiling).

---

## Phase 3: Focused Human Crossover Pilot (`N ≈ 15–20` Listeners)

### Protocol (`~15 minutes` booth time per participant)
- **Cohort:** `~10` Normal-hearing adults (including listeners with self-reported speech-in-noise difficulty) + `~10` Slight-to-Moderate sensorineural HL listeners.
- **Conditions (counterbalanced):**
  1. Two `50`-trial **Adaptive-SNR** runs (test + retest; `~4 min` total).
  2. One (or two) `100`-trial **Standard Fixed-SNR** run (`~4–8 min`).

### Primary Validation Endpoints
1. **Test-Retest Repeatability:** Confirm that `50`-trial Adaptive `MAD` (`dB`) matches or improves upon `100`-trial Fixed-SNR repeatability (`predicted: 2.36× all-cohort speedup, 4.05× speedup for PTA ≤ 25 dB HL`).
2. **Low-SNR Psychometric Slope Verification:** Verify empirical human slopes at `-15` to `-5 dB SNR` under real-time Web Audio LTASS mixing and fine-tune $\tau_{0,c}$ norms if needed before locking clinical cutoffs.

---

## Reference Simulation Files (`sim/`)
- Calibration & Plomp Band-Masking Model: `sim/export_human_listeners.py`, `sim/human_data/human_listeners_98.json`
- Optimal Static Single-SNR vs. 5-SNR Screening Study: `sim/exp_optimal_snrs_per_vcv.py`
- Static vs. Adaptive Suprathreshold Efficiency Benchmark: `sim/exp_suprathreshold_efficiency.py`, `sim/human_calibrated_results/fig_suprathreshold_efficiency.png`
