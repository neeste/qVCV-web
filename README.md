# qVCV Web Application: Architectural Overview & User Guide

The **qVCV** web application is a clinical-grade, closed-set speech-in-noise test designed to evaluate consonant confusion. The basic task requires the listener to hear a short vowel-consonant-vowel sequence and identify the consonant heard from a grid of options. To increase the difficulty of the task and measure performance, varying amounts of speech-shaped noise are baked into the audio stimuli.

## 1. Technical Architecture
The application is built entirely with **HTML5, CSS3, and JavaScript**. It requires no backend database or cloud processing, making it highly secure and hardware-independent.
- **100% Client-Side Execution:** All logic, state management, and audio routing execute locally within the web browser.
- **HIPAA Compliant by Default:** Because there is no external server, patient data (such as Participant IDs and trial results) never leaves the physical testing room or touches the internet.
- **LAN Deployment Model:** The gold standard for clinical implementation is to host the application on the audiologist's computer using a lightweight local web server. The testing computer inside the sound booth navigates to the audiologist's local IP address, ensuring zero-latency audio playback and immunity to external internet outages.
- **Audio Calibration Standard:** Digital audio headroom is strictly managed. A 1 kHz calibration tone is generated natively via the Web Audio API with a target RMS amplitude of 0.01. This 0.01 RMS reference mathematically equates to 60 dB SPL at the listener's ear canal, allowing the hardware to scale appropriately without digital clipping.

## 2. Directory Structure & Setup
Because the application relies on fetching local `.wav` files via JavaScript, modern browser security protocols (CORS) require the files to be served over HTTP/HTTPS rather than opened directly from a desktop folder.

To set up the repository for deployment, ensure your directory matches this exact structure:
- `index.html`: The main user interface and Dual-Mode setup/results layout.
- `style.css`: The styling rules, grid layouts, summary cards, and visual elements.
- `qvcv_adaptive.js`: Calibrated psychometric engine (non-repeating schedulers, Hierarchical Bayesian Adaptive-SNR tracker, Plomp audibility/clarity decoder, published `PPTA` model, and Miller-Nicely phonetic feature transmission).
- `app.js`: Web Audio API stereo ear-routing controller, 24-bit WAV preloader, live speech + LTASS noise mixer, Practice QC gate, and CSV exporter.
- `Stimuli/practice/`: Must contain the 20 clear (`60 dB SNR`) `.wav` files.
- `Stimuli/test/`: Must contain the 20 fixed-SNR noise-masked `.wav` files (used by `Standard` mode).
- `Stimuli/ltass_noise.wav`: Calibrated `8.0`-second circularly periodic speech-shaped noise (`0 dB SNR` reference RMS = `0.01018`, used by `Adaptive-SNR` mode).

## 3. Test Structure, Dual Modes & Phasing
The application utilizes a strict state machine to guide the participant through setup, calibration, practice, and graded testing.

### Step 1: Configuration & Calibration
- **Participant & Protocol:** Enter `Participant ID`, `Test Ear` (`Left`, `Right`, or `Both` — routed strictly to the selected physical stereo channel[s] via Web Audio `ChannelMergerNode`), `Condition` (`Unaided` or `Aided`), and **`Test Mode`**:
  - **`Standard (Fixed SNR, 100 trials)`** *(default)*: Presents the published per-consonant fixed SNRs (`5 × 20` blocks with zero consecutive consonant repeats), preserving 100% compatibility with published `PPTA` norms.
  - **`Adaptive-SNR (50 trials, ~2 min)`** *(or `100 trials, ~4 min`)*: Mixes the 20 clear (`60 dB SNR`) speech tokens on the fly with random circular slices of `Stimuli/ltass_noise.wav` in `1-dB` steps (`-18` to `+30 dB SNR`), adapting SNR trial-by-trial via a coupled hierarchical Bayesian tracker over balanced 10-consonant blocks to target `~56%` accuracy (`2.36×` overall speedup; `4.05×` speedup for `PTA ≤ 25 dB HL`).
- **Optional Pure-Tone Audiogram (`dB HL`):** Collapsible `250–8000 Hz` input; when provided, post-test scoring decomposes total Speech-in-Noise Loss into **Audibility Component (`dB`)** and **Suprathreshold Clarity Deficit (`dB ± SE`)**.
- **Calibration:** Plays either continuous clear VCV speech or a `1 kHz` tone (`0.01 RMS` = `60 dB SPL`) routed through the selected `Test Ear`.

### Step 2: Instructions Pop-Up, Practice Phase & Booth QC Gate
- **Instructions Pop-Up (`OK` Gate):** Clicking **Step 2: Ready to Begin Test** (and **Step 3: Begin Graded Test**) opens a modal Instructions dialog—matching `qCLS`—so no audio plays until the participant or clinician clicks **OK**.
- **Stimuli:** Presents 10 randomly selected clear stimuli (`60 dB SNR`, one per consonant) with green visual feedback on the target button.
- **Booth QC Check:** Logs all Practice trials and displays the Practice score upon completion. If `< 8/10` (`< 80%`), displays an amber QC alert and offers a **Repeat Practice** button before starting the Graded phase. (On repeat runs via *New Run (Keep Setup)*, Practice can also be toggled off for rapid crossover retests.)

### Step 3: Graded Test Phase & Mid-Test Pause / Restart
- **Standard Mode:** Presents `100` fixed-SNR stimuli (`5 × 20` blocks, alternating token parities, zero consecutive consonant repeats).
- **Adaptive-SNR Mode:** Presents `50` (or `100`) live-mixed stimuli (`5 × 10` or `10 × 10` balanced consonant blocks, alternating Token 1 / Token 2, zero consecutive consonant repeats, `1-dB` expected-Fisher-information SNR selection).
- **Mid-Test Pause / Stop (Keep Settings):** Clicking **Pause / Stop** during Practice or Graded testing immediately halts playback and offers three choices—**Resume Test (Replay Current Trial)**, **Restart Test from Trial 1 (Keep Settings)**, or **Stop & Return to Setup (Keep Settings)**—without losing Participant ID, Ear, Condition, Mode, or Audiogram entries.
- **Feedback:** No visual feedback is provided to the listener.

## 4. User Interface, Clinical Readout & Data Export

### The Response Grid
The interface features an 11-button grid: the 10 target consonants (`B, D, G, K, N, S, SH, T, V, Z`) plus `???` to prevent forced-choice guessing artifacts, with a dynamic `Trial X of Y` progress bar.

### Clinical Summary & Confusion Matrix
At test completion, the Results screen displays:
- **Accuracy, Practice QC & Median Reaction Time (`ms`)**
- **Speech-in-Noise Loss (`dB ± SE` vs. 32-listener NH Norm)** with Low/Mid Voiced-Nasal (`B,D,G,N,V`) vs. High-Frequency Sibilant/Stop (`K,S,SH,T,Z`) sub-shifts
- **Predicted PTA (`PPTA`, `dB HL`)** in Standard mode, or **Plomp Audibility vs. Suprathreshold Clarity Split (`dB`)** whenever an audiogram is entered
- **Miller-Nicely Phonetic Feature Information Transmitted (`%`)** for **Voicing**, **Manner**, and **Place**
- **10 × 11 Confusion Matrix Heatmap** with a **`Counts` | `Row %`** toggle

### CSV Export
Clicking **Download CSV Results** exports both `Practice` and `Graded` trials (100% compatible with `qPRO`, `sim/`, `pandas`, and MATLAB):
- **Dynamic Naming (Date + `HHMM` Timestamp on All Modes):** `qVCV_[ID]_[Ear]_[YYYYMMDD]_[HHMM].csv` for Standard Unaided runs (e.g., `qVCV_NH001_L_20261119_1105.csv`), and inserts `AIDED` and/or `ADP50` / `ADP100` tags before `[YYYYMMDD]_[HHMM]` for Aided or Adaptive runs so back-to-back blocks never overwrite one another.
- **Data Columns:** `Trial,Phase,Stimulus,Target,Response,Target_SNR_dB,RT_ms,Running_SNR_Loss_dB,Running_SE_dB,Final_SNR_Loss_dB,Mode,Condition,Ear`.

## 5. Headless Tracking Simulation & Evaluation (`sim/`)
Mirroring `qCLS/sim/`, the `sim/` directory provides a standalone Python simulation and benchmark suite to evaluate confusion-matrix recovery accuracy, test-retest reliability, per-cell bias, trial allocation, and convergence across trial budgets:

- `sim/qvcv_simulator.py`: Virtual-listener generator and ground-truth loader. Auto-generates an 18-listener synthetic clinical cohort (`sim/virtual_listeners.csv` / `.json`) across six audiological profiles (Normal Hearing, Mild Sloping HF, Steep Ski-Slope HF, Spectral Smearing, Temporal/Voicing Deficit, and Poor SIN / High Guess), and also loads empirical `qVCV_*.csv` trial exports or MATLAB `.mat` confusion matrices.
- `sim/qvcv_trackers.py`: Stimulus-selection policies (`block`, `random`, `adaptive_var`, `adaptive_eig`, `adaptive_feature`) and confusion-matrix estimators (`empirical`, `dirichlet`, `feature_eb`).
- `sim/qvcv_tracking_sim.py`: CLI harness evaluating per-repetition accuracy (Cell MAE/RMSE, Mean & Worst-Row Total Variation Distance, Per-Consonant ID MAE, Jensen-Shannon divergence, and Miller-Nicely phonetic feature information transmission for voicing, manner, and place), pairwise test-retest reliability, per-cell signed bias, and `--sweep_trials` convergence sweeps.
- `sim/exp_adaptive_vs_block.py`: Full factorial benchmark comparing the shipped 5x20 block procedure against adaptive consonant selection and coupled phonetic-feature Empirical-Bayes estimation, outputting summary tables and diagnostic figures in `sim/benchmark_results/`.
- `sim/qvcv_multisnr_simulator.py`, `sim/qvcv_multisnr_designs.py`, & `sim/exp_multisnr_study.py`: Continuous SNR-dependent virtual-listener simulator and stimulus-design benchmark comparing single-SNR repetitions against multi-SNR stimulus sets (consonant-relative and shared absolute 5-SNR grids) as well as 2D Consonant × SNR adaptive tracking vs. stratified block/random selection (`sim/multisnr_results/`).
- `sim/export_human_listeners.py`, `sim/exp_human_calibrated_study.py`, `sim/exp_optimal_snrs_per_vcv.py`, & `sim/exp_suprathreshold_efficiency.py`: Calibrates virtual listeners directly to the empirical 98-listener `qtest_dataset` cohort (32 Normal, 23 Slight, 26 Mild, 17 Moderate; 19,600 unaided trials + 60-dB-SNR Practice ceiling + consonant-specific Plomp power-sum LTASS band masking + exact `mnr_vcv_pta` clinical PPTA regression) in `sim/human_data/`, and evaluates fixed-SNR tracking, unbiased LOOCV Hellinger $A_{\text{ROC}}$, optimal Single-SNR vs. 5-SNR stimulus sets, and static vs. trial-by-trial adaptive SNR efficiency for quantifying suprathreshold hearing deficits (`sim/human_calibrated_results/`).
- `sim/qvcv_biophysical_model.py` & `sim/characterize_human_listeners.py`: Biophysical spectro-temporal + paired `qCLS` virtual-listener engine and human listener characterization suite. Extracts 3-epoch × 10-ERB intervocalic consonant templates, flanking `/a/` vowel masking spectra, F0 pitch periodicity, and envelope continuity directly from the `.wav` waveforms; couples 10-frequency `qCLS` equal-loudness contours (`96/98` paired listeners) to partition Outer-Hair-Cell (OHC) filter broadening & upward spread of masking vs. Inner-Hair-Cell (IHC) threshold loss and neural suprathreshold deficit; models directional cue-loss confusions (`V->Z`, `K->T`, `S->T`, `D->G`) + a `1.5%` attention-lapse floor; and characterizes both cohort (`98` unaided + `67` aided sessions) and standalone `qVCV_*.csv` files.
- `sim/prototype_clinical_reports.py` & `sim/CLINICAL_INTERPRETER_SPEC.md`: Automated 1-page single-view clinical suprathreshold report generator combining the Pure-Tone Audiogram (`0.25–8 kHz`, BTNRH `er3a` / `legacy` RETSPLs), `qCLS` equal-loudness contours (`qCLS_ELC_*.csv`), and `qVCV` trial exports (`qVCV_*.csv` + optional Aided verification) into printable US Letter Landscape PDF/PNG clinical decision-support reports (`sim/clinical_reports/`).
- **Companion Web App (`qPRO`):** The real-time C99/WebAssembly browser port of this clinical interpreter and `qPRO-Rx` hearing-aid prescription generator lives in the sibling repository [`neeste/qPRO`](https://github.com/neeste/qPRO) (`http://audres.org/qPRO/`).
- `sim/ADAPTIVE_QVCV_ROADMAP.md`: Saved 3-phase implementation plan for adding real-time Web Audio adaptive SNR tracking alongside the standard fixed-SNR screening mode.

Example commands from the repository root:
```bash
# Generate 1-page prototype clinical reports (PNG + PDF) for 6 real qtest_dataset cases & 98-listener norms
python3 sim/prototype_clinical_reports.py

# Generate a 1-page PDF/PNG report for any individual qtest_dataset listener (e.g. HL054)
python3 sim/prototype_clinical_reports.py --listener_id HL054

# Generate a 1-page clinical report from exported web-app qVCV + qCLS CSVs + Audiogram
python3 sim/prototype_clinical_reports.py \
  --qvcv_csv path/to/qVCV_NH001_L_20260511.csv \
  --qcls_elc_csv path/to/qCLS_ELC_NH001_L_20260511.csv \
  --audiogram 10,10,15,15,20,25,35,45,55,60 \
  --transducer er3a

# Calibrate & validate the biophysical spectro-temporal + qCLS listener model
python3 sim/characterize_human_listeners.py

# Characterize a single exported human qVCV CSV (+ optional 10-freq audiogram in dB HL)
python3 sim/characterize_human_listeners.py --csv path/to/qVCV_NH001_L_20260511.csv \
  --audiogram 10,10,15,15,20,25,35,45,55,60
# Export 98 human-calibrated virtual listeners & run human-calibrated benchmarks
python3 sim/export_human_listeners.py
python3 sim/exp_human_calibrated_study.py
python3 sim/exp_optimal_snrs_per_vcv.py
python3 sim/exp_suprathreshold_efficiency.py

# Compare Single-SNR vs. 5-SNR stimulus sets and adaptive vs. block/random selection
python3 sim/exp_multisnr_study.py

# Run the fixed-SNR multi-arm convergence & trial-efficiency benchmark
python3 sim/exp_adaptive_vs_block.py --n_reps 30

# Evaluate a single tracking mode & save confusion-matrix comparison plots
python3 sim/qvcv_tracking_sim.py --mode adaptive_feature --n_trials 60 --n_reps 25 --save_plots

# Run a convergence sweep on an exported empirical qVCV CSV (or directory of CSVs)
python3 sim/qvcv_tracking_sim.py --ground_truth path/to/qVCV_NH001_L_20260511.csv \
  --mode adaptive_feature --sweep_trials 20 40 60 80 100
```
