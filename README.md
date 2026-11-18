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
- `index.html`: The main user interface and application controller.
- `style.css`: The styling rules, grid layouts, and visual elements.
- `app.js`: The core logic for tracking trials, block randomization, and generating the heatmap.
- `Stimuli/practice/`: Must contain the 20 clear (60 SNR) `.wav` files.
- `Stimuli/test/`: Must contain the 20 noise-masked `.wav` files.

## 3. Test Structure & Phasing
The application utilizes a strict state machine to guide the participant through three distinct phases.

### Step 1: Configuration & Calibration
- **Participant Data:** The audiologist or user inputs a "Participant ID" and selects the "Test Ear" (Left, Right, or Both) from a dropdown menu.
- **Calibration:** The user selects a calibration source: either a continuous loop of clear speech tokens or a 1 kHz Tone. The external hardware volume should be adjusted until the output reaches a comfortable listening level or exactly 60 dB SPL.

### Step 2: Practice Phase
- **Stimuli:** Presents 10 randomly selected clear stimuli (60 SNR). The algorithm ensures exactly one instance of each consonant is played.
- **Feedback:** Provides visual feedback by highlighting the correct consonant button in green after the user responds.
- **Instructions:** "Listen to the stimulus and select the consonant you heard. If you’re unsure, take your best guess. Select ??? only if you think you heard a consonant that is not one of the choices.".

### Step 3: Graded Test Phase
- **Stimuli:** Presents 100 noise-masked stimuli.
- **Block Randomization:** To prevent cluster biases, the 20 unique test files are shuffled into blocks of 20. This process is repeated 5 times, ensuring even distribution and preventing the same consonant from playing consecutively.
- **Feedback:** No visual feedback is provided to the listener.

## 4. User Interface & Data Export

### The Response Grid
The interface features an 11-button grid. It includes the 10 target consonants (B, D, G, K, N, S, SH, T, V, Z) and an 11th catch-all button labeled "???". This 11th option is standard in psychometric testing to prevent forced-choice errors from skewing the final data. An invisible CSS spacer is utilized to ensure the 4-column grid remains perfectly symmetrical.
A dynamic progress bar displays "Trial X of Y" above the grid to reduce patient anxiety and testing fatigue.

### Results & Visualization
At the conclusion of the 100-trial Graded Phase, the application instantly renders an 11x10 Heatmap. This table plots the Actual Stimulus against the Listener Response, with the background color intensity of each cell mathematically scaled based on the frequency of that specific error.

### CSV Export
Clicking the "Download CSV Results" button exports the complete trial-by-trial history.
- **Dynamic Naming:** The resulting file is automatically named using the metadata captured on the first screen: `qVCV_[ParticipantID]_[Ear]_[YYYYMMDD].csv` (e.g., `qVCV_NH001_L_20260511.csv`).
- **Data Columns:** The CSV tracks Trial Number, Phase, Stimulus Filename, Target Consonant, and User Response.

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
