# qPRO — Quick Suprathreshold Clinical Profile Specification

Design, normative cutoffs, and C99/WebAssembly implementation specification for **`qPRO`** (*Quick Suprathreshold Profile*), implemented in:
- **WebAssembly Clinical App:** `../qPRO/` (`qpro_core.c` → `qpro_core.wasm`, `index.html`, `qpro_report_svg.js`, `guide.html`)
- **Python Reference & Booklet Generator:** `sim/prototype_clinical_reports.py`

---

## 1. Audiologist-Facing 1-Page Landscape Design (`11:8.5` US Letter Landscape, `3 × 2` Grid)

1. **Header Banner:**
   - Title (`qPRO — QUICK SUPRATHRESHOLD CLINICAL PROFILE`), Color-Coded **Test Ear Badge** (`RIGHT EAR [O]` in Red `#DC2626`, `LEFT EAR [X]` in Blue `#2563EB`, `BINAURAL`, or `MONAURAL`), Patient ID, `PTA (1,2,4 kHz)`, `HF-PTA (3–8 kHz)`, `qVCV` Score (`Test / Retest`), QuickSIN-style **Estimated SNR Loss (`dB` + Normal / Mild / Moderate / Severe)**, **Clinical Summary Badge**, and **Aided Verification Summary**.
2. **Row 1 (`Panels A, B, C`):**
   - **Panel A — Loudness Scaling & SPL-o-gram (`0.25–8 kHz`, `dB SPL`):** Overlays `CU 5..50` contours against Normal-Hearing `Soft / UCL (±1 SD)` bands, pure-tone thresholds in `dB SPL`, `60 dB SPL` conversational speech line, and **Consonant Cue Frequency Zones** (*Voicing / Nasals*, *Place Bursts*, *Fricatives `S/SH/Z`*).
   - **Panel B — Audiogram & Cochlear Loss (`dB HL`):** Standard **Top-Down Clinical Audiogram** (`-10 dB HL` at top, toggleable to Bottom-Up) partitioning threshold elevation into **Outer Hair Cell Loss (Loudness Recruitment)** vs. **Inner Hair Cell / Sensitivity Loss**, with `Right (O)` / `Left (X)` ear symbols and `Min / Mild / Mod / Sev` **Spectral Smearing** tags.
   - **Panel C — Estimated SNR Loss Breakdown (`dB` ± `1 SE`):** Additive decomposition into **(1) Inaudibility** (*Restorable with Gain*), **(2) Recruitment & Upward Masking** (*`qCLS`-guided WDRC & MPO*), and **(3) Suprathreshold Clarity Loss** (*Requires directional / remote-mic SNR support*), with direct adaptive-SNR (`SRT Loss dB`) input readiness.
3. **Row 2 (`Panels D, E, F`):**
   - **Panel D — Expected vs. Actual `qVCV` Score (`%`):** Compares Normal-Hearing Avg (`97.5%`) → Expected (Audiogram Only) → Expected (Audiogram + `qCLS`) → Actual Unaided `qVCV` (+ Actual Aided `qVCV`).
   - **Panel E — Consonant Confusion Pattern (`%`) — De-Noised Estimate:** Empirical-Bayes smoothed percentage confusions + **Speech Cue Clarity (`%` of Normal)** for **Voicing**, **Manner**, and **Place**.
   - **Panel F — Clinical Interpretation, Test Quality & Fitting Guidance:**
     - **1. Test Quality & Booth Cross-Checks:** `[PASS]` / `[WARN]` / `[ALERT]` checks including `Audiogram vs. qCLS CU 5`, `Clear-Speech Practice`, `Repeatability & '???' Unsure Rate`, and **Binaural Interaural Asymmetry** (`|ΔPTA| >= 15 dB`, `|ΔqVCV| >= 15%`, or `|ΔClarity| >= 4 dB`).
     - **2. Main Consonant Confusions:** Plain text (`Heard 'G' as 'Z' (11/20, 55%)`).
     - **3. Clinical Fitting & Counseling Guidance + Direct REM Strip:** Patient-specific `0.5 / 1 / 2 / 4 kHz` **`Comfort CU 25`** and **`MPO Cap CU 50`** (`dB SPL`) strip for Audioscan Verifit / REM verification, followed by actionable Gain, Compression/MPO, and SNR/Aided Verification guidance.

---

## 2. WebAssembly parity & Performance (`qPRO/qpro_core.wasm`)

- **Size:** `59 KB` standalone WebAssembly MVP module (`qpro_core.wasm`) + offline `file://` base64-capable JS bridge (`qpro_core.js`).
- **Execution Time:** `10–24 ms` per patient in V8 WebAssembly (including 36-draw `SplitMix64` parametric bootstrap SE).
- **Parity vs. Python Reference Across All 98 `qtest_dataset` Listeners (`make test`):**
  - Max `|WASM - Python|` across Total, Audibility, Recruitment, and Clarity SNR Loss: **`0.020 dB`**
  - Automated Clinical Summary Agreement: **`98 / 98` (`100%`)**
