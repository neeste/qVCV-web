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
