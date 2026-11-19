/*
 * app.js — Dual-Mode qVCV Controller & Calibrated Web Audio Engine
 *
 * Modes:
 *   - standard_100 : Shipped Fixed-SNR screening & PPTA protocol (5x20 blocks,
 *                    zero consecutive consonant repeats).
 *   - adaptive_50  : Real-time 1-dB LTASS Adaptive-SNR tracking (5x10 balanced
 *                    blocks, ~2 min; hierarchical Bayesian tracker).
 *   - adaptive_100 : High-precision Adaptive-SNR tracking (10x10 blocks, ~4 min).
 *
 * Operator UX (per clinical feedback):
 *   - Instructions pop-up modal on "Step 2: Ready to Begin Test" (and Step 3)
 *     requiring "OK" before playing the first stimulus (matching qCLS).
 *   - Mid-test "Pause / Stop" button allowing Resume (replaying current trial),
 *     Restart from Trial 1 (keeping all settings), or Return to Setup (keeping settings).
 *   - Date + HHMM timestamp on all exported CSV filenames (both Fixed and Adaptive).
 */

const stimuliConsonants = window.QVCVEngine.STIM_CONSONANTS;
const displayConsonants = window.QVCVEngine.DISP_CONSONANTS;
const audFreqLabels = ["250", "500", "750", "1k", "1.5k", "2k", "3k", "4k", "6k", "8k"];

const practiceFiles = [
    "VCV_asa_1_60SNR.wav", "VCV_asa_2_60SNR.wav", "VCV_aba_1_60SNR.wav", "VCV_asha_1_60SNR.wav",
    "VCV_aba_2_60SNR.wav", "VCV_asha_2_60SNR.wav", "VCV_ada_1_60SNR.wav", "VCV_ata_1_60SNR.wav",
    "VCV_ada_2_60SNR.wav", "VCV_ata_2_60SNR.wav", "VCV_aga_1_60SNR.wav", "VCV_ava_1_60SNR.wav",
    "VCV_aga_2_60SNR.wav", "VCV_ava_2_60SNR.wav", "VCV_aka_1_60SNR.wav", "VCV_aza_1_60SNR.wav",
    "VCV_aka_2_60SNR.wav", "VCV_aza_2_60SNR.wav", "VCV_ana_1_60SNR.wav", "VCV_ana_2_60SNR.wav"
];

const testFilesBase = [
    "VCV_aba_2_18SNR.wav", "VCV_aga_1_3SNR.wav", "VCV_aka_2_0SNR.wav", "VCV_asa_1_0SNR.wav",
    "VCV_asha_2_0SNR.wav", "VCV_ava_1_18SNR.wav", "VCV_aza_2_0SNR.wav", "VCV_ada_1_4SNR.wav",
    "VCV_aga_2_3SNR.wav", "VCV_ana_1_15SNR.wav", "VCV_asa_2_0SNR.wav", "VCV_ata_1_0SNR.wav",
    "VCV_ava_2_18SNR.wav", "VCV_aba_1_18SNR.wav", "VCV_ada_2_4SNR.wav", "VCV_aka_1_0SNR.wav",
    "VCV_ana_2_15SNR.wav", "VCV_asha_1_0SNR.wav", "VCV_ata_2_0SNR.wav", "VCV_aza_1_0SNR.wav"
];

const MODE_DESCRIPTIONS = {
    standard_100: "Fixed per-consonant SNRs (5\u00d720 blocks); preserves published PPTA norms.",
    adaptive_50:  "Real-time 1-dB LTASS noise tracking (~56% target) over 5\u00d710 balanced blocks (~2 min).",
    adaptive_100: "High-precision 1-dB LTASS noise tracking (~56% target) over 10\u00d710 balanced blocks (~4 min)."
};

let state = "IDLE"; /* IDLE | PRACTICE | PRACTICE_DONE | GRADED | FINISHED */
let currentMode = "standard_100";
let currentRunUsedPractice = true;
let pendingLaunchTarget = null; /* "PRACTICE" | "GRADED" | null */
let isPaused = false;
let pendingTimerId = null;

let currentQueue = [];
let currentTrialSpec = null;
let adaptiveTracker = null;
let practiceResults = [];
let trialResults = [];
let confusionMatrix = {};
let lastSessionSummary = null;
let heatmapView = "counts"; /* "counts" | "pct" */
let isWaitingForAudio = false;
let stimOffsetTimestampMs = null;

// Progress tracking
let totalPhaseTrials = 0;
let currentTrialNumber = 0;

// Web Audio API graph & preloaded buffer cache
let audioCtx = null;
let monoBus = null;
let leftEarGain = null;
let rightEarGain = null;
let earMerger = null;
const bufferCache = new Map();

// Active audio nodes
let calibSourceNode = null;
let calibEnvGain = null;
let activeTrialSource = null;
let isCalibrating = false;

// DOM Elements
const appContainer = document.getElementById("app-container");
const grid = document.getElementById("consonant-grid");
const startBtn = document.getElementById("start-btn");
const repeatPracticeBtn = document.getElementById("repeat-practice-btn");
const interfaceDiv = document.getElementById("interface");
const resultsArea = document.getElementById("results-area");
const statusMsg = document.getElementById("status-msg");
const phaseTitle = document.getElementById("phase-title");
const downloadBtn = document.getElementById("download-btn");
const newRunBtn = document.getElementById("new-run-btn");
const setupArea = document.getElementById("setup-area");
const calibBtn = document.getElementById("calib-btn");
const progressBar = document.getElementById("test-progress");
const currentTrialSpan = document.getElementById("trial-current");
const totalTrialSpan = document.getElementById("trial-total");
const calibRadios = document.getElementsByName("calib-source");
const earSelect = document.getElementById("test-ear");
const condSelect = document.getElementById("test-condition");
const modeSelect = document.getElementById("test-mode");
const modeHint = document.getElementById("mode-hint");
const includePracticeChk = document.getElementById("include-practice");
const qcBanner = document.getElementById("practice-qc-banner");
const audGridEl = document.getElementById("audiogram-grid");
const hmCountsBtn = document.getElementById("hm-counts-btn");
const hmPctBtn = document.getElementById("hm-pct-btn");

// Dialog Elements
const instructionsModal = document.getElementById("instructions-modal");
const instructionsTitle = document.getElementById("instructions-modal-title");
const instructionsList = document.getElementById("instructions-modal-list");
const instructionsOkBtn = document.getElementById("instructions-ok-btn");
const instructionsBackBtn = document.getElementById("instructions-back-btn");

const pauseBtn = document.getElementById("pause-btn");
const pauseModal = document.getElementById("pause-modal");
const pauseModalStatus = document.getElementById("pause-modal-status");
const resumeTestBtn = document.getElementById("resume-test-btn");
const restartTestBtn = document.getElementById("restart-test-btn");
const abortSetupBtn = document.getElementById("abort-setup-btn");

/* Build Optional 10-Frequency Audiogram Inputs */
audFreqLabels.forEach((lbl, idx) => {
    const cell = document.createElement("div");
    cell.className = "aud-cell";
    cell.innerHTML = `<span>${lbl} Hz</span><input type="number" min="-10" max="120" step="5" data-fidx="${idx}" placeholder="--">`;
    audGridEl.appendChild(cell);
});

function resetConfusionMatrix() {
    confusionMatrix = {};
    stimuliConsonants.forEach(s => {
        confusionMatrix[s] = {};
        displayConsonants.forEach(r => {
            confusionMatrix[s][r] = 0;
        });
    });
}
resetConfusionMatrix();

/* Build 11-Button Consonant Response Grid */
displayConsonants.forEach(c => {
    if (c === "???") {
        const spacer = document.createElement("div");
        spacer.className = "grid-spacer";
        grid.appendChild(spacer);
    }
    const btn = document.createElement("button");
    btn.type = "button";
    btn.innerText = c;
    btn.className = "consonant-btn";
    btn.dataset.consonant = c;
    btn.onclick = () => handleResponse(c);
    grid.appendChild(btn);
});

/* Event Listeners */
modeSelect.addEventListener("change", () => {
    modeHint.textContent = MODE_DESCRIPTIONS[modeSelect.value] || "";
});

earSelect.addEventListener("change", () => {
    applyEarRouting();
});

calibRadios.forEach(radio => {
    radio.addEventListener("change", () => {
        if (isCalibrating) {
            stopCalibration();
            startCalibration();
        }
    });
});

calibBtn.addEventListener("click", () => {
    if (!isCalibrating) {
        startCalibration();
    } else {
        stopCalibration();
    }
});

startBtn.addEventListener("click", () => requestPhaseStart(false));
repeatPracticeBtn.addEventListener("click", () => requestPhaseStart(true));
downloadBtn.addEventListener("click", downloadCSV);
newRunBtn.addEventListener("click", () => returnToSetupKeepSettings(true));

/* Instructions Modal Events */
instructionsOkBtn.addEventListener("click", confirmInstructionsAndLaunch);
instructionsBackBtn.addEventListener("click", cancelInstructionsModal);
instructionsModal.addEventListener("cancel", (e) => {
    e.preventDefault();
    cancelInstructionsModal();
});

/* Mid-Test Pause / Stop Modal Events */
pauseBtn.addEventListener("click", openPauseModal);
resumeTestBtn.addEventListener("click", resumePausedTest);
restartTestBtn.addEventListener("click", restartFromTrialOneKeepSettings);
abortSetupBtn.addEventListener("click", () => {
    if (pauseModal.open) pauseModal.close();
    isPaused = false;
    returnToSetupKeepSettings(false);
});
pauseModal.addEventListener("cancel", (e) => {
    /* Prevent accidental Escape key leaving test paused with no modal visible */
    e.preventDefault();
});

hmCountsBtn.addEventListener("click", () => {
    heatmapView = "counts";
    hmCountsBtn.classList.add("active");
    hmPctBtn.classList.remove("active");
    renderHeatmap();
});
hmPctBtn.addEventListener("click", () => {
    heatmapView = "pct";
    hmPctBtn.classList.add("active");
    hmCountsBtn.classList.remove("active");
    renderHeatmap();
});

/* --- TIMER & AUDIO CLEANUP HELPERS --- */

function clearPendingTimer() {
    if (pendingTimerId !== null) {
        clearTimeout(pendingTimerId);
        pendingTimerId = null;
    }
}

function stopActiveTrialAudio() {
    if (activeTrialSource) {
        try {
            activeTrialSource.onended = null;
            activeTrialSource.stop();
            activeTrialSource.disconnect();
        } catch (_) {}
        activeTrialSource = null;
    }
    isWaitingForAudio = true;
    stimOffsetTimestampMs = null;
}

/* --- WEB AUDIO ENGINE & EAR ROUTING --- */

function ensureAudioGraph() {
    if (!audioCtx) {
        const Ctx = window.AudioContext || window.webkitAudioContext;
        audioCtx = new Ctx();
        if (audioCtx.destination.maxChannelCount >= 2) {
            audioCtx.destination.channelCount = 2;
            audioCtx.destination.channelCountMode = "explicit";
            audioCtx.destination.channelInterpretation = "speakers";
        }
        monoBus = audioCtx.createGain();
        monoBus.gain.value = 1.0;
        leftEarGain = audioCtx.createGain();
        rightEarGain = audioCtx.createGain();
        earMerger = audioCtx.createChannelMerger(2);

        monoBus.connect(leftEarGain);
        monoBus.connect(rightEarGain);
        leftEarGain.connect(earMerger, 0, 0);
        rightEarGain.connect(earMerger, 0, 1);
        earMerger.connect(audioCtx.destination);
    }
    if (audioCtx.state === "suspended") {
        audioCtx.resume();
    }
    applyEarRouting();
}

function applyEarRouting() {
    if (!leftEarGain || !rightEarGain || !audioCtx) return;
    const ear = earSelect.value;
    const now = audioCtx.currentTime;
    const lVal = (ear === "Left" || ear === "Both") ? 1.0 : 0.0;
    const rVal = (ear === "Right" || ear === "Both") ? 1.0 : 0.0;
    leftEarGain.gain.setValueAtTime(lVal, now);
    rightEarGain.gain.setValueAtTime(rVal, now);
}

/* Deterministic 16/24/32-bit PCM RIFF WAV decoder (with browser decodeAudioData fallback) */
async function decodeWavArrayBuffer(arrayBuf) {
    try {
        const view = new DataView(arrayBuf);
        const riff = String.fromCharCode(view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
        const wave = String.fromCharCode(view.getUint8(8), view.getUint8(9), view.getUint8(10), view.getUint8(11));
        if (riff === "RIFF" && wave === "WAVE") {
            let fmt = null;
            let dataOffset = -1;
            let dataSize = 0;
            let pos = 12;
            while (pos + 8 <= view.byteLength) {
                const cid = String.fromCharCode(
                    view.getUint8(pos), view.getUint8(pos + 1), view.getUint8(pos + 2), view.getUint8(pos + 3)
                );
                const csize = view.getUint32(pos + 4, true);
                if (cid === "fmt ") {
                    fmt = {
                        audioFormat: view.getUint16(pos + 8, true),
                        numChannels: view.getUint16(pos + 10, true),
                        sampleRate: view.getUint32(pos + 12, true),
                        bitsPerSample: view.getUint16(pos + 22, true)
                    };
                } else if (cid === "data") {
                    dataOffset = pos + 8;
                    dataSize = Math.min(csize, view.byteLength - dataOffset);
                    break;
                }
                pos += 8 + csize + (csize & 1);
            }

            if (fmt && dataOffset >= 0 && (fmt.audioFormat === 1 || fmt.audioFormat === 3)) {
                const bytesPerSample = fmt.bitsPerSample >> 3;
                const blockAlign = bytesPerSample * fmt.numChannels;
                const numFrames = Math.floor(dataSize / blockAlign);
                const audioBuf = audioCtx.createBuffer(fmt.numChannels, numFrames, fmt.sampleRate);
                const bytes = new Uint8Array(arrayBuf, dataOffset, numFrames * blockAlign);

                for (let ch = 0; ch < fmt.numChannels; ch++) {
                    const out = audioBuf.getChannelData(ch);
                    let bIdx = ch * bytesPerSample;
                    if (fmt.audioFormat === 1 && fmt.bitsPerSample === 24) {
                        for (let i = 0; i < numFrames; i++, bIdx += blockAlign) {
                            const s24 = ((bytes[bIdx] | (bytes[bIdx + 1] << 8) | (bytes[bIdx + 2] << 16)) << 8) >> 8;
                            out[i] = s24 / 8388608.0;
                        }
                    } else if (fmt.audioFormat === 1 && fmt.bitsPerSample === 16) {
                        for (let i = 0; i < numFrames; i++, bIdx += blockAlign) {
                            out[i] = view.getInt16(dataOffset + bIdx, true) / 32768.0;
                        }
                    } else if (fmt.audioFormat === 1 && fmt.bitsPerSample === 32) {
                        for (let i = 0; i < numFrames; i++, bIdx += blockAlign) {
                            out[i] = view.getInt32(dataOffset + bIdx, true) / 2147483648.0;
                        }
                    } else if (fmt.audioFormat === 3 && fmt.bitsPerSample === 32) {
                        for (let i = 0; i < numFrames; i++, bIdx += blockAlign) {
                            out[i] = view.getFloat32(dataOffset + bIdx, true);
                        }
                    }
                }
                return audioBuf;
            }
        }
    } catch (_) {
        /* Fall through to browser native decoder */
    }
    return await audioCtx.decodeAudioData(arrayBuf.slice(0));
}

async function loadAudioBuffer(relPath) {
    if (bufferCache.has(relPath)) {
        return bufferCache.get(relPath);
    }
    ensureAudioGraph();
    const resp = await fetch(relPath);
    if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} loading ${relPath}`);
    }
    const arr = await resp.arrayBuffer();
    const decoded = await decodeWavArrayBuffer(arr);
    bufferCache.set(relPath, decoded);
    return decoded;
}

async function preloadBuffersForRun(mode, withPractice) {
    const needed = [];
    if (withPractice || mode !== "standard_100") {
        for (const f of practiceFiles) {
            needed.push("Stimuli/practice/" + f);
        }
    }
    if (mode === "standard_100") {
        for (const f of testFilesBase) {
            needed.push("Stimuli/test/" + f);
        }
    } else {
        needed.push("Stimuli/ltass_noise.wav");
    }
    const unique = Array.from(new Set(needed));
    await Promise.all(unique.map(p => loadAudioBuffer(p)));
}

/* Mix clear 60-dB-SNR token + circular random LTASS slice at target SNR (dB) */
function buildMixedAdaptiveBuffer(clearBuf, noiseBuf, snrDb) {
    const sr = clearBuf.sampleRate;
    const len = clearBuf.length;
    const speech = clearBuf.getChannelData(0);
    const noise = noiseBuf.getChannelData(0);
    const nLen = noise.length;
    const offset = Math.floor(Math.random() * nLen);

    /* 10-ms raised-cosine onset/offset ramp on noise with exact RMS power compensation */
    const rampSamples = Math.min(Math.round(0.010 * sr), Math.floor(len / 4));
    let sumWinSq = len - 2 * rampSamples;
    for (let k = 0; k < rampSamples; k++) {
        const w = 0.5 * (1.0 - Math.cos((Math.PI * k) / rampSamples));
        sumWinSq += 2.0 * w * w;
    }
    const rmsComp = Math.sqrt(len / Math.max(sumWinSq, 1.0));
    const gNoise = Math.pow(10.0, -snrDb / 20.0) * rmsComp;

    const mixed = audioCtx.createBuffer(1, len, sr);
    const out = mixed.getChannelData(0);
    let peak = 0.0;

    for (let i = 0; i < len; i++) {
        let w = 1.0;
        if (i < rampSamples) {
            w = 0.5 * (1.0 - Math.cos((Math.PI * i) / rampSamples));
        } else if (i >= len - rampSamples) {
            const rem = len - 1 - i;
            w = 0.5 * (1.0 - Math.cos((Math.PI * rem) / rampSamples));
        }
        const s = speech[i] + gNoise * w * noise[(offset + i) % nLen];
        out[i] = s;
        const a = Math.abs(s);
        if (a > peak) peak = a;
    }

    if (peak > 0.98) {
        const lim = 0.98 / peak;
        for (let i = 0; i < len; i++) out[i] *= lim;
    }
    return mixed;
}

/* --- CALIBRATION --- */

async function startCalibration() {
    ensureAudioGraph();
    const selectedSource = document.querySelector('input[name="calib-source"]:checked').value;

    calibEnvGain = audioCtx.createGain();
    calibEnvGain.gain.setValueAtTime(0.0, audioCtx.currentTime);
    calibEnvGain.connect(monoBus);

    if (selectedSource === "tone") {
        const osc = audioCtx.createOscillator();
        osc.type = "sine";
        osc.frequency.value = 1000;
        const targetAmplitude = 0.01 * Math.SQRT2; /* 0.01 RMS == 60 dB SPL */
        calibEnvGain.gain.linearRampToValueAtTime(targetAmplitude, audioCtx.currentTime + 0.05);
        osc.connect(calibEnvGain);
        osc.start();
        calibSourceNode = osc;
    } else {
        try {
            const buf = await loadAudioBuffer("Stimuli/practice/VCV_aba_1_60SNR.wav");
            if (!calibEnvGain) return; /* Stopped while loading */
            const src = audioCtx.createBufferSource();
            src.buffer = buf;
            src.loop = true;
            calibEnvGain.gain.linearRampToValueAtTime(1.0, audioCtx.currentTime + 0.04);
            src.connect(calibEnvGain);
            src.start();
            calibSourceNode = src;
        } catch (e) {
            console.error("Calibration speech load failed:", e);
            return;
        }
    }

    calibBtn.innerText = "Stop Calibration Audio";
    calibBtn.classList.add("active");
    isCalibrating = true;
}

function stopCalibration() {
    if (!isCalibrating) return;
    if (audioCtx && calibEnvGain && calibSourceNode) {
        try {
            const now = audioCtx.currentTime;
            calibEnvGain.gain.cancelScheduledValues(now);
            calibEnvGain.gain.setValueAtTime(calibEnvGain.gain.value, now);
            calibEnvGain.gain.linearRampToValueAtTime(0.0, now + 0.04);
            calibSourceNode.stop(now + 0.045);
        } catch (_) {}
    }
    calibSourceNode = null;
    calibEnvGain = null;
    calibBtn.innerText = "Play Calibration Audio";
    calibBtn.classList.remove("active");
    isCalibrating = false;
}

/* --- INSTRUCTIONS POP-UP MODAL & PHASE START --- */

function readOptionalAudiogram() {
    const vals = new Array(10).fill(null);
    let count = 0;
    audGridEl.querySelectorAll("input").forEach(inp => {
        const raw = inp.value.trim();
        if (raw !== "" && Number.isFinite(Number(raw))) {
            vals[Number(inp.dataset.fidx)] = Number(raw);
            count++;
        }
    });
    return count >= 2 ? vals : null;
}

function openInstructionsModal(targetPhase) {
    pendingLaunchTarget = targetPhase;
    const nGraded = currentMode === "adaptive_50" ? 50 : 100;

    if (targetPhase === "PRACTICE") {
        instructionsTitle.innerText = "Practice Phase Instructions";
        instructionsList.innerHTML =
            "<li>You will hear <strong>10 clear speech sounds</strong> (&ldquo;a-consonant-a&rdquo;) without background noise.</li>" +
            "<li>Listen to each stimulus and select the consonant you heard from the grid.</li>" +
            "<li>After each choice, the target consonant will highlight in <strong>green</strong> as feedback.</li>" +
            "<li>If you&rsquo;re unsure, take your best guess. Select <strong>???</strong> only if you think you heard a consonant that is not one of the choices.</li>" +
            "<li>Click <strong>OK</strong> when you are ready to hear the first stimulus.</li>";
    } else {
        instructionsTitle.innerText = `Test Instructions (${nGraded} Trials)`;
        instructionsList.innerHTML =
            `<li>You will hear <strong>${nGraded} speech sounds</strong> in background noise (sometimes the noise will be louder than the speech).</li>` +
            "<li>No green feedback will be shown during the test.</li>" +
            "<li>Listen to each stimulus and select the consonant you heard.</li>" +
            "<li>If you&rsquo;re unsure, take your best guess. Select <strong>???</strong> only if you think you heard a consonant that is not one of the choices.</li>" +
            "<li>Click <strong>OK</strong> when you are ready to hear the first stimulus.</li>";
    }

    /* Show the consonant grid screen behind the modal (matching qCLS UX) */
    appContainer.classList.remove("two-column-mode");
    setupArea.classList.add("hidden");
    qcBanner.classList.add("hidden");
    startBtn.classList.add("hidden");
    repeatPracticeBtn.classList.add("hidden");
    interfaceDiv.classList.remove("hidden");
    isWaitingForAudio = true;

    if (targetPhase === "PRACTICE") {
        phaseTitle.innerText = "Practice Phase";
        totalPhaseTrials = 10;
    } else {
        const isAdaptive = currentMode !== "standard_100";
        phaseTitle.innerText = isAdaptive
            ? `Graded Test Phase (Adaptive-SNR, ${nGraded} Trials)`
            : "Graded Test Phase (Standard, 100 Trials)";
        totalPhaseTrials = nGraded;
    }
    currentTrialNumber = 0;
    updateProgressUI();

    instructionsModal.showModal();
    instructionsOkBtn.focus();
}

function cancelInstructionsModal() {
    if (instructionsModal.open) {
        instructionsModal.close();
    }
    const prevTarget = pendingLaunchTarget;
    pendingLaunchTarget = null;

    /* If cancelled from Step 3 (after Practice completed), return to Practice Complete screen; otherwise return to Setup */
    if (prevTarget === "GRADED" && state === "PRACTICE_DONE") {
        interfaceDiv.classList.add("hidden");
        qcBanner.classList.remove("hidden");
        startBtn.classList.remove("hidden");
        const hits = practiceResults.filter(r => r.correct).length;
        const pct = practiceResults.length > 0 ? Math.round((100 * hits) / practiceResults.length) : 100;
        if (pct < 80) repeatPracticeBtn.classList.remove("hidden");
        phaseTitle.innerText = "Practice Complete";
    } else {
        returnToSetupKeepSettings(false);
    }
}

function confirmInstructionsAndLaunch() {
    const target = pendingLaunchTarget;
    pendingLaunchTarget = null;
    if (instructionsModal.open) {
        instructionsModal.close();
    }
    if (target === "PRACTICE") {
        beginPracticePhase();
    } else if (target === "GRADED") {
        beginGradedPhase();
    }
}

async function requestPhaseStart(repeatPractice) {
    ensureAudioGraph();
    stopCalibration();

    if (state === "IDLE" || repeatPractice) {
        currentMode = modeSelect.value;
        const doPractice = repeatPractice ? true : includePracticeChk.checked;
        currentRunUsedPractice = doPractice;

        startBtn.disabled = true;
        const prevText = startBtn.innerText;
        startBtn.innerText = "Loading Calibrated Audio...";
        try {
            await preloadBuffersForRun(currentMode, doPractice);
        } catch (err) {
            console.error("Audio preload error:", err);
            alert("Failed to load audio stimuli. Ensure qVCV is served over HTTP/HTTPS (not file://).");
            startBtn.disabled = false;
            startBtn.innerText = prevText;
            return;
        }
        startBtn.disabled = false;
        startBtn.innerText = prevText;

        openInstructionsModal(doPractice ? "PRACTICE" : "GRADED");
    } else if (state === "PRACTICE_DONE") {
        openInstructionsModal("GRADED");
    }
}

function beginPracticePhase() {
    clearPendingTimer();
    stopActiveTrialAudio();
    isPaused = false;
    state = "PRACTICE";
    practiceResults = [];
    phaseTitle.innerText = "Practice Phase";
    statusMsg.innerHTML =
        "Listen to the stimulus and select the consonant you heard.<br>" +
        "If you\u2019re unsure, take your best guess. Select <strong>???</strong> only if you heard a consonant not shown.";

    const practiceSelection = practiceFiles
        .filter(f => f.includes("_1_"))
        .map(f => {
            const c = window.QVCVEngine.parseConsonantFromFilename(f);
            return {
                consonant: c,
                consonantIdx: stimuliConsonants.indexOf(c),
                tokenIdx: 0,
                stimulus: f,
                clearFile: f,
                snrDb: 60.0
            };
        });
    currentQueue = shuffle([...practiceSelection]);
    totalPhaseTrials = currentQueue.length;
    currentTrialNumber = 0;
    updateProgressUI();
    advanceToNextTrial();
}

function beginGradedPhase() {
    clearPendingTimer();
    stopActiveTrialAudio();
    isPaused = false;
    state = "GRADED";
    trialResults = [];
    resetConfusionMatrix();

    const isAdaptive = currentMode !== "standard_100";
    const nTrials = currentMode === "adaptive_50" ? 50 : 100;

    phaseTitle.innerText = isAdaptive
        ? `Graded Test Phase (Adaptive-SNR, ${nTrials} Trials)`
        : "Graded Test Phase (Standard, 100 Trials)";
    statusMsg.innerHTML =
        "Test in progress (background noise added; no feedback).<br>" +
        "Listen to the stimulus and select the consonant you heard, or take your best guess.";

    if (isAdaptive) {
        adaptiveTracker = window.QVCVEngine.createAdaptiveTracker();
        currentQueue = window.QVCVEngine.buildAdaptiveSchedule(nTrials);
    } else {
        adaptiveTracker = null;
        currentQueue = window.QVCVEngine.buildStandardFixedSchedule(testFilesBase, 5);
    }

    totalPhaseTrials = currentQueue.length;
    currentTrialNumber = 0;
    updateProgressUI();
    advanceToNextTrial();
}

/* --- MID-TEST PAUSE / RESUME / RESTART (KEEPING ALL SETTINGS) --- */

function openPauseModal() {
    if (state !== "PRACTICE" && state !== "GRADED") return;
    isPaused = true;
    clearPendingTimer();
    stopActiveTrialAudio();
    resetButtons();

    const phaseLabel = state === "PRACTICE" ? "Practice Phase" : "Graded Test";
    const trNum = Math.max(1, currentTrialNumber);
    pauseModalStatus.innerHTML =
        `<strong>${phaseLabel} &mdash; Trial ${trNum} of ${totalPhaseTrials}</strong><br>` +
        `Playback is stopped. All participant, ear, mode, and audiogram settings are preserved.`;
    resumeTestBtn.innerText = `Resume Test (Replay Trial ${trNum})`;
    pauseModal.showModal();
    resumeTestBtn.focus();
}

function resumePausedTest() {
    if (!isPaused) return;
    if (pauseModal.open) pauseModal.close();
    isPaused = false;
    if (currentTrialSpec) {
        playCurrentTrialSpec(450);
    } else {
        advanceToNextTrial();
    }
}

function restartFromTrialOneKeepSettings() {
    if (pauseModal.open) pauseModal.close();
    isPaused = false;
    clearPendingTimer();
    stopActiveTrialAudio();
    resetButtons();

    /* Restart from the beginning of the run (Practice if enabled on this run, else Graded) with all settings intact */
    state = "IDLE";
    practiceResults = [];
    trialResults = [];
    resetConfusionMatrix();
    openInstructionsModal(currentRunUsedPractice ? "PRACTICE" : "GRADED");
}

function returnToSetupKeepSettings(completedRun) {
    clearPendingTimer();
    stopActiveTrialAudio();
    stopCalibration();
    resetButtons();
    isPaused = false;
    pendingLaunchTarget = null;
    state = "IDLE";

    resultsArea.classList.add("hidden");
    interfaceDiv.classList.add("hidden");
    qcBanner.classList.add("hidden");
    repeatPracticeBtn.classList.add("hidden");

    appContainer.classList.add("two-column-mode");
    setupArea.classList.remove("hidden");
    phaseTitle.innerText = "qVCV Test";
    statusMsg.innerText = completedRun
        ? "Ready for next run. Participant and protocol settings are preserved."
        : "Test stopped. All participant and protocol settings are preserved.";
    startBtn.innerText = "Step 2: Ready to Begin Test";
    startBtn.classList.remove("hidden");

    if (completedRun && practiceResults.length >= 5) {
        includePracticeChk.checked = false;
    }
}

/* --- TRIAL PRESENTATION & RESPONSE HANDLING --- */

function updateProgressUI() {
    progressBar.max = totalPhaseTrials;
    progressBar.value = currentTrialNumber;
    currentTrialSpan.innerText = currentTrialNumber === 0 ? 1 : currentTrialNumber;
    totalTrialSpan.innerText = totalPhaseTrials;
}

function advanceToNextTrial() {
    if (isPaused) return;
    resetButtons();
    isWaitingForAudio = true;
    stimOffsetTimestampMs = null;

    if (currentQueue.length === 0) {
        endPhase();
        return;
    }

    const spec = currentQueue.shift();
    currentTrialNumber++;
    updateProgressUI();

    if (state === "GRADED" && adaptiveTracker) {
        const snrDb = adaptiveTracker.selectSnrForConsonant(spec.consonantIdx);
        const snrTag = snrDb >= 0 ? `+${snrDb}SNR` : `${snrDb}SNR`;
        spec.snrDb = snrDb;
        spec.stimulus = `VCV_a${spec.consonant.toLowerCase()}a_${spec.tokenIdx + 1}_${snrTag}.wav`;
    }
    currentTrialSpec = spec;
    playCurrentTrialSpec(450);
}

function playCurrentTrialSpec(delayMs) {
    clearPendingTimer();
    stopActiveTrialAudio();
    const spec = currentTrialSpec;
    if (!spec || isPaused) return;

    pendingTimerId = setTimeout(async () => {
        pendingTimerId = null;
        if (isPaused || !currentTrialSpec) return;

        try {
            ensureAudioGraph();
            let playBuf = null;

            if (state === "PRACTICE") {
                playBuf = await loadAudioBuffer("Stimuli/practice/" + spec.clearFile);
            } else if (adaptiveTracker) {
                const clearBuf = await loadAudioBuffer("Stimuli/practice/" + spec.clearFile);
                const noiseBuf = await loadAudioBuffer("Stimuli/ltass_noise.wav");
                playBuf = buildMixedAdaptiveBuffer(clearBuf, noiseBuf, spec.snrDb);
            } else {
                playBuf = await loadAudioBuffer("Stimuli/test/" + spec.stimulus);
            }

            if (isPaused) return;

            const src = audioCtx.createBufferSource();
            src.buffer = playBuf;
            src.connect(monoBus);
            activeTrialSource = src;

            src.onended = () => {
                if (activeTrialSource !== src || isPaused) return;
                activeTrialSource = null;
                stimOffsetTimestampMs = performance.now();
                isWaitingForAudio = false;
            };
            src.start();
        } catch (err) {
            console.error("Stimulus playback failed:", err);
            if (!isPaused) {
                stimOffsetTimestampMs = performance.now();
                isWaitingForAudio = false;
            }
        }
    }, delayMs);
}

function handleResponse(selected) {
    if (isPaused || isWaitingForAudio || !currentTrialSpec) return;

    const rtMs = stimOffsetTimestampMs !== null
        ? Math.max(0, Math.round(performance.now() - stimOffsetTimestampMs))
        : null;
    const actual = currentTrialSpec.consonant;
    const isCorrect = selected === actual;
    isWaitingForAudio = true;

    if (state === "PRACTICE") {
        practiceResults.push({
            stimulus: currentTrialSpec.stimulus,
            target: actual,
            response: selected,
            snrDb: 60,
            rtMs,
            correct: isCorrect
        });
        document.querySelectorAll(".consonant-btn").forEach(btn => {
            if (btn.dataset.consonant === actual) {
                btn.classList.add("correct-highlight");
            }
        });
        clearPendingTimer();
        pendingTimerId = setTimeout(() => {
            pendingTimerId = null;
            advanceToNextTrial();
        }, 1400);
    } else if (state === "GRADED") {
        let runningLossDb = null;
        let runningSeDb = null;
        if (adaptiveTracker) {
            const st = adaptiveTracker.updateWithTrial(
                currentTrialSpec.consonantIdx,
                currentTrialSpec.tokenIdx,
                currentTrialSpec.snrDb,
                isCorrect
            );
            runningLossDb = st.nhReferencedShiftDb;
            runningSeDb = st.globalSeDb;
        }

        trialResults.push({
            consonantIdx: currentTrialSpec.consonantIdx,
            tokenIdx: currentTrialSpec.tokenIdx,
            stimulus: currentTrialSpec.stimulus,
            target: actual,
            response: selected,
            snrDb: currentTrialSpec.snrDb,
            rtMs,
            correct: isCorrect,
            runningLossDb,
            runningSeDb
        });
        confusionMatrix[actual][selected]++;
        clearPendingTimer();
        pendingTimerId = setTimeout(() => {
            pendingTimerId = null;
            advanceToNextTrial();
        }, 420);
    }
}

function endPhase() {
    clearPendingTimer();
    stopActiveTrialAudio();
    currentTrialSpec = null;
    interfaceDiv.classList.add("hidden");

    if (state === "PRACTICE") {
        state = "PRACTICE_DONE";
        const hits = practiceResults.filter(r => r.correct).length;
        const tot = practiceResults.length;
        const pct = tot > 0 ? Math.round((100 * hits) / tot) : 0;
        const nTest = currentMode === "adaptive_50" ? 50 : 100;

        phaseTitle.innerText = "Practice Complete";
        statusMsg.innerText =
            `You have completed the practice phase. Click Step 3 to view the test instructions and begin (${nTest} trials in background noise).`;

        qcBanner.classList.remove("hidden", "qc-warn", "qc-ok");
        if (pct < 80) {
            qcBanner.classList.add("qc-warn");
            qcBanner.innerHTML =
                `<strong>Booth QC Check \u2014 Practice Score: ${hits} / ${tot} (${pct}%) at 60 dB SNR.</strong><br>` +
                `Below the 80% clear-speech reference. Verify ear routing, transducer seal, calibration level, and task instructions, or click <em>Repeat Practice</em>.`;
            repeatPracticeBtn.classList.remove("hidden");
        } else {
            qcBanner.classList.add("qc-ok");
            qcBanner.innerHTML = `<strong>Practice Score: ${hits} / ${tot} (${pct}%)</strong> \u2014 Clear-speech audibility &amp; task check passed.`;
            repeatPracticeBtn.classList.add("hidden");
        }

        startBtn.innerText = `Step 3: Begin Graded Test (${nTest} Trials)`;
        startBtn.classList.remove("hidden");
    } else if (state === "GRADED") {
        state = "FINISHED";
        phaseTitle.innerText = "Test Complete";
        const ear = earSelect.value;
        const cond = condSelect.value;
        const pId = document.getElementById("participant-id").value.trim() || "UNKNOWN";
        statusMsg.innerText = `Participant: ${pId} \u00b7 Ear: ${ear} \u00b7 Condition: ${cond}`;
        resultsArea.classList.remove("hidden");
        renderClinicalSummary();
        renderHeatmap();
    }
}

function resetButtons() {
    document.querySelectorAll(".consonant-btn").forEach(btn => {
        btn.classList.remove("correct-highlight");
    });
}

function shuffle(array) {
    let currentIndex = array.length;
    while (currentIndex !== 0) {
        const randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        const tmp = array[currentIndex];
        array[currentIndex] = array[randomIndex];
        array[randomIndex] = tmp;
    }
    return array;
}

/* --- RESULTS SUMMARY & HEATMAP --- */

function buildCounts10x11() {
    return stimuliConsonants.map(s => displayConsonants.map(r => confusionMatrix[s][r] || 0));
}

function fmtSignedDb(v) {
    if (!Number.isFinite(v)) return "--";
    return (v >= 0 ? "+" : "") + v.toFixed(1);
}

function renderClinicalSummary() {
    const counts10x11 = buildCounts10x11();
    const isAdaptive = currentMode !== "standard_100";
    const audHL = readOptionalAudiogram();

    const summary = window.QVCVEngine.summarizeSession({
        trials: trialResults,
        counts10x11,
        isAdaptive,
        audiogramHL: audHL
    });
    lastSessionSummary = summary;

    const pracHits = practiceResults.filter(r => r.correct).length;
    const pracStr = practiceResults.length > 0
        ? `Practice: ${pracHits}/${practiceResults.length} (${Math.round((100 * pracHits) / practiceResults.length)}%)`
        : "Practice: skipped";
    const rtStr = summary.medianRtMs !== null ? `Median RT: ${summary.medianRtMs} ms` : "";

    let pillClass = "pill-ok";
    let pillText = "Normal Range";
    if (summary.globalSnrLossDb > 6.0) {
        pillClass = "pill-high";
        pillText = "Marked Deficit";
    } else if (summary.globalSnrLossDb > 2.6) {
        pillClass = "pill-mild";
        pillText = "Elevated";
    }

    let card3Html = "";
    if (summary.hasAudiogram) {
        card3Html = `
            <div class="summary-card">
                <div class="card-label">Plomp Audibility vs. Clarity</div>
                <div class="card-value">${fmtSignedDb(summary.supraDeficitDb)} &plusmn; ${summary.supraSeDb.toFixed(1)} dB</div>
                <div class="card-sub">
                    Suprathreshold Clarity Deficit<br>
                    Audibility Component: <strong>+${summary.audibilityComponentDb.toFixed(1)} dB</strong> (PTA: ${summary.pta124Db.toFixed(0)} dB HL)
                </div>
            </div>`;
    } else if (!isAdaptive && summary.pptaDbHL !== null) {
        card3Html = `
            <div class="summary-card">
                <div class="card-label">Predicted PTA (PPTA)</div>
                <div class="card-value">${summary.pptaDbHL.toFixed(1)} dB HL</div>
                <div class="card-sub">
                    Multinomial confusion-matrix estimate<br>
                    (Hajicek, Harris, &amp; Neely, 2026)
                </div>
            </div>`;
    } else {
        card3Html = `
            <div class="summary-card">
                <div class="card-label">Mean Presented SNR</div>
                <div class="card-value">${fmtSignedDb(summary.meanSnrDb)} dB SNR</div>
                <div class="card-sub">
                    Tracked near ~56% steepest slope<br>
                    (Enter Audiogram on Setup to split Audibility)
                </div>
            </div>`;
    }

    document.getElementById("summary-cards").innerHTML = `
        <div class="summary-card">
            <div class="card-label">${isAdaptive ? "Adaptive-SNR Accuracy" : "Fixed-SNR Accuracy"}</div>
            <div class="card-value">${summary.nCorrect} / ${summary.nTrials} (${summary.pctCorrect.toFixed(0)}%)</div>
            <div class="card-sub">${pracStr}${rtStr ? " &middot; " + rtStr : ""}</div>
        </div>
        <div class="summary-card">
            <div class="card-label">Speech-in-Noise Loss (vs. NH)</div>
            <div class="card-value">
                ${fmtSignedDb(summary.globalSnrLossDb)} &plusmn; ${summary.globalSeDb.toFixed(1)} dB
                <span class="status-pill ${pillClass}">${pillText}</span>
            </div>
            <div class="card-sub">
                LF Voiced/Nasal: <strong>${fmtSignedDb(summary.lfShiftDb)} dB</strong> &middot;
                HF Sibilant/Stop: <strong>${fmtSignedDb(summary.hfShiftDb)} dB</strong>
            </div>
        </div>
        ${card3Html}
    `;

    if (summary.features) {
        const f = summary.features;
        document.getElementById("feature-strip").innerHTML =
            `<strong>Phonetic Feature Information Transmitted (Miller-Nicely):</strong> &nbsp; ` +
            `Voicing: <strong>${f.voicing.infoPct.toFixed(0)}%</strong> (${f.voicing.accPct.toFixed(0)}% hit) &nbsp;|&nbsp; ` +
            `Manner: <strong>${f.manner.infoPct.toFixed(0)}%</strong> (${f.manner.accPct.toFixed(0)}% hit) &nbsp;|&nbsp; ` +
            `Place: <strong>${f.place.infoPct.toFixed(0)}%</strong> (${f.place.accPct.toFixed(0)}% hit)`;
    }
}

function renderHeatmap() {
    const container = document.getElementById("heatmap-container");
    let html = "<table><tr><th>Actual \\ Resp</th>";
    displayConsonants.forEach(c => { html += `<th>${c}</th>`; });
    html += "</tr>";

    stimuliConsonants.forEach(actual => {
        let rowTot = 0;
        displayConsonants.forEach(r => { rowTot += confusionMatrix[actual][r]; });
        const denom = Math.max(rowTot, 1);

        html += `<tr><td><strong>${actual}</strong></td>`;
        displayConsonants.forEach(response => {
            const count = confusionMatrix[actual][response];
            const frac = count / denom;
            const isDiag = actual === response;
            const bg = count === 0
                ? "transparent"
                : (isDiag
                    ? `rgba(46, 160, 95, ${(0.14 + 0.72 * frac).toFixed(3)})`
                    : `rgba(235, 95, 20, ${(0.16 + 0.78 * frac).toFixed(3)})`);
            const cellText = heatmapView === "pct"
                ? (count === 0 ? "0" : `${Math.round(100 * frac)}%`)
                : String(count);
            html += `<td style="background-color: ${bg}">${cellText}</td>`;
        });
        html += "</tr>";
    });

    html += "</table>";
    container.innerHTML = html;
}

/* --- CSV EXPORT (Date + HHMM Timestamp on Both Fixed & Adaptive Runs) --- */

function downloadCSV() {
    let pId = document.getElementById("participant-id").value.trim() || "UNKNOWN";
    pId = pId.replace(/[^A-Za-z0-9_-]/g, "");
    const earFull = earSelect.value;
    const earCode = earFull.charAt(0); /* L, R, or B */
    const cond = condSelect.value;     /* Unaided or Aided */

    const finalLossStr = (lastSessionSummary && Number.isFinite(lastSessionSummary.globalSnrLossDb))
        ? lastSessionSummary.globalSnrLossDb.toFixed(3)
        : "";

    const rows = [
        "Trial,Phase,Stimulus,Target,Response,Target_SNR_dB,RT_ms,Running_SNR_Loss_dB,Running_SE_dB,Final_SNR_Loss_dB,Mode,Condition,Ear"
    ];

    practiceResults.forEach((r, idx) => {
        const rt = r.rtMs !== null ? r.rtMs : "";
        rows.push(
            `${idx + 1},Practice,${r.stimulus},${r.target},${r.response},60,${rt},,,,${currentMode},${cond},${earFull}`
        );
    });

    trialResults.forEach((r, idx) => {
        const rt = r.rtMs !== null ? r.rtMs : "";
        const runLoss = r.runningLossDb !== null && Number.isFinite(r.runningLossDb) ? r.runningLossDb.toFixed(3) : "";
        const runSe = r.runningSeDb !== null && Number.isFinite(r.runningSeDb) ? r.runningSeDb.toFixed(3) : "";
        rows.push(
            `${idx + 1},Graded,${r.stimulus},${r.target},${r.response},${r.snrDb},${rt},${runLoss},${runSe},${finalLossStr},${currentMode},${cond},${earFull}`
        );
    });

    const date = new Date();
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, "0");
    const dd = String(date.getDate()).padStart(2, "0");
    const hh = String(date.getHours()).padStart(2, "0");
    const min = String(date.getMinutes()).padStart(2, "0");
    const dateTimeStr = `${yyyy}${mm}${dd}_${hh}${min}`;

    /* Include date + HHMM timestamp on ALL tests (both Fixed Standard and Adaptive):
     *   Standard Unaided : qVCV_NH001_L_20261119_1105.csv
     *   Adaptive Unaided : qVCV_NH001_L_ADP50_20261119_1105.csv
     *   Aided            : qVCV_NH001_L_AIDED_20261119_1105.csv */
    const tagParts = [];
    if (cond === "Aided") tagParts.push("AIDED");
    if (currentMode === "adaptive_50") tagParts.push("ADP50");
    else if (currentMode === "adaptive_100") tagParts.push("ADP100");
    const extraTag = tagParts.length > 0 ? `_${tagParts.join("_")}` : "";
    const filename = `qVCV_${pId}_${earCode}${extraTag}_${dateTimeStr}.csv`;

    const blob = new Blob([rows.join("\n") + "\n"], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    setTimeout(() => URL.revokeObjectURL(url), 1500);
}
