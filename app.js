const stimuliConsonants = ["B", "D", "G", "K", "N", "S", "SH", "T", "V", "Z"];
const displayConsonants = ["B", "D", "G", "K", "N", "S", "SH", "T", "V", "Z", "???"];

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

let state = "IDLE"; 
let currentQueue = [];
let currentStimulus = null;
let trialResults = [];
let confusionMatrix = {};
let isWaitingForAudio = false;

// Progress tracking
let totalPhaseTrials = 0;
let currentTrialNumber = 0;

// Web Audio API context (Tone)
let audioCtx;
let calibOsc;
let calibGain;

// HTML Audio context (Speech Stream - v3 style)
let calibSpeechAudio = new Audio("Stimuli/practice/VCV_aba_1_60SNR.wav");
calibSpeechAudio.loop = true;

let isCalibrating = false;

// Elements
const grid = document.getElementById('consonant-grid');
const startBtn = document.getElementById('start-btn');
const interfaceDiv = document.getElementById('interface');
const resultsArea = document.getElementById('results-area');
const statusMsg = document.getElementById('status-msg');
const phaseTitle = document.getElementById('phase-title');
const downloadBtn = document.getElementById('download-btn');
const setupArea = document.getElementById('setup-area');
const calibBtn = document.getElementById('calib-btn');
const progressBar = document.getElementById('test-progress');
const currentTrialSpan = document.getElementById('trial-current');
const totalTrialSpan = document.getElementById('trial-total');
const calibRadios = document.getElementsByName('calib-source');

// Initialize Data Structure
stimuliConsonants.forEach(s => {
    confusionMatrix[s] = {};
    displayConsonants.forEach(r => {
        confusionMatrix[s][r] = 0;
    });
});

// Initialize Grid
displayConsonants.forEach(c => {
    if (c === "???") {
        const spacer = document.createElement('div');
        spacer.className = "grid-spacer";
        grid.appendChild(spacer);
    }

    const btn = document.createElement('button');
    btn.innerText = c;
    btn.className = "consonant-btn";
    btn.dataset.consonant = c;
    btn.onclick = () => handleResponse(c);
    grid.appendChild(btn);
});

// Event Listeners
startBtn.addEventListener('click', startPhase);
downloadBtn.addEventListener('click', downloadCSV);

// Stop calibration automatically if they switch options while it's playing
calibRadios.forEach(radio => {
    radio.addEventListener('change', () => {
        if (isCalibrating) {
            stopCalibration();
        }
    });
});

calibBtn.addEventListener('click', () => {
    if (!isCalibrating) {
        startCalibration();
    } else {
        stopCalibration();
    }
});

function startCalibration() {
    const selectedSource = document.querySelector('input[name="calib-source"]:checked').value;

    if (selectedSource === 'tone') {
        if (!audioCtx) {
            audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        }
        calibOsc = audioCtx.createOscillator();
        calibGain = audioCtx.createGain();

        calibOsc.type = 'sine';
        calibOsc.frequency.value = 1000; // 1 kHz Tone
        
        const targetAmplitude = 0.01 * Math.SQRT2; // 0.01 RMS

        // 50 ms onset ramp
        calibGain.gain.setValueAtTime(0, audioCtx.currentTime);
        calibGain.gain.linearRampToValueAtTime(targetAmplitude, audioCtx.currentTime + 0.05);

        calibOsc.connect(calibGain);
        calibGain.connect(audioCtx.destination);
        calibOsc.start();

    } else if (selectedSource === 'speech') {
        calibSpeechAudio.play().catch(e => {
            console.error("Audio playback blocked", e);
        });
    }

    calibBtn.innerText = "Stop Calibration Audio";
    calibBtn.classList.add('active');
    isCalibrating = true;
}

function stopCalibration() {
    if (!isCalibrating) return;

    // Handle Tone offset
    if (audioCtx && calibGain && calibOsc) {
        calibGain.gain.setValueAtTime(calibGain.gain.value, audioCtx.currentTime);
        calibGain.gain.linearRampToValueAtTime(0, audioCtx.currentTime + 0.05);
        calibOsc.stop(audioCtx.currentTime + 0.05);
    }
    
    // Handle Speech offset (v3 style)
    calibSpeechAudio.pause();
    calibSpeechAudio.currentTime = 0;
    
    calibBtn.innerText = "Play Calibration Audio";
    calibBtn.classList.remove('active');
    isCalibrating = false;
}

function startPhase() {
    // Standard audio unlock for browsers (prevents Autoplay blocks if user clicks start too fast)
    const unlockAudio = new Audio();
    unlockAudio.play().catch(e => {});

    // Revert to single column layout for the actual testing grid
    document.getElementById('app-container').classList.remove('two-column-mode');
    
    stopCalibration();

    startBtn.classList.add('hidden');
    setupArea.classList.add('hidden');
    interfaceDiv.classList.remove('hidden');

    // Reset button styles for later phases
    startBtn.style.cssText = "background-color: #fff; border: 1px solid #ccc; padding: 10px 20px; font-weight: bold; font-size: 18px; text-align: center; color: #555; text-decoration: none;";

    if (state === "IDLE") {
        state = "PRACTICE";
        phaseTitle.innerText = "Practice Phase";
        statusMsg.innerHTML = "Listen to the stimulus and select the consonant you heard.<br>If you’re unsure, take your best guess.<br>Select <strong>???</strong> only if you think you heard a consonant that is not one of the choices.";
        
        let practiceSelection = practiceFiles.filter(f => f.includes('_1_'));
        currentQueue = shuffle([...practiceSelection]);
        
        totalPhaseTrials = currentQueue.length;
        currentTrialNumber = 0;
        updateProgressUI();

        setupNextTrial();
    } else if (state === "PRACTICE_DONE") {
        state = "GRADED";
        phaseTitle.innerText = "Graded Test Phase";
        statusMsg.innerHTML = "Test in progress. Noise has been added. No feedback will be provided.<br><br>Listen to the stimulus and select the consonant you heard.<br>If you’re unsure, take your best guess.<br>Select <strong>???</strong> only if you think you heard a consonant that is not one of the choices.";
        
        let fullTestArray = [];
        for (let i = 0; i < 5; i++) {
            let shuffledBlock = shuffle([...testFilesBase]);
            fullTestArray = fullTestArray.concat(shuffledBlock);
        }
        currentQueue = fullTestArray;

        totalPhaseTrials = currentQueue.length;
        currentTrialNumber = 0;
        updateProgressUI();

        setupNextTrial();
    }
}

function updateProgressUI() {
    progressBar.max = totalPhaseTrials;
    progressBar.value = currentTrialNumber;
    currentTrialSpan.innerText = currentTrialNumber === 0 ? 1 : currentTrialNumber; 
    totalTrialSpan.innerText = totalPhaseTrials;
}

function setupNextTrial() {
    resetButtons();
    isWaitingForAudio = true; 

    if (currentQueue.length === 0) {
        endPhase();
        return;
    }
    
    currentStimulus = currentQueue.shift();
    currentTrialNumber++;
    updateProgressUI();
    
    setTimeout(() => {
        const folder = state === "PRACTICE" ? "Stimuli/practice/" : "Stimuli/test/";
        const audio = new Audio(folder + currentStimulus);
        
        audio.onended = () => { isWaitingForAudio = false; };
        audio.onerror = () => { 
            console.warn("Audio missing: " + currentStimulus); 
            isWaitingForAudio = false; 
        };
        
        audio.play().catch(e => {
            console.error("Audio failed", e);
            isWaitingForAudio = false;
        });
    }, 500);
}

function handleResponse(selected) {
    if (isWaitingForAudio) return;

    const actual = parseConsonant(currentStimulus);
    isWaitingForAudio = true; 

    if (state === "PRACTICE") {
        const buttons = document.querySelectorAll('.consonant-btn');
        buttons.forEach(btn => {
            if (btn.dataset.consonant === actual) {
                btn.classList.add('correct-highlight');
            }
        });
        setTimeout(setupNextTrial, 1500); 
    } else if (state === "GRADED") {
        trialResults.push({ stimulus: currentStimulus, target: actual, response: selected });
        confusionMatrix[actual][selected]++;
        setTimeout(setupNextTrial, 500);
    }
}

function endPhase() {
    interfaceDiv.classList.add('hidden');
    
    if (state === "PRACTICE") {
        state = "PRACTICE_DONE";
        phaseTitle.innerText = "Practice Complete";
        statusMsg.innerText = "You have completed the practice phase. The graded test will now begin. There will be 100 trials with noise added, and no feedback will be provided.";
        startBtn.innerText = "Step 3: Begin Graded Test";
        startBtn.classList.remove('hidden');
    } else if (state === "GRADED") {
        state = "FINISHED";
        phaseTitle.innerText = "Test Complete";
        statusMsg.innerText = "Thank you. The test is now complete.";
        resultsArea.classList.remove('hidden');
        renderHeatmap();
    }
}

function resetButtons() {
    const buttons = document.querySelectorAll('.consonant-btn');
    buttons.forEach(btn => {
        btn.classList.remove('correct-highlight');
    });
}

function parseConsonant(filename) {
    const part = filename.split('_')[1]; 
    return part.substring(1, part.length - 1).toUpperCase(); 
}

function shuffle(array) {
    let currentIndex = array.length, randomIndex;
    while (currentIndex !== 0) {
        randomIndex = Math.floor(Math.random() * currentIndex);
        currentIndex--;
        [array[currentIndex], array[randomIndex]] = [array[randomIndex], array[currentIndex]];
    }
    return array;
}

function renderHeatmap() {
    const container = document.getElementById('heatmap-container');
    let html = '<table><tr><th>Actual \ Resp</th>';
    
    displayConsonants.forEach(c => html += `<th>${c}</th>`);
    html += '</tr>';

    stimuliConsonants.forEach(actual => {
        html += `<tr><td><strong>${actual}</strong></td>`;
        displayConsonants.forEach(response => {
            const count = confusionMatrix[actual][response];
            const intensity = count / 10; 
            const color = `rgba(255, 100, 0, ${intensity})`;
            html += `<td style="background-color: ${color}">${count}</td>`;
        });
        html += '</tr>';
    });
    
    html += '</table>';
    container.innerHTML = html;
}

function downloadCSV() {
    let csvContent = "data:text/csv;charset=utf-8,Trial,Phase,Stimulus,Target,Response\n";
    trialResults.forEach((r, index) => {
        csvContent += `${index + 1},Graded,${r.stimulus},${r.target},${r.response}\n`;
    });
    
    // Dynamic Filename Logic: qVCV_NH001_L_20260511.csv
    let pId = document.getElementById('participant-id').value.trim();
    if (!pId) {
        pId = "UNKNOWN"; // Fallback if left blank
    }
    
    const ear = document.getElementById('test-ear').value.charAt(0); // L, R, or B
    
    const date = new Date();
    const yyyy = date.getFullYear();
    const mm = String(date.getMonth() + 1).padStart(2, '0');
    const dd = String(date.getDate()).padStart(2, '0');
    const dateString = `${yyyy}${mm}${dd}`;
    
    const filename = `qVCV_${pId}_${ear}_${dateString}.csv`;
    
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", filename);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}
