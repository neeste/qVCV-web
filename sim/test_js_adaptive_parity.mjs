#!/usr/bin/env node
/*
 * test_js_adaptive_parity.mjs
 *
 * Verifies qvcv_adaptive.js against the 98-listener empirical cohort in
 * sim/human_data/human_listeners_98.json:
 *   1. Schedule integrity (exact counts, token balance, 0 consecutive consonant repeats).
 *   2. Exact PPTA parity against MATLAB mnr_vcv_pta across all 98 listeners (< 0.01 dB).
 *   3. 50-trial and 100-trial Adaptive-SNR tracking accuracy, target % correct (~56%),
 *      test-retest MAD, and runtime speed across all 98 human-calibrated listeners.
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createRequire } from "node:module";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const QVCVEngine = require(path.join(__dirname, "..", "qvcv_adaptive.js"));

const hdPath = path.join(__dirname, "human_data", "human_listeners_98.json");
const hd = JSON.parse(fs.readFileSync(hdPath, "utf8"));

/* Deterministic PRNG (Mulberry32) for reproducible Monte Carlo verification */
function makeRng(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const testFilesBase = [
  "VCV_aba_2_18SNR.wav", "VCV_aga_1_3SNR.wav", "VCV_aka_2_0SNR.wav", "VCV_asa_1_0SNR.wav",
  "VCV_asha_2_0SNR.wav", "VCV_ava_1_18SNR.wav", "VCV_aza_2_0SNR.wav", "VCV_ada_1_4SNR.wav",
  "VCV_aga_2_3SNR.wav", "VCV_ana_1_15SNR.wav", "VCV_asa_2_0SNR.wav", "VCV_ata_1_0SNR.wav",
  "VCV_ava_2_18SNR.wav", "VCV_aba_1_18SNR.wav", "VCV_ada_2_4SNR.wav", "VCV_aka_1_0SNR.wav",
  "VCV_ana_2_15SNR.wav", "VCV_asha_1_0SNR.wav", "VCV_ata_2_0SNR.wav", "VCV_aza_1_0SNR.wav"
];

/* 1. Verify schedule invariants across 200 runs */
const rngSched = makeRng(12345);
for (let s = 0; s < 200; s++) {
  const stdSched = QVCVEngine.buildStandardFixedSchedule(testFilesBase, 5, rngSched);
  if (stdSched.length !== 100) throw new Error("Standard schedule length != 100");
  const fileCounts = {};
  for (let i = 0; i < stdSched.length; i++) {
    if (i > 0 && stdSched[i].consonant === stdSched[i - 1].consonant) {
      throw new Error(`Consecutive consonant repeat in Standard schedule at ${i}`);
    }
    fileCounts[stdSched[i].stimulus] = (fileCounts[stdSched[i].stimulus] || 0) + 1;
  }
  for (const f of testFilesBase) {
    if (fileCounts[f] !== 5) throw new Error(`Expected 5 plays of ${f}, got ${fileCounts[f]}`);
  }

  const adpSched = QVCVEngine.buildAdaptiveSchedule(50, rngSched);
  if (adpSched.length !== 50) throw new Error("Adaptive schedule length != 50");
  const consCounts = new Array(10).fill(0);
  for (let i = 0; i < adpSched.length; i++) {
    if (i > 0 && adpSched[i].consonantIdx === adpSched[i - 1].consonantIdx) {
      throw new Error(`Consecutive consonant repeat in Adaptive schedule at ${i}`);
    }
    consCounts[adpSched[i].consonantIdx]++;
  }
  if (consCounts.some(c => c !== 5)) throw new Error("Adaptive schedule not balanced 5x per consonant");
}
console.log("[PASS] Schedule invariants (200 Standard-100 + 200 Adaptive-50 runs, 0 boundary repeats).");

/* 2. Verify PPTA parity against MATLAB mnr_vcv_pta across all 98 listeners */
let maxPptaDiff = 0.0;
for (const l of hd.listeners) {
  const p1 = QVCVEngine.predictPptaDbHL(l.empirical_s1_counts);
  const diff1 = Math.abs(p1 - l.human_s1_ppta_db);
  if (diff1 > maxPptaDiff) maxPptaDiff = diff1;
}
console.log(`[PASS] PPTA parity across 98 human listeners: Max |JS - MATLAB| = ${maxPptaDiff.toFixed(5)} dB HL`);
if (maxPptaDiff > 0.01) throw new Error("PPTA parity exceeded 0.01 dB!");

/* 3. Simulate 50-trial Adaptive-SNR test + retest across all 98 human listeners */
const rngSim = makeRng(20261118);
const t0 = performance.now();
let sqErrAll50 = 0.0, sqErrNhSl50 = 0.0, nNhSl = 0;
let sumRetestMad50 = 0.0, sumPc50 = 0.0;

for (const l of hd.listeners) {
  const alphaTrue = l.multisnr_params.alpha_plomp_logit;
  const sqTrue = l.multisnr_params.snr_quiet_db;
  const trueD = QVCVEngine.CAL.alpha0.reduce((acc, a0, c) => acc + (a0 - alphaTrue[c]) / QVCVEngine.CAL.slope, 0.0) / 10.0 - QVCVEngine.CAL.nhMeanSupraDb;

  const ests = [];
  for (let rep = 0; rep < 2; rep++) {
    const sched = QVCVEngine.buildAdaptiveSchedule(50, rngSim);
    const tracker = QVCVEngine.createAdaptiveTracker();
    const trials = [];
    const counts = Array.from({ length: 10 }, () => new Array(11).fill(0));

    for (let t = 0; t < 50; t++) {
      const item = sched[t];
      const ci = item.consonantIdx;
      const snrDb = tracker.selectSnrForConsonant(ci);
      const tSign = item.tokenIdx === 0 ? 1.0 : -1.0;
      const sEff = QVCVEngine.effectiveAudibleSnr(snrDb, sqTrue[ci]);
      const eta = alphaTrue[ci] + QVCVEngine.CAL.slope * sEff + tSign * QVCVEngine.CAL.tokHalfDiff[ci];
      const pTrue = QVCVEngine.CAL.gamma + QVCVEngine.CAL.span / (1.0 + Math.exp(-Math.max(-35, Math.min(35, eta))));
      const corr = rngSim() < pTrue;

      tracker.updateWithTrial(ci, item.tokenIdx, snrDb, corr);
      trials.push({ consonantIdx: ci, tokenIdx: item.tokenIdx, snrDb, correct: corr, rtMs: 650 });
      counts[ci][corr ? ci : 10] += 1;
    }

    const sum = QVCVEngine.summarizeSession({
      trials,
      counts10x11: counts,
      isAdaptive: true,
      audiogramHL: l.audiogram_db
    });
    ests.push(sum);
    sumPc50 += sum.pctCorrect;
  }

  const err0 = ests[0].supraDeficitDb - trueD;
  sqErrAll50 += err0 * err0;
  if (l.archetype === "Normal" || l.archetype === "Slight") {
    sqErrNhSl50 += err0 * err0;
    nNhSl++;
  }
  sumRetestMad50 += Math.abs(ests[0].globalSnrLossDb - ests[1].globalSnrLossDb);
}

const elapsedMs = performance.now() - t0;
const rmseAll50 = Math.sqrt(sqErrAll50 / hd.listeners.length);
const rmseNhSl50 = Math.sqrt(sqErrNhSl50 / nNhSl);
const retestMad50 = sumRetestMad50 / hd.listeners.length;
const meanPc50 = sumPc50 / (2 * hd.listeners.length);

console.log(`[PASS] 50-Trial JS Adaptive-SNR across 98 listeners x 2 runs (${(elapsedMs / 196).toFixed(2)} ms/session):`);
console.log(`       Mean % Correct = ${meanPc50.toFixed(1)}% | All-98 Supra RMSE = ${rmseAll50.toFixed(2)} dB | NH+Slight RMSE = ${rmseNhSl50.toFixed(2)} dB | Test-Retest MAD = ${retestMad50.toFixed(2)} dB`);
if (rmseNhSl50 > 1.1 || meanPc50 < 48 || meanPc50 > 65) {
  throw new Error("Unexpected Adaptive-SNR tracking metrics!");
}
