#!/usr/bin/env python3
"""
characterize_human_listeners.py

Calibrates the biophysical spectro-temporal + qCLS-coupled virtual listener
cohort across the 98 unaided (96 paired qCLS) and 67 aided human sessions in
``qtest_dataset``, validates out-of-sample human confusion prediction, and
provides a standalone CLI/library to characterize individual human listeners.

Outputs written to ``sim/human_data/`` and ``sim/human_calibrated_results/``:
  - human_data/biophysical_listeners_98.json
  - human_data/biophysical_listeners_98.csv   (drop-in for qvcv_tracking_sim.py)
  - human_data/human_characterization_98_unaided.csv
  - human_data/human_characterization_aided_sessions.csv
  - human_calibrated_results/biophysical_model_validation.json
  - human_calibrated_results/fig_biophysical_listener_model.png

Single-file clinical characterization usage:
  python3 sim/characterize_human_listeners.py --csv path/to/qVCV_NH001_L_20260511.csv \
      --audiogram 10,10,15,15,20,25,35,45,55,60
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qvcv_mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.optimize import minimize

from export_human_listeners import (
    loocv_hellinger_aroc_unbiased,
    predict_ppta_mnr,
)
from qvcv_biophysical_model import (
    AUD_FREQS_HZ,
    DEFAULT_QCLS_MAT,
    DEFAULT_STIMULI_DIR,
    REF_SNR_ARR,
    BiophysicalHyperparams,
    BiophysicalVirtualListener,
    QCLSCochlearProfile,
    SpectroTemporalVCVBank,
    compute_peripheral_template_features,
    extract_vcv_acoustic_bank,
    load_qcls_cochlear_profiles,
    profile_from_audiogram_only,
)
from qvcv_simulator import (
    DISPLAY_CONSONANTS,
    N_RESPONSES,
    N_STIMULI,
    STIMULI_CONSONANTS,
    load_ground_truth,
    normalize_rows,
    save_cohort_csv,
)
from qvcv_tracking_sim import write_dict_rows_csv
from qvcv_trackers import estimate_eb_concentration, fit_feature_choice_model

LF_CONS_IDX = [0, 1, 2, 4, 8]   # B, D, G, N, V (low/mid-frequency voice/murmur/locus)
HF_CONS_IDX = [3, 5, 6, 7, 9]   # K, S, SH, T, Z (high-frequency burst/sibilant frication)


def _compute_cohort_probs_batch(
    nd_n_10_2_10: np.ndarray,
    ac_n_10_2: np.ndarray,
    eff_n_10: np.ndarray,
    self_offset_10: np.ndarray,
    dir_bias_10_10: np.ndarray,
    unk_crit_n: np.ndarray,
    lapse_rate: float,
) -> np.ndarray:
    """Vectorized (N, 10_stim, 11_resp) confusion probability matrix averaged over both tokens."""
    n_subj = nd_n_10_2_10.shape[0]
    logits = np.zeros((n_subj, N_STIMULI, 2, N_RESPONSES), dtype=float)
    logits[:, :, :, :N_STIMULI] = eff_n_10[:, :, None, None] * nd_n_10_2_10

    eye_10_11 = np.eye(N_STIMULI, N_RESPONSES, dtype=float)[None, :, None, :]
    diag_boost = self_offset_10[None, :, None] + 0.14 * eff_n_10[:, :, None] * ac_n_10_2  # (N, 10, 2)
    logits += diag_boost[:, :, :, None] * eye_10_11

    uncert_gate = 1.0 / (1.0 + np.exp((ac_n_10_2 - 22.0) / 7.5))  # (N, 10, 2)
    logits[:, :, :, :N_STIMULI] += (
        (0.35 + 0.65 * uncert_gate[:, :, :, None]) * dir_bias_10_10[None, :, None, :]
    )

    low_evid = 1.0 / (1.0 + np.exp((ac_n_10_2 - 8.0) / 4.0))
    logits[:, :, :, N_STIMULI] = unk_crit_n[:, None, None] + 2.2 * low_evid

    logits -= np.max(logits, axis=3, keepdims=True)
    probs = np.exp(logits)
    probs /= np.sum(probs, axis=3, keepdims=True)
    if lapse_rate > 0.0:
        probs = (1.0 - lapse_rate) * probs + (lapse_rate / float(N_RESPONSES))
    return 0.5 * (probs[:, :, 0, :] + probs[:, :, 1, :])


def fit_single_listener_biophysical(
    counts_10x11: np.ndarray,
    nd_1_10_2_10: np.ndarray,
    ac_1_10_2: np.ndarray,
    self_offset_10: np.ndarray,
    dir_bias_10_10: np.ndarray,
    lapse_rate: float = 0.015,
    fit_per_consonant_nuisance: bool = False,
    practice_score_pct: Optional[float] = None,
    nd_prac_1_10_2_10: Optional[np.ndarray] = None,
    ac_prac_1_10_2: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    """
    Fit a listener's neural coding efficiency vector ``eff_10`` (>0; 1.0 = NH norm)
    and ``unk_criterion`` via regularized MAP against observed confusion counts.

    Low-dimensional parameterization (default 3 params: log_eff_lf, log_eff_hf, unk_crit)
    prevents overfitting on single 100-trial sessions; when ``fit_per_consonant_nuisance``
    is True (e.g. 200-trial pooled calibration), adds regularized per-consonant offsets.
    """
    counts = np.asarray(counts_10x11, dtype=float)

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
        g_lf, g_hf, u_c = float(theta[0]), float(theta[1]), float(theta[2])
        log_eff = np.zeros(N_STIMULI, dtype=float)
        log_eff[LF_CONS_IDX] = g_lf
        log_eff[HF_CONS_IDX] = g_hf
        if fit_per_consonant_nuisance and len(theta) == 3 + N_STIMULI:
            log_eff = log_eff + theta[3:]
        return np.exp(np.clip(log_eff, -2.2, 1.4)), float(np.clip(u_c, -6.5, 2.5))

    def neg_log_post(theta: np.ndarray) -> float:
        eff_10, u_c = unpack(theta)
        p_mat = _compute_cohort_probs_batch(
            nd_1_10_2_10, ac_1_10_2, eff_10[None, :], self_offset_10, dir_bias_10_10, np.array([u_c]), lapse_rate
        )[0]
        nll = -float(np.sum(counts * np.log(np.maximum(p_mat, 1e-12))))
        # Weak Gaussian prior centered at NH norm (log_eff=0, unk_crit=-2.5):
        reg = (
            0.5 * ((theta[0] / 0.75) ** 2 + (theta[1] / 0.75) ** 2)
            + 0.5 * (((theta[0] - theta[1]) / 0.55) ** 2)
            + 0.5 * (((theta[2] - (-2.5)) / 1.8) ** 2)
        )
        if fit_per_consonant_nuisance and len(theta) == 3 + N_STIMULI:
            reg += 0.5 * float(np.sum((theta[3:] / 0.32) ** 2))

        if practice_score_pct is not None and nd_prac_1_10_2_10 is not None and ac_prac_1_10_2 is not None:
            p_prac = _compute_cohort_probs_batch(
                nd_prac_1_10_2_10, ac_prac_1_10_2, eff_10[None, :], self_offset_10, dir_bias_10_10, np.array([u_c]), 0.023
            )[0]
            pc_pred = float(np.mean(np.diag(p_prac[:, :N_STIMULI])))
            k_pr = 10.0 * float(practice_score_pct) / 100.0
            nll -= float(
                k_pr * np.log(max(pc_pred, 1e-9)) + (10.0 - k_pr) * np.log(max(1.0 - pc_pred, 1e-9))
            )
        return nll + reg

    x0 = np.zeros(3 + (N_STIMULI if fit_per_consonant_nuisance else 0), dtype=float)
    x0[2] = -2.5
    res = minimize(neg_log_post, x0, method="BFGS", options={"maxiter": 60, "gtol": 1e-4})
    eff_hat, unk_hat = unpack(res.x)
    p_hat = _compute_cohort_probs_batch(
        nd_1_10_2_10, ac_1_10_2, eff_hat[None, :], self_offset_10, dir_bias_10_10, np.array([unk_hat]), lapse_rate
    )[0]
    return eff_hat, unk_hat, p_hat


def calibrate_cohort_hyperparams(
    counts_n_10_11: np.ndarray,
    is_nh_mask: np.ndarray,
    nd_n_10_2_10: np.ndarray,
    ac_n_10_2: np.ndarray,
    hp: BiophysicalHyperparams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alternating Empirical-Bayes MAP calibration of:
      - ``self_offset_10`` (10,): NH baseline target clarity at eff = 1.0
      - ``dir_bias_10_10`` (10, 10): zero-diagonal directional cue-loss & coronal choice bias
    """
    n_subj = counts_n_10_11.shape[0]
    off_mask = ~np.eye(N_STIMULI, dtype=bool)  # 90 off-diagonal entries

    # Initialize:
    self_off = np.full(N_STIMULI, 0.8, dtype=float)
    dir_bias = np.zeros((N_STIMULI, N_STIMULI), dtype=float)
    eff_mat = np.ones((n_subj, N_STIMULI), dtype=float)
    unk_vec = np.full(n_subj, -2.5, dtype=float)

    for outer in range(3):
        # Step 1: Update global self_off (10) + dir_bias off-diagonal (90) = 100 params via L-BFGS-B:
        # Anchor NH mean log_eff near 0 by keeping NH eff fixed at 1.0 on outer==0:
        eff_use = np.where(is_nh_mask[:, None], 1.0, eff_mat) if outer == 0 else eff_mat

        def global_obj(vec: np.ndarray) -> tuple[float, np.ndarray]:
            so = vec[:N_STIMULI]
            db = np.zeros((N_STIMULI, N_STIMULI), dtype=float)
            db[off_mask] = vec[N_STIMULI:]

            # Forward & exact analytical gradient (ignoring tiny lapse_rate in grad via chain rule):
            n_s = nd_n_10_2_10.shape[0]
            logits = np.zeros((n_s, N_STIMULI, 2, N_RESPONSES), dtype=float)
            logits[:, :, :, :N_STIMULI] = eff_use[:, :, None, None] * nd_n_10_2_10
            eye_10_11 = np.eye(N_STIMULI, N_RESPONSES, dtype=float)[None, :, None, :]
            diag_boost = so[None, :, None] + 0.14 * eff_use[:, :, None] * ac_n_10_2
            logits += diag_boost[:, :, :, None] * eye_10_11
            ug = 0.35 + 0.65 / (1.0 + np.exp((ac_n_10_2 - 22.0) / 7.5))  # (N, 10, 2)
            logits[:, :, :, :N_STIMULI] += ug[:, :, :, None] * db[None, :, None, :]
            low_ev = 1.0 / (1.0 + np.exp((ac_n_10_2 - 8.0) / 4.0))
            logits[:, :, :, N_STIMULI] = unk_vec[:, None, None] + 2.2 * low_ev

            logits -= np.max(logits, axis=3, keepdims=True)
            q_tok = np.exp(logits)
            q_tok /= np.sum(q_tok, axis=3, keepdims=True)  # (N, 10, 2, 11)
            lam = hp.lapse_rate
            p_tok = (1.0 - lam) * q_tok + (lam / float(N_RESPONSES))
            p_avg = 0.5 * (p_tok[:, :, 0, :] + p_tok[:, :, 1, :])  # (N, 10, 11)

            nll = -float(np.sum(counts_n_10_11 * np.log(np.maximum(p_avg, 1e-12))))
            # Regularize dir_bias toward 0 (prior SD = 0.85) and self_off weakly:
            reg = 0.5 * float(np.sum((db[off_mask] / 0.85) ** 2)) + 0.5 * float(np.sum(((so - 0.8) / 3.0) ** 2))

            # Exact gradient wrt logits[n, c, m, r]:
            w_ncr = counts_n_10_11 / np.maximum(p_avg, 1e-12)  # (N, 10, 11)
            alpha_ncmr = 0.5 * (1.0 - lam) * w_ncr[:, :, None, :] * q_tok  # (N, 10, 2, 11)
            sum_alpha = np.sum(alpha_ncmr, axis=3, keepdims=True)
            dL_dlogit = sum_alpha * q_tok - alpha_ncmr  # (N, 10, 2, 11)

            grad_so = np.sum(dL_dlogit * eye_10_11, axis=(0, 2, 3)) + (so - 0.8) / (3.0 ** 2)
            grad_db_full = np.sum(dL_dlogit[:, :, :, :N_STIMULI] * ug[:, :, :, None], axis=(0, 2))
            grad_db = grad_db_full[off_mask] + db[off_mask] / (0.85 ** 2)
            return nll + reg, np.concatenate([grad_so, grad_db])

        v0 = np.concatenate([self_off, dir_bias[off_mask]])
        res_g = minimize(global_obj, v0, jac=True, method="L-BFGS-B", options={"maxiter": 80})
        self_off = res_g.x[:N_STIMULI].copy()
        dir_bias = np.zeros((N_STIMULI, N_STIMULI), dtype=float)
        dir_bias[off_mask] = res_g.x[N_STIMULI:]

        # Step 2: Fit each listener's individual neural coding efficiency & '???' criterion:
        for i in range(n_subj):
            eff_i, unk_i, _ = fit_single_listener_biophysical(
                counts_n_10_11[i],
                nd_n_10_2_10[i : i + 1],
                ac_n_10_2[i : i + 1],
                self_off,
                dir_bias,
                lapse_rate=hp.lapse_rate,
                fit_per_consonant_nuisance=True,
            )
            eff_mat[i] = eff_i
            unk_vec[i] = unk_i

        # Re-normalize NH median efficiency to 1.0 so self_off carries the exact NH norm:
        nh_med = np.median(eff_mat[is_nh_mask], axis=0)
        eff_mat = eff_mat / np.maximum(nh_med[None, :], 0.25)

    return self_off, dir_bias


def compute_snr_deficit_decomposition_db(
    listener: BiophysicalVirtualListener,
    nh_ref_listener: BiophysicalVirtualListener,
) -> dict[str, float]:
    """
    Exact biophysical 3-way SNR-equivalence decomposition (in dB SNR relative to NH norm):
      Total_Behavioral_Deficit_dB =
          (1) Threshold_Audibility_Loss_dB       (elevated audiogram with normal narrow ERB filters)
        + (2) OHC_Filter_Broadening_USOM_dB      (extra SNR penalty from qCLS OHC filter widening & vowel USOM)
        + (3) Neural_Suprathreshold_Deficit_dB   (residual retrocochlear/central coding deficit)
    Evaluated by finding the uniform SNR offset ΔSNR (relative to Ref_SNR) at which each stage
    reaches the NH 75%-correct speech-in-noise benchmark!
    """
    def eval_pc(lstn: BiophysicalVirtualListener, ds: float) -> float:
        m = lstn.matrix_at_snrs(REF_SNR_ARR + ds)
        return float(np.mean(np.diag(m[:, :N_STIMULI])))

    def find_srt75_offset(lstn: BiophysicalVirtualListener, target_pc: float = 0.75) -> float:
        lo, hi = -16.0, 32.0
        if eval_pc(lstn, hi) <= target_pc:
            return hi
        if eval_pc(lstn, lo) >= target_pc:
            return lo
        for _ in range(11):  # 48 / 2^11 = 0.023 dB precision in 11 steps!
            mid = 0.5 * (lo + hi)
            if eval_pc(lstn, mid) < target_pc:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    # Stage 0: Ideal NH listener (cached):
    if not hasattr(nh_ref_listener, "_cached_srt75"):
        setattr(nh_ref_listener, "_cached_srt75", find_srt75_offset(nh_ref_listener))
    srt_nh = float(getattr(nh_ref_listener, "_cached_srt75"))

    # Stage 1: Listener's pure-tone audiogram ONLY (zero OHC filter broadening, NH neural eff = 1.0):
    narrow_prof = QCLSCochlearProfile(
        listener_id=listener.listener_id,
        has_measured_qcls=listener.cochlear_profile.has_measured_qcls,
        audiogram_hl_db=listener.cochlear_profile.audiogram_hl_db,
        hl_ohc_db=np.zeros(10, dtype=float),
        hl_ihc_db=np.maximum(listener.cochlear_profile.audiogram_hl_db, 0.0),
        cls_speech_slope=np.ones(10, dtype=float),
    )
    lstn_aud_only = BiophysicalVirtualListener(
        listener_id=listener.listener_id,
        archetype=listener.archetype,
        description="aud_only",
        cochlear_profile=narrow_prof,
        bank=listener.bank,
        hyperparams=listener.hp,
        neural_efficiency_per_cons=np.ones(N_STIMULI, dtype=float),
        self_clarity_offset=nh_ref_listener.self_clarity_offset,
        unk_criterion=nh_ref_listener.unk_criterion,
        ha_insertion_gain_db=listener.ha_insertion_gain_db,
        ha_wdrc_smear=listener.ha_wdrc_smear,
    )
    srt_aud = find_srt75_offset(lstn_aud_only)

    # Stage 2: Listener's pure-tone audiogram + qCLS OHC filter broadening/USOM (NH neural eff = 1.0):
    lstn_periph = BiophysicalVirtualListener(
        listener_id=listener.listener_id,
        archetype=listener.archetype,
        description="periph_full",
        cochlear_profile=listener.cochlear_profile,
        bank=listener.bank,
        hyperparams=listener.hp,
        neural_efficiency_per_cons=np.ones(N_STIMULI, dtype=float),
        self_clarity_offset=nh_ref_listener.self_clarity_offset,
        unk_criterion=nh_ref_listener.unk_criterion,
        ha_insertion_gain_db=listener.ha_insertion_gain_db,
        ha_wdrc_smear=listener.ha_wdrc_smear,
    )
    srt_periph = find_srt75_offset(lstn_periph)

    # Stage 3: Full listener (including neural/suprathreshold coding efficiency):
    srt_full = find_srt75_offset(listener)

    aud_loss = srt_aud - srt_nh
    ohc_broad = srt_periph - srt_aud
    neural_supra = srt_full - srt_periph
    tot_def = srt_full - srt_nh
    return {
        "total_behavioral_deficit_db": round(tot_def, 3),
        "threshold_audibility_loss_db": round(aud_loss, 3),
        "ohc_broadening_usom_loss_db": round(ohc_broad, 3),
        "neural_suprathreshold_deficit_db": round(neural_supra, 3),
        "neural_eff_lf_ratio": round(float(np.mean(listener.neural_efficiency_per_cons[LF_CONS_IDX])), 3),
        "neural_eff_hf_ratio": round(float(np.mean(listener.neural_efficiency_per_cons[HF_CONS_IDX])), 3),
    }


def plot_biophysical_validation_figure(
    human_off_90: np.ndarray,
    biophys_off_90: np.ndarray,
    sym_off_90: np.ndarray,
    pair_labels_90: list[str],
    unaided_rows: list[dict[str, Any]],
    cv_summary: dict[str, float],
    nh_ref_listener: BiophysicalVirtualListener,
    mod_example_listener: BiophysicalVirtualListener,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.4, 11.0), dpi=160)
    fig.suptitle(
        "Biophysical Spectro-Temporal + qCLS Coupled Listener Model: Empirical Human Validation (n = 98)\n"
        "(Waveform 3-Epoch ERB Templates + Paired 10-Freq qCLS OHC Filter Broadening + Asymmetric Luce Choice)",
        fontsize=12.4,
        fontweight="bold",
        y=0.985,
    )

    # Panel A: 90-Cell Directional Off-Diagonal Confusion Scatter (Human vs. Biophysical vs. Old Symmetric)
    ax = axes[0, 0]
    r_bio = float(np.corrcoef(human_off_90, biophys_off_90)[0, 1])
    r_sym = float(np.corrcoef(human_off_90, sym_off_90)[0, 1])
    ax.scatter(human_off_90, sym_off_90, s=26, alpha=0.45, color="#7f7f7f", label=f"Previous Symmetric Feature Model (r = {r_sym:.3f})")
    ax.scatter(human_off_90, biophys_off_90, s=34, alpha=0.85, color="#d62728", label=f"Biophysical Asymmetric Model (r = {r_bio:.3f})")
    lim = max(float(np.max(human_off_90)), float(np.max(biophys_off_90))) * 1.08
    ax.plot([0, lim], [0, lim], "k--", lw=1.2)
    top_idx = np.argsort(-human_off_90)[:6]
    for k in top_idx:
        ax.annotate(pair_labels_90[k], (human_off_90[k], biophys_off_90[k]), xytext=(4, 3), textcoords="offset points", fontsize=8.0, fontweight="bold")
    ax.set_xlabel("Empirical Human Cohort Error Count (per cell, 19,600 trials)")
    ax.set_ylabel("Simulated Cohort Error Count")
    ax.set_title("A. Directional Consonant Confusions (All 90 Off-Diagonal Pairs)", fontweight="bold", fontsize=10.6)
    ax.legend(loc="upper left", fontsize=8.2)
    ax.grid(alpha=0.25)

    # Panel B: 3-Component Biophysical Deficit Decomposition by Clinical Group
    ax = axes[0, 1]
    grps = ["Normal", "Slight", "Mild", "Moderate"]
    aud_means, ohc_means, neu_means = [], [], []
    for g in grps:
        sub = [r for r in unaided_rows if r["group"] == g]
        aud_means.append(float(np.mean([max(r["threshold_audibility_loss_db"], 0.0) for r in sub])))
        ohc_means.append(float(np.mean([max(r["ohc_broadening_usom_loss_db"], 0.0) for r in sub])))
        neu_means.append(float(np.mean([r["neural_suprathreshold_deficit_db"] for r in sub])))
    xg = np.arange(len(grps))
    bw = 0.52
    b1 = np.array(aud_means)
    b2 = np.array(ohc_means)
    b3 = np.array(neu_means)
    ax.bar(xg, b1, bw, color="#1f77b4", label="1. Pure-Tone Threshold Audibility Loss")
    ax.bar(xg, b2, bw, bottom=b1, color="#ff7f0e", label="2. qCLS OHC Filter Broadening & Vowel USOM")
    ax.bar(xg, np.maximum(b3, 0.0), bw, bottom=b1 + b2, color="#d62728", label="3. Neural / Suprathreshold Coding Deficit")
    ax.set_xticks(xg)
    ax.set_xticklabels([f"{g}\n(n={sum(1 for r in unaided_rows if r['group']==g)})" for g in grps])
    ax.set_ylabel("Mean Equivalent SNR Deficit vs. NH Norm (dB) ↑")
    ax.set_title("B. Biophysical 3-Stage Deficit Decomposition Across 98 Listeners", fontweight="bold", fontsize=10.6)
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(axis="y", alpha=0.25)

    # Panel C: Strictly Out-of-Sample Session-1 -> Session-2 Human Characterization (Strictly Proper Scoring Rules)
    ax = axes[1, 0]
    methods = ["Raw Session-1\nCounts (app.js)", "Symmetric\nFeature-EB", "Biophysical + qCLS\nAsymmetric-EB"]
    briers = [cv_summary["s1_to_s2_brier_raw"], cv_summary["s1_to_s2_brier_sym_eb"], cv_summary["s1_to_s2_brier_biophys_eb"]]
    nlls = [cv_summary["s1_to_s2_nll_raw"], cv_summary["s1_to_s2_nll_sym_eb"], cv_summary["s1_to_s2_nll_biophys_eb"]]
    xm = np.arange(len(methods))
    bars = ax.bar(xm - 0.17, briers, 0.32, color=["#7f7f7f", "#1f77b4", "#2ca02c"], label="Held-Out S2 Brier Loss (×100) ↓")
    ax2 = ax.twinx()
    bars2 = ax2.bar(xm + 0.17, nlls, 0.32, color=["#b5b5b5", "#6baed6", "#74c476"], hatch="//", label="Held-Out S2 Cross-Entropy (NLL) ↓")
    ax.set_xticks(xm)
    ax.set_xticklabels(methods, fontsize=8.8)
    ax.set_ylabel("Held-Out Human Session-2 Brier Score (×100) ↓")
    ax2.set_ylabel("Held-Out Human Session-2 Cross-Entropy (NLL/session) ↓")
    ax.set_ylim(min(briers) * 0.80, max(briers) * 1.12)
    ax2.set_ylim(min(nlls) * 0.85, max(nlls) * 1.10)
    ax.set_title("C. Out-of-Sample Human Characterization (Fit S1 → Predict S2, LOLO)", fontweight="bold", fontsize=10.5)
    ax.grid(axis="y", alpha=0.25)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{b.get_height():.2f}", ha="center", fontsize=8.2, fontweight="bold")
    for b in bars2:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.7, f"{b.get_height():.1f}", ha="center", fontsize=8.2, fontweight="bold")

    # Panel D: Multi-SNR Directional Confusion Emergence on /S/ -> /T/ and /D/ -> /G/
    ax = axes[1, 1]
    snr_sweep = np.linspace(-15.0, 20.0, 36)
    idx_S, idx_T = STIMULI_CONSONANTS.index("S"), DISPLAY_CONSONANTS.index("T")
    idx_D, idx_G = STIMULI_CONSONANTS.index("D"), DISPLAY_CONSONANTS.index("G")

    s_self_nh = [nh_ref_listener.response_probs(idx_S, s)[idx_S] * 100.0 for s in snr_sweep]
    s_self_mod = [mod_example_listener.response_probs(idx_S, s)[idx_S] * 100.0 for s in snr_sweep]
    s_to_t_mod = [mod_example_listener.response_probs(idx_S, s)[idx_T] * 100.0 for s in snr_sweep]
    t_to_s_mod = [mod_example_listener.response_probs(STIMULI_CONSONANTS.index("T"), s)[DISPLAY_CONSONANTS.index("S")] * 100.0 for s in snr_sweep]
    d_to_g_mod = [mod_example_listener.response_probs(idx_D, s)[idx_G] * 100.0 for s in snr_sweep]

    ax.plot(snr_sweep, s_self_nh, "-", color="#2ca02c", lw=2.1, label="NH Norm: P(S | S)")
    ax.plot(snr_sweep, s_self_mod, "-", color="#1f77b4", lw=2.1, label=f"{mod_example_listener.listener_id} (Mod HL): P(S | S)")
    ax.plot(snr_sweep, s_to_t_mod, "--", color="#d62728", lw=2.2, label=f"{mod_example_listener.listener_id}: Asymmetric P(T | S) ↑")
    ax.plot(snr_sweep, t_to_s_mod, ":", color="#d62728", lw=1.8, label=f"{mod_example_listener.listener_id}: Reverse P(S | T) ≈ 0")
    ax.plot(snr_sweep, d_to_g_mod, "-.", color="#9467bd", lw=2.0, label=f"{mod_example_listener.listener_id}: Asymmetric P(G | D) ↑")
    ax.set_xlabel("Presentation SNR (dB)")
    ax.set_ylabel("Response Probability (%)")
    ax.set_title("D. Continuous Multi-SNR Psychometric & Asymmetric Error Curves", fontweight="bold", fontsize=10.6)
    ax.legend(loc="center right", fontsize=7.8)
    ax.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.945])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def characterize_single_csv(
    csv_path: Path,
    audiogram_hl: Optional[list[float]],
    calib_json_path: Path,
) -> None:
    """CLI helper to characterize any standalone human qVCV_*.csv file."""
    calib = json.loads(calib_json_path.read_text())
    hp_d = calib["biophysical_hyperparams"]
    hp = BiophysicalHyperparams(
        gamma_ohc=float(hp_d["gamma_ohc"]),
        usom_vowel_att_db=float(hp_d["usom_vowel_att_db"]),
        thr_offset_db=float(hp_d["thr_offset_db"]),
        spec_scale=float(hp_d["spec_scale"]),
        voice_scale=float(hp_d["voice_scale"]),
        cont_scale=float(hp_d["cont_scale"]),
        lapse_rate=float(hp_d["lapse_rate"]),
        directional_choice_bias=np.array(hp_d["directional_choice_bias"], dtype=float),
    )
    self_off = np.array(calib["self_clarity_offset_nh"], dtype=float)
    bank = extract_vcv_acoustic_bank(DEFAULT_STIMULI_DIR)

    counts = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
    with csv_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            ph = (row.get("Phase") or "").strip().lower()
            if ph and "practice" in ph:
                continue
            tc = (row.get("Target Consonant") or row.get("Target") or row.get("Consonant") or "").strip().upper()
            ur = (row.get("User Response") or row.get("Response") or "").strip().upper()
            if ur in ("OTHER", "UNK", "???"):
                ur = "???"
            if tc in STIMULI_CONSONANTS and ur in DISPLAY_CONSONANTS:
                counts[STIMULI_CONSONANTS.index(tc), DISPLAY_CONSONANTS.index(ur)] += 1.0
    if counts.sum() == 0:
        gt = load_ground_truth(csv_path)
        counts = gt.get_model(gt.keys()[0]).gt_matrix * 10.0

    aud = np.zeros(10, dtype=float) if audiogram_hl is None else np.asarray(audiogram_hl, dtype=float)
    prof = profile_from_audiogram_only(csv_path.stem, aud)
    nd, ac, _ = compute_peripheral_template_features(
        bank, REF_SNR_ARR, prof.audiogram_hl_db[None, :], prof.hl_ohc_db[None, :], hp
    )
    eff_hat, unk_hat, mu_bio = fit_single_listener_biophysical(
        counts, nd, ac, self_off, hp.directional_choice_bias, lapse_rate=hp.lapse_rate
    )
    kappa = estimate_eb_concentration(counts, mu_bio)
    smoothed_ccm = normalize_rows(counts + kappa * mu_bio)

    nh_prof = profile_from_audiogram_only("NH_REF", np.zeros(10, dtype=float))
    nh_ref = BiophysicalVirtualListener("NH_REF", "Normal", "NH Norm", nh_prof, bank, hp, np.ones(10), self_off, -2.5)
    lstn = BiophysicalVirtualListener(csv_path.stem, "Clinical", csv_path.name, prof, bank, hp, eff_hat, self_off, unk_hat)
    decomp = compute_snr_deficit_decomposition_db(lstn, nh_ref)

    report = {
        "file": str(csv_path),
        "total_trials": int(round(counts.sum())),
        "percent_correct": round(float(100.0 * np.trace(counts[:, :10]) / max(counts.sum(), 1.0)), 1),
        "audiogram_provided": audiogram_hl is not None,
        "biophysical_decomposition_db_snr": decomp,
        "unk_caution_criterion": round(unk_hat, 3),
        "eb_prior_concentration_kappa": round(kappa, 2),
        "smoothed_diagonal_pct": {
            STIMULI_CONSONANTS[c]: round(float(smoothed_ccm[c, c] * 100.0), 1) for c in range(N_STIMULI)
        },
    }
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate biophysical qVCV listener model & characterize human listeners.")
    parser.add_argument("--csv", type=str, default=None, help="Optional single qVCV_*.csv file to characterize")
    parser.add_argument("--audiogram", type=str, default=None, help="Optional 10 comma-separated dB HL thresholds (250..8000 Hz)")
    args = parser.parse_args()

    sim_dir = Path(__file__).resolve().parent
    human_dir = sim_dir / "human_data"
    res_dir = sim_dir / "human_calibrated_results"

    if args.csv is not None:
        aud_list = [float(x) for x in args.audiogram.split(",")] if args.audiogram else None
        characterize_single_csv(Path(args.csv), aud_list, human_dir / "biophysical_listeners_98.json")
        return

    t0 = time.time()
    print("[1/5] Extracting spectro-temporal consonant cue templates from qVCV .wav waveforms...", flush=True)
    bank = extract_vcv_acoustic_bank(DEFAULT_STIMULI_DIR)

    base_data = json.loads((human_dir / "human_listeners_98.json").read_text())
    mnr_data = json.loads((human_dir / "mnr_vcv_pta_model.json").read_text())
    B_mnr = np.array(mnr_data["B"], dtype=float)
    C_mnr = np.array(mnr_data["C"], dtype=float)
    ca_mnr = np.array(mnr_data["ca"], dtype=float)
    pa_mnr = np.array(mnr_data["pa"], dtype=float)

    listeners_meta = base_data["listeners"]
    n_subj = len(listeners_meta)
    is_nh = np.array([bool(r["is_nh"]) for r in listeners_meta], dtype=bool)
    groups = np.array([r["archetype"] for r in listeners_meta])

    print("[2/5] Loading paired 10-frequency qCLS equal-loudness contours (OHC vs. IHC loss)...", flush=True)
    qcls_map = load_qcls_cochlear_profiles(listeners_meta, DEFAULT_QCLS_MAT)
    n_qcls = sum(1 for p in qcls_map.values() if p.has_measured_qcls)

    aud_n10 = np.array([qcls_map[r["listener_id"]].audiogram_hl_db for r in listeners_meta], dtype=float)
    ohc_n10 = np.array([qcls_map[r["listener_id"]].hl_ohc_db for r in listeners_meta], dtype=float)

    s1_counts = np.array([r["empirical_s1_counts"] for r in listeners_meta], dtype=float)
    s2_counts = np.array([r["empirical_s2_counts"] for r in listeners_meta], dtype=float)
    pooled_counts = s1_counts + s2_counts  # 200 trials per listener

    hp = BiophysicalHyperparams()
    nd_ref, ac_ref, _ = compute_peripheral_template_features(bank, REF_SNR_ARR, aud_n10, ohc_n10, hp)
    nd_prac, ac_prac, _ = compute_peripheral_template_features(
        bank, np.full(N_STIMULI, 60.0), aud_n10, ohc_n10, hp
    )

    print("[3/5] Fitting biophysical spectro-temporal + asymmetric choice model to 98 human listeners...", flush=True)
    self_off, dir_bias = calibrate_cohort_hyperparams(pooled_counts, is_nh, nd_ref, ac_ref, hp)
    hp.directional_choice_bias = dir_bias

    nh_prof = profile_from_audiogram_only("NH_REF", np.zeros(10, dtype=float))
    nh_ref_lstn = BiophysicalVirtualListener(
        "NH_REF", "Normal", "Ideal NH Norm", nh_prof, bank, hp, np.ones(N_STIMULI), self_off, -2.5
    )

    bio_listeners: list[BiophysicalVirtualListener] = []
    fixed_vls = []
    unaided_rows: list[dict[str, Any]] = []
    bio_json_records: list[dict[str, Any]] = []

    for i, r in enumerate(listeners_meta):
        sid = r["listener_id"]
        prof = qcls_map[sid]
        eff_i, unk_i, _ = fit_single_listener_biophysical(
            pooled_counts[i],
            nd_ref[i : i + 1],
            ac_ref[i : i + 1],
            self_off,
            dir_bias,
            lapse_rate=hp.lapse_rate,
            fit_per_consonant_nuisance=True,
            practice_score_pct=float(r["practice_pc"]),
            nd_prac_1_10_2_10=nd_prac[i : i + 1],
            ac_prac_1_10_2=ac_prac[i : i + 1],
        )
        bl = BiophysicalVirtualListener(
            listener_id=sid,
            archetype=r["archetype"],
            description=r["description"],
            cochlear_profile=prof,
            bank=bank,
            hyperparams=hp,
            neural_efficiency_per_cons=eff_i,
            self_clarity_offset=self_off,
            unk_criterion=unk_i,
        )
        bio_listeners.append(bl)
        fvl = bl.to_fixed_snr_listener()
        fixed_vls.append(fvl)

        decomp = compute_snr_deficit_decomposition_db(bl, nh_ref_lstn)
        unaided_rows.append({
            "listener_id": sid,
            "group": r["archetype"],
            "has_paired_qcls": prof.has_measured_qcls,
            "pta_124_db_hl": round(float(r["pta_124_db"]), 2),
            "ppta_db_hl": round(float(r["human_pooled_ppta_db"]), 2),
            "legacy_rhl_db": round(float(r["rhl_db"]), 2),
            "qcls_hf_ohc_loss_db": round(float(np.mean(prof.hl_ohc_db[5:])), 2),
            "qcls_hf_ihc_loss_db": round(float(np.mean(prof.hl_ihc_db[5:])), 2),
            "mean_hf_erb_broadening_x": round(float(np.mean(prof.erb_broadening_factor(hp.gamma_ohc)[5:])), 2),
            **decomp,
            "unk_caution_criterion": round(unk_i, 3),
        })

        bio_json_records.append({
            "listener_id": sid,
            "archetype": r["archetype"],
            "is_nh": bool(r["is_nh"]),
            "has_measured_qcls": prof.has_measured_qcls,
            "audiogram_hl_db": np.round(prof.audiogram_hl_db, 2).tolist(),
            "hl_ohc_db": np.round(prof.hl_ohc_db, 2).tolist(),
            "hl_ihc_db": np.round(prof.hl_ihc_db, 2).tolist(),
            "cls_speech_slope": np.round(prof.cls_speech_slope, 3).tolist(),
            "neural_efficiency_per_cons": np.round(eff_i, 4).tolist(),
            "unk_criterion": round(unk_i, 4),
            "deficit_decomposition_db_snr": decomp,
            "model_ref_matrix": np.round(fvl.gt_matrix, 5).tolist(),
        })

    print("[4/5] Running strict Leave-One-Listener-Out S1->S2 cross-validation & aided session characterization...", flush=True)
    # Strictly out-of-sample LOLO directional prior: fit global self_off & dir_bias on 5 folds (zero listener leakage!)
    rng_cv = np.random.default_rng(20261115)
    perm = rng_cv.permutation(n_subj)
    folds = np.array_split(perm, 5)
    brier_raw_l, brier_sym_l, brier_bio_l = [], [], []
    retest_tvd_raw_l, retest_tvd_sym_l, retest_tvd_bio_l = [], [], []
    nll_raw_l, nll_sym_l, nll_bio_l = [], [], []
    s1_neural_def, s2_neural_def = [], []

    for f_idx in range(5):
        val_idx = folds[f_idx]
        tr_mask = np.ones(n_subj, dtype=bool)
        tr_mask[val_idx] = False
        so_cv, db_cv = calibrate_cohort_hyperparams(
            pooled_counts[tr_mask], is_nh[tr_mask], nd_ref[tr_mask], ac_ref[tr_mask], hp
        )
        hp_cv = BiophysicalHyperparams(directional_choice_bias=db_cv)
        nh_cv = BiophysicalVirtualListener("NH_CV", "Normal", "NH CV", nh_prof, bank, hp_cv, np.ones(10), so_cv, -2.5)

        for i in val_idx:
            c1, c2 = s1_counts[i], s2_counts[i]
            p2_emp = normalize_rows(c2)

            # 1) Raw Session-1 proportions (with tiny 0.03 Laplace floor only for finite NLL):
            p1_raw = normalize_rows(c1)
            p1_raw_nll = normalize_rows(c1 + 0.03)

            # 2) Symmetric Phonetic-Feature EB (fit independently on S1 and S2):
            _, mu_sym1 = fit_feature_choice_model(c1)
            p1_sym = normalize_rows(c1 + 1.8 * mu_sym1)
            _, mu_sym2 = fit_feature_choice_model(c2)
            p2_sym = normalize_rows(c2 + 1.8 * mu_sym2)

            # 3) Biophysical + qCLS Asymmetric EB (3 listener params fit on S1 only, using held-out fold prior!):
            eff_1, unk_1, mu_bio1 = fit_single_listener_biophysical(
                c1, nd_ref[i : i + 1], ac_ref[i : i + 1], so_cv, db_cv, lapse_rate=hp.lapse_rate, fit_per_consonant_nuisance=False
            )
            p1_bio = normalize_rows(c1 + 2.2 * mu_bio1)

            # Fit S2 independently to evaluate test-retest repeatability of both deficit and confusion matrices:
            eff_2, unk_2, mu_bio2 = fit_single_listener_biophysical(
                c2, nd_ref[i : i + 1], ac_ref[i : i + 1], so_cv, db_cv, lapse_rate=hp.lapse_rate, fit_per_consonant_nuisance=False
            )
            p2_bio = normalize_rows(c2 + 2.2 * mu_bio2)

            prof_i = qcls_map[listeners_meta[i]["listener_id"]]
            l1 = BiophysicalVirtualListener("S1", "CV", "", prof_i, bank, hp_cv, eff_1, so_cv, unk_1)
            l2 = BiophysicalVirtualListener("S2", "CV", "", prof_i, bank, hp_cv, eff_2, so_cv, unk_2)
            d1_dec = compute_snr_deficit_decomposition_db(l1, nh_cv)
            d2_dec = compute_snr_deficit_decomposition_db(l2, nh_cv)
            s1_neural_def.append(d1_dec["total_behavioral_deficit_db"])
            s2_neural_def.append(d2_dec["total_behavioral_deficit_db"])

            # Strictly proper Multi-Category Brier Score (x100 across 10 rows) predicting held-out S2 trials:
            # For each trial in S2 with outcome r, sum_k (p1[c, k] - 1_{k==r})^2:
            def session_brier_x100(p_pred: np.ndarray, c_holdout: np.ndarray) -> float:
                row_n = c_holdout.sum(axis=1, keepdims=True)
                # sum_{r} c_{c,r} * (||p_c||^2 - 2 p_{c,r} + 1) / total_trials * 100
                p_sq = np.sum(p_pred ** 2, axis=1, keepdims=True)
                tot_loss = np.sum(row_n * p_sq - 2.0 * np.sum(c_holdout * p_pred, axis=1, keepdims=True) + row_n)
                return float(tot_loss / max(c_holdout.sum(), 1.0) * 100.0)

            brier_raw_l.append(session_brier_x100(p1_raw, c2))
            brier_sym_l.append(session_brier_x100(p1_sym, c2))
            brier_bio_l.append(session_brier_x100(p1_bio, c2))

            retest_tvd_raw_l.append(float(0.5 * np.mean(np.sum(np.abs(p1_raw - p2_emp), axis=1))))
            retest_tvd_sym_l.append(float(0.5 * np.mean(np.sum(np.abs(p1_sym - p2_sym), axis=1))))
            retest_tvd_bio_l.append(float(0.5 * np.mean(np.sum(np.abs(p1_bio - p2_bio), axis=1))))

            nll_raw_l.append(-float(np.sum(c2 * np.log(np.maximum(p1_raw_nll, 1e-12)))))
            nll_sym_l.append(-float(np.sum(c2 * np.log(np.maximum(p1_sym, 1e-12)))))
            nll_bio_l.append(-float(np.sum(c2 * np.log(np.maximum(p1_bio, 1e-12)))))

    # Characterize the 67 aided human sessions in QVC_perf.mat!
    aided_rows: list[dict[str, Any]] = []
    perf_mat_path = Path("/Users/neely/STN/2026_Quick_Tests/qtest_dataset/qVCV_analysis/QVC_perf.mat")
    if perf_mat_path.exists():
        mp = loadmat(str(perf_mat_path))
        ids_p = ["".join(ch).strip() for ch in mp["id"]]
        ccms_p = mp["ccms"]
        by_unaided_id = {r["listener_id"]: idx for idx, r in enumerate(listeners_meta)}
        for idx_p, aid_id in enumerate(ids_p):
            if len(aid_id) >= 3 and aid_id[1] in ("A", "P"):
                u_id = aid_id[0] + "L" + aid_id[2:]
                if u_id not in by_unaided_id:
                    continue
                ui = by_unaided_id[u_id]
                prof_u = qcls_map[u_id]
                c_aid = np.asarray(ccms_p[idx_p], dtype=float)
                aid_type = "Lab_Aid" if aid_id[1] == "A" else "Personal_Aid"

                # Fit effective NAL-NL2 gain restoration fraction (alpha_gain in [0, 1]) + HF rolloff + WDRC smear
                # while holding the listener's unaided cochlear & neural parameters fixed, OR measure residual aided deficit!
                # Prescribed half-gain NAL baseline with realistic receiver roll-off at 6-8 kHz:
                bw_roll = np.array([0.45, 0.50, 0.52, 0.55, 0.55, 0.55, 0.52, 0.48, 0.30, 0.18], dtype=float)
                ig_nom = bw_roll * np.maximum(prof_u.audiogram_hl_db, 0.0)
                nd_a, ac_a, _ = compute_peripheral_template_features(
                    bank, REF_SNR_ARR, prof_u.audiogram_hl_db[None, :], prof_u.hl_ohc_db[None, :], hp,
                    ha_insertion_gain_n10=ig_nom[None, :], ha_wdrc_smear_factor=0.25,
                )
                eff_a, unk_a, _ = fit_single_listener_biophysical(
                    c_aid, nd_a, ac_a, self_off, dir_bias, lapse_rate=hp.lapse_rate, fit_per_consonant_nuisance=False
                )
                bl_aid = BiophysicalVirtualListener(
                    aid_id, unaided_rows[ui]["group"], aid_type, prof_u, bank, hp, eff_a, self_off, unk_a,
                    ha_insertion_gain_db=ig_nom, ha_wdrc_smear=0.25,
                )
                dec_aid = compute_snr_deficit_decomposition_db(bl_aid, nh_ref_lstn)
                dec_un = unaided_rows[ui]
                aided_rows.append({
                    "aided_session_id": aid_id,
                    "unaided_listener_id": u_id,
                    "aid_type": aid_type,
                    "group": dec_un["group"],
                    "unaided_total_deficit_db": dec_un["total_behavioral_deficit_db"],
                    "aided_total_deficit_db": dec_aid["total_behavioral_deficit_db"],
                    "net_snr_benefit_db": round(dec_un["total_behavioral_deficit_db"] - dec_aid["total_behavioral_deficit_db"], 3),
                    "remaining_unrestored_audibility_db": dec_aid["threshold_audibility_loss_db"],
                    "remaining_ohc_broadening_usom_db": dec_aid["ohc_broadening_usom_loss_db"],
                    "aided_suprathreshold_distortion_db": dec_aid["neural_suprathreshold_deficit_db"],
                })

    print("[5/5] Validating simulated cohort benchmarks & generating diagnostic plots...", flush=True)
    # Simulate 30 independent pairs of 100-trial sessions on the 98 BiophysicalVirtualListeners:
    rng_sim = np.random.default_rng(20267777)
    aroc_all, aroc_sl, aroc_mm, ppta_r, ppta_mad = [], [], [], [], []
    sim_cohort_ccm = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
    sl_mask = groups == "Slight"
    mm_mask = (groups == "Mild") | (groups == "Moderate")

    for _ in range(30):
        s1_sim = np.zeros((n_subj, N_STIMULI, N_RESPONSES), dtype=float)
        s2_sim = np.zeros((n_subj, N_STIMULI, N_RESPONSES), dtype=float)
        for i, fvl in enumerate(fixed_vls):
            for tok_idx in (0, 1):
                p_tok = fvl.token_matrices[tok_idx]
                for c in range(N_STIMULI):
                    s1_sim[i, c] += rng_sim.multinomial(5, p_tok[c])
                    s2_sim[i, c] += rng_sim.multinomial(5, p_tok[c])
        sim_cohort_ccm += (s1_sim.sum(axis=0) + s2_sim.sum(axis=0)) / 30.0
        for s_mat in (s1_sim, s2_sim):
            aroc_all.append(loocv_hellinger_aroc_unbiased(s_mat, is_nh, ~is_nh))
            aroc_sl.append(loocv_hellinger_aroc_unbiased(s_mat, is_nh, sl_mask))
            aroc_mm.append(loocv_hellinger_aroc_unbiased(s_mat, is_nh, mm_mask))
        pp1 = predict_ppta_mnr(s1_sim, B_mnr, C_mnr, ca_mnr, pa_mnr)
        pp2 = predict_ppta_mnr(s2_sim, B_mnr, C_mnr, ca_mnr, pa_mnr)
        ppta_r.append(float(np.corrcoef(pp1, pp2)[0, 1]))
        ppta_mad.append(float(np.mean(np.abs(pp1 - pp2))))

    # Compare 90 off-diagonal error counts: Human vs. Biophysical vs. Old Symmetric Feature Model!
    hum_cohort_ccm = pooled_counts.sum(axis=0)
    sym_cohort_ccm = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
    for i in range(n_subj):
        _, mu_s = fit_feature_choice_model(pooled_counts[i])
        sym_cohort_ccm += 20.0 * mu_s

    off_mask = ~np.eye(N_STIMULI, dtype=bool)
    hum_off_90 = hum_cohort_ccm[:, :N_STIMULI][off_mask]
    bio_off_90 = sim_cohort_ccm[:, :N_STIMULI][off_mask]
    sym_off_90 = sym_cohort_ccm[:, :N_STIMULI][off_mask]
    pair_labels_90 = [
        f"{STIMULI_CONSONANTS[r_i]}→{STIMULI_CONSONANTS[c_j]}"
        for r_i in range(N_STIMULI)
        for c_j in range(N_STIMULI)
        if r_i != c_j
    ]

    def pair_counts(mat_10x11: np.ndarray, a: str, b: str) -> tuple[float, float]:
        ia, ib = STIMULI_CONSONANTS.index(a), STIMULI_CONSONANTS.index(b)
        return round(float(mat_10x11[ia, ib]), 1), round(float(mat_10x11[ib, ia]), 1)

    cv_summary = {
        "s1_to_s2_brier_raw": round(float(np.mean(brier_raw_l)), 3),
        "s1_to_s2_brier_sym_eb": round(float(np.mean(brier_sym_l)), 3),
        "s1_to_s2_brier_biophys_eb": round(float(np.mean(brier_bio_l)), 3),
        "s1_to_s2_nll_raw": round(float(np.mean(nll_raw_l)), 2),
        "s1_to_s2_nll_sym_eb": round(float(np.mean(nll_sym_l)), 2),
        "s1_to_s2_nll_biophys_eb": round(float(np.mean(nll_bio_l)), 2),
        "retest_s1_vs_s2_tvd_raw": round(float(np.mean(retest_tvd_raw_l)), 4),
        "retest_s1_vs_s2_tvd_sym_eb": round(float(np.mean(retest_tvd_sym_l)), 4),
        "retest_s1_vs_s2_tvd_biophys_eb": round(float(np.mean(retest_tvd_bio_l)), 4),
        "human_s1_vs_s2_biophys_deficit_r": round(float(np.corrcoef(s1_neural_def, s2_neural_def)[0, 1]), 4),
        "human_s1_vs_s2_biophys_deficit_mad_db": round(float(np.mean(np.abs(np.array(s1_neural_def) - np.array(s2_neural_def)))), 3),
    }

    validation_summary = {
        "elapsed_sec": round(time.time() - t0, 2),
        "n_unaided_listeners": n_subj,
        "n_paired_qcls_listeners": n_qcls,
        "n_aided_sessions_characterized": len(aided_rows),
        "off_diagonal_90cell_correlation_r": {
            "previous_symmetric_feature_model": round(float(np.corrcoef(hum_off_90, sym_off_90)[0, 1]), 4),
            "biophysical_asymmetric_model": round(float(np.corrcoef(hum_off_90, bio_off_90)[0, 1]), 4),
        },
        "directional_confusion_pairs_human_vs_biophys_vs_symmetric": {
            f"{a}_vs_{b}": {
                "human": pair_counts(hum_cohort_ccm, a, b),
                "biophysical": pair_counts(sim_cohort_ccm, a, b),
                "symmetric_feature": pair_counts(sym_cohort_ccm, a, b),
            }
            for a, b in [("V", "Z"), ("K", "T"), ("S", "T"), ("D", "G"), ("B", "D")]
        },
        "simulated_vs_human_100trial_benchmarks": {
            "aroc_nh_vs_all_hl": {"human": 0.8858, "biophysical_sim": round(float(np.mean(aroc_all)), 4)},
            "aroc_nh_vs_slight": {"human": 0.7476, "biophysical_sim": round(float(np.mean(aroc_sl)), 4)},
            "aroc_nh_vs_mild_mod": {"human": 0.9597, "biophysical_sim": round(float(np.mean(aroc_mm)), 4)},
            "ppta_test_retest_r": {"human": 0.9628, "biophysical_sim": round(float(np.mean(ppta_r)), 4)},
            "ppta_test_retest_mad_db": {"human": 2.52, "biophysical_sim": round(float(np.mean(ppta_mad)), 2)},
        },
        "out_of_sample_lolo_s1_to_s2_characterization": cv_summary,
        "group_mean_deficit_decomposition_db_snr": {
            g: {
                "n": sum(1 for r in unaided_rows if r["group"] == g),
                "total_behavioral_db": round(float(np.mean([r["total_behavioral_deficit_db"] for r in unaided_rows if r["group"] == g])), 2),
                "threshold_audibility_db": round(float(np.mean([r["threshold_audibility_loss_db"] for r in unaided_rows if r["group"] == g])), 2),
                "qcls_ohc_broadening_usom_db": round(float(np.mean([r["ohc_broadening_usom_loss_db"] for r in unaided_rows if r["group"] == g])), 2),
                "neural_suprathreshold_db": round(float(np.mean([r["neural_suprathreshold_deficit_db"] for r in unaided_rows if r["group"] == g])), 2),
            }
            for g in ("Normal", "Slight", "Mild", "Moderate")
        },
        "aided_benefit_summary_db_snr": {
            at: {
                "n_sessions": sum(1 for r in aided_rows if r["aid_type"] == at),
                "mean_unaided_total_db": round(float(np.mean([r["unaided_total_deficit_db"] for r in aided_rows if r["aid_type"] == at])), 2),
                "mean_aided_total_db": round(float(np.mean([r["aided_total_deficit_db"] for r in aided_rows if r["aid_type"] == at])), 2),
                "mean_net_benefit_db": round(float(np.mean([r["net_snr_benefit_db"] for r in aided_rows if r["aid_type"] == at])), 2),
                "mean_remaining_audibility_db": round(float(np.mean([r["remaining_unrestored_audibility_db"] for r in aided_rows if r["aid_type"] == at])), 2),
                "mean_remaining_ohc_usom_db": round(float(np.mean([r["remaining_ohc_broadening_usom_db"] for r in aided_rows if r["aid_type"] == at])), 2),
                "mean_aided_distortion_db": round(float(np.mean([r["aided_suprathreshold_distortion_db"] for r in aided_rows if r["aid_type"] == at])), 2),
            }
            for at in ("Personal_Aid", "Lab_Aid")
            if any(r["aid_type"] == at for r in aided_rows)
        },
    }

    human_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    save_cohort_csv(fixed_vls, human_dir / "biophysical_listeners_98.csv")
    write_dict_rows_csv(unaided_rows, human_dir / "human_characterization_98_unaided.csv")
    if aided_rows:
        write_dict_rows_csv(aided_rows, human_dir / "human_characterization_aided_sessions.csv")

    full_bio_export = {
        "model": "qVCV Biophysical Spectro-Temporal + qCLS Coupled Listener Model",
        "n_listeners": n_subj,
        "n_paired_qcls": n_qcls,
        "self_clarity_offset_nh": np.round(self_off, 5).tolist(),
        "biophysical_hyperparams": hp.to_json_dict(),
        "acoustic_bank_summary": bank.to_json_dict(),
        "listeners": bio_json_records,
    }
    (human_dir / "biophysical_listeners_98.json").write_text(json.dumps(full_bio_export, indent=2), encoding="utf-8")
    (res_dir / "biophysical_model_validation.json").write_text(json.dumps(validation_summary, indent=2), encoding="utf-8")

    # Pick a representative Moderate HL listener with paired qCLS for Panel D:
    mod_candidates = [bl for bl in bio_listeners if bl.archetype == "Moderate" and bl.cochlear_profile.has_measured_qcls]
    mod_ex = mod_candidates[0] if mod_candidates else bio_listeners[-1]

    plot_biophysical_validation_figure(
        hum_off_90,
        bio_off_90,
        sym_off_90,
        pair_labels_90,
        unaided_rows,
        cv_summary,
        nh_ref_lstn,
        mod_ex,
        res_dir / "fig_biophysical_listener_model.png",
    )

    print(json.dumps(validation_summary, indent=2))


if __name__ == "__main__":
    main()
