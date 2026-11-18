#!/usr/bin/env python3
"""
exp_human_calibrated_study.py

Comprehensive qVCV tracking & multi-SNR simulation study on the 98 virtual
listeners calibrated directly to the empirical human dataset in
``~/STN/2026_Quick_Tests/qtest_dataset`` (32 Normal, 23 Slight, 26 Mild,
17 Moderate; 19,600 unaided trials + Practice +SNR block scores).

Evaluates:
  1. Human-vs-Simulated Calibration Fidelity (Test & Practice PC%, per-consonant
     NH/HL accuracy, 100-trial Session 1 vs. Session 2 repeatability on PC% and
     clinical PPTA, and LOOCV Hellinger A_ROC).
  2. Adaptive Tracking vs. Static Block / Random Selection on the 98 human
     listeners (Fixed Ref SNR and Multi-SNR sets across N = 20..100).
  3. Single Optimal SNR (5 reps @ Ref) vs. 5-SNR Stimulus Sets (symmetric,
     downward, upward, and shared absolute grids) across:
       - LOOCV Hellinger A_ROC (NH vs All HL, NH vs Slight HL, NH vs Mild+Mod HL,
         Mild vs Moderate HL)
       - LOOCV Audiometric PTA Prediction MAE (dB HL)
       - Consonant SRT50 MAE (dB) & 3D SNR Confusion-Surface TVD (%)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qvcv_mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from export_human_listeners import (
    compute_aroc,
    loocv_hellinger_aroc_unbiased,
    loocv_hellinger_scores,
    predict_ppta_mnr,
)
from qvcv_multisnr_designs import (
    EVAL_REL_SNRS_DB,
    MultiSNRTracker,
    StimulusSetDesign,
    fit_multisnr_psychometric_model,
)
from qvcv_multisnr_simulator import (
    HumanCalibratedMultiSNRListener,
    load_human_multisnr_cohort,
)
from qvcv_simulator import (
    CONSONANT_SNR_DB,
    N_RESPONSES,
    N_STIMULI,
    STIMULI_CONSONANTS,
    load_ground_truth,
    relative_information_transmitted,
)
from qvcv_trackers import QVCVTrackerAdapter
from qvcv_tracking_sim import write_dict_rows_csv


def total_variation_distance_pct(est: np.ndarray, gt: np.ndarray) -> float:
    return float(50.0 * np.mean(np.sum(np.abs(est - gt), axis=1)))

REF_SNR_VEC = np.array([CONSONANT_SNR_DB[c] for c in STIMULI_CONSONANTS], dtype=float)
EVAL_POINTS = [20, 40, 60, 80, 100]


def build_human_study_designs() -> list[StimulusSetDesign]:
    """8 candidate 100-stimulus qVCV designs (10 consonants x 2 tokens x 5 SNR slots)."""
    specs = [
        ("single_ref",     "Single SNR (5x @ Ref)",          [0.0, 0.0, 0.0, 0.0, 0.0],     True),
        ("rel_pm2",        "Rel ±2 dB (-2..+2, 1 dB)",       [-2.0, -1.0, 0.0, 1.0, 2.0],   True),
        ("rel_pm4",        "Rel ±4 dB (-4..+4, 2 dB)",       [-4.0, -2.0, 0.0, 2.0, 4.0],   True),
        ("rel_pm6",        "Rel ±6 dB (-6..+6, 3 dB)",       [-6.0, -3.0, 0.0, 3.0, 6.0],   True),
        ("rel_pm9",        "Rel ±9 dB (-9..+9, 4.5 dB)",     [-9.0, -4.5, 0.0, 4.5, 9.0],   True),
        ("rel_down_m6_0",  "Rel Downward (-6..0, 1.5 dB)",   [-6.0, -4.5, -3.0, -1.5, 0.0], True),
        ("rel_up_m3_p9",   "Rel Upward (-3..+9, 3 dB)",      [-3.0, 0.0, 3.0, 6.0, 9.0],    True),
        ("abs_shared",     "Abs Shared (-6..+18, 6 dB)",     [-6.0, 0.0, 6.0, 12.0, 18.0],  False),
    ]
    out: list[StimulusSetDesign] = []
    for name, label, vals, is_rel in specs:
        v = np.asarray(vals, dtype=float)[None, :]
        grid = (REF_SNR_VEC[:, None] + v) if is_rel else np.repeat(v, N_STIMULI, axis=0)
        out.append(StimulusSetDesign(name=name, label=label, snr_grid=grid))
    return out


def loocv_ridge_pta_mae(
    ccms_n_10_11: np.ndarray,
    true_pta: np.ndarray,
    knee_db: float = 16.32,
    ridge_lam: float = 120.0,
) -> tuple[float, float]:
    """
    Leave-one-out cross-validated PTA prediction from the normalized 10x11
    confusion matrix (fairly retrained per stimulus-set design), returning:
      (MAE vs audibility-clamped max(knee_db, PTA), MAE vs raw PTA on HI ears).
    """
    n = ccms_n_10_11.shape[0]
    totals = np.maximum(ccms_n_10_11.sum(axis=(1, 2), keepdims=True), 1e-12)
    X = (ccms_n_10_11 * (100.0 / totals)).reshape(n, -1)
    y_aud = np.maximum(knee_db, true_pta)

    preds = np.zeros(n, dtype=float)
    idx_all = np.arange(n)
    reg_eye = ridge_lam * np.eye(X.shape[1], dtype=float)
    for i in range(n):
        tr = idx_all != i
        X_tr = X[tr]
        xm = X_tr.mean(axis=0, keepdims=True)
        Xc = X_tr - xm
        ym = float(y_aud[tr].mean())
        w = np.linalg.solve(Xc.T @ Xc + reg_eye, Xc.T @ (y_aud[tr] - ym))
        preds[i] = ym + float(((X[i : i + 1] - xm) @ w).item())

    mae_aud_all = float(np.mean(np.abs(preds - y_aud)))
    hi_mask = true_pta > 15.0
    mae_raw_hi = float(np.mean(np.abs(preds[hi_mask] - true_pta[hi_mask])))
    return mae_aud_all, mae_raw_hi


def run_fixed_snr_tracking_experiment(
    human_dir: Path,
    n_reps: int = 30,
    seed: int = 20261115,
) -> list[dict[str, Any]]:
    """
    Experiment A: Adaptive vs. Block vs. Random on the 98 human listeners at
    qVCV's fixed reference SNRs (matched feature_eb estimator), measuring TVD,
    IT MAE, clinical PPTA accuracy & test-retest MAD, and LOOCV Hellinger A_ROC.
    """
    gt_profiles = load_ground_truth(human_dir / "human_listeners_98.csv")
    lids = gt_profiles.keys()
    n_subj = len(lids)

    with (human_dir / "mnr_vcv_pta_model.json").open("r", encoding="utf-8") as f:
        mnr = json.load(f)
    B = np.asarray(mnr["B"], dtype=float)
    C = np.asarray(mnr["C"], dtype=float)
    ca = np.asarray(mnr["ca"], dtype=float)
    pa = np.asarray(mnr["pa"], dtype=float)

    with (human_dir / "human_listeners_98.json").open("r", encoding="utf-8") as f:
        raw_json = json.load(f)["listeners"]
    is_nh = np.array([bool(r["is_nh"]) for r in raw_json], dtype=bool)
    is_slight = np.array([r["archetype"] == "Slight" for r in raw_json], dtype=bool)
    is_mm = np.array([r["archetype"] in ("Mild", "Moderate") for r in raw_json], dtype=bool)

    true_mats = np.array([gt_profiles[lid] for lid in lids], dtype=float)
    true_ppta = predict_ppta_mnr(true_mats * 100.0, B, C, ca, pa)
    true_it = np.array(
        [relative_information_transmitted(true_mats[i])["overall"] for i in range(n_subj)],
        dtype=float,
    )

    modes = ["random", "block", "adaptive_var", "adaptive_eig", "adaptive_feature"]
    rows: list[dict[str, Any]] = []

    for m_idx, mode in enumerate(modes):
        est_cohort = {nt: np.zeros((n_reps, n_subj, N_STIMULI, N_RESPONSES), dtype=float) for nt in EVAL_POINTS}
        for rep in range(n_reps):
            for li, lid in enumerate(lids):
                sseq = np.random.SeedSequence([seed, 10, m_idx, rep, li])
                l_seed, t_seed = [int(s.generate_state(1)[0]) for s in sseq.spawn(2)]
                sim_l = gt_profiles.new_virtual_listener(lid, np.random.default_rng(l_seed))
                adapter = QVCVTrackerAdapter(mode=mode, estimator="feature_eb")
                fits = adapter.run_session_at_points(sim_l, EVAL_POINTS, seed=t_seed)
                for nt in EVAL_POINTS:
                    est_cohort[nt][rep, li] = fits[nt]

        for nt in EVAL_POINTS:
            mats = est_cohort[nt]  # (n_reps, 98, 10, 11)
            tvd_vals = []
            it_errs = []
            ppta_by_rep = np.zeros((n_reps, n_subj), dtype=float)
            aroc_all_l, aroc_sl_l, aroc_mm_l = [], [], []

            for rep in range(n_reps):
                m_rep = mats[rep]
                for li in range(n_subj):
                    tvd_vals.append(total_variation_distance_pct(m_rep[li], true_mats[li]))
                    it_errs.append(abs(relative_information_transmitted(m_rep[li])["overall"] - true_it[li]))
                ppta_hat = predict_ppta_mnr(m_rep * 100.0, B, C, ca, pa)
                ppta_by_rep[rep] = ppta_hat

                pseudo_counts = m_rep * (nt / 10.0)
                aroc_all_l.append(loocv_hellinger_aroc_unbiased(pseudo_counts, is_nh, ~is_nh))
                aroc_sl_l.append(loocv_hellinger_aroc_unbiased(pseudo_counts, is_nh, is_slight))
                aroc_mm_l.append(loocv_hellinger_aroc_unbiased(pseudo_counts, is_nh, is_mm))

            ppta_mae = float(np.mean(np.abs(ppta_by_rep - true_ppta[None, :])))
            retest_mad = float(np.mean(np.abs(ppta_by_rep[0::2] - ppta_by_rep[1::2])))
            rows.append({
                "mode": mode,
                "n_trials": nt,
                "tvd_pct": round(float(np.mean(tvd_vals)), 3),
                "it_overall_mae_pct": round(float(np.mean(it_errs)), 3),
                "ppta_mae_vs_asymp_db": round(ppta_mae, 3),
                "ppta_retest_mad_db": round(retest_mad, 3),
                "aroc_nh_vs_all_hl": round(float(np.mean(aroc_all_l)), 4),
                "aroc_nh_vs_slight_hl": round(float(np.mean(aroc_sl_l)), 4),
                "aroc_nh_vs_mild_mod_hl": round(float(np.mean(aroc_mm_l)), 4),
            })
    return rows


def run_multisnr_human_design_experiment(
    cohort: list[HumanCalibratedMultiSNRListener],
    designs: list[StimulusSetDesign],
    n_reps: int = 24,
    seed: int = 20261116,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """
    Experiment B: Single Optimal SNR vs. 5-SNR Stimulus Sets (and Multi-SNR
    Adaptive vs. Block) on the 98 human-calibrated multi-SNR listeners.
    """
    n_subj = len(cohort)
    is_nh = np.array([l.is_nh for l in cohort], dtype=bool)
    is_slight = np.array([l.archetype == "Slight" for l in cohort], dtype=bool)
    is_mild = np.array([l.archetype == "Mild" for l in cohort], dtype=bool)
    is_mod = np.array([l.archetype == "Moderate" for l in cohort], dtype=bool)
    is_mm = is_mild | is_mod
    true_pta = np.array([l.pta_124_db for l in cohort], dtype=float)

    eval_grid = REF_SNR_VEC[:, None] + EVAL_REL_SNRS_DB[None, :]  # (10, 7)
    true_srt50 = np.zeros((n_subj, N_STIMULI), dtype=float)
    true_surf = np.zeros((n_subj, N_STIMULI, len(EVAL_REL_SNRS_DB), N_RESPONSES), dtype=float)
    for li, l in enumerate(cohort):
        for ci in range(N_STIMULI):
            true_srt50[li, ci] = l.true_srt50_db(ci)
            for si, snr_v in enumerate(eval_grid[ci]):
                true_surf[li, ci, si] = l.response_probs(ci, float(snr_v), token_idx=None)

    # Population Hellinger distance curves across Delta_SNR in [-15, +15] dB
    dsnr_sweep = np.linspace(-15.0, 15.0, 61)
    hellinger_curves: dict[str, list[float]] = {c: [] for c in STIMULI_CONSONANTS}
    hellinger_slight_mean: list[float] = []
    for dsnr in dsnr_sweep:
        h_sl_all = []
        for ci, sc in enumerate(STIMULI_CONSONANTS):
            snr_v = float(REF_SNR_VEC[ci] + dsnr)
            p_nh = np.mean([cohort[i].response_probs(ci, snr_v) for i in np.where(is_nh)[0]], axis=0)
            p_hl = np.mean([cohort[i].response_probs(ci, snr_v) for i in np.where(~is_nh)[0]], axis=0)
            p_sl = np.mean([cohort[i].response_probs(ci, snr_v) for i in np.where(is_slight)[0]], axis=0)
            h_c = float(np.sqrt(0.5 * np.sum((np.sqrt(p_nh) - np.sqrt(p_hl)) ** 2)))
            h_sl = float(np.sqrt(0.5 * np.sum((np.sqrt(p_nh) - np.sqrt(p_sl)) ** 2)))
            hellinger_curves[sc].append(round(h_c, 5))
            h_sl_all.append(h_sl)
        hellinger_slight_mean.append(round(float(np.mean(h_sl_all)), 5))

    design_rows: list[dict[str, Any]] = []

    for d_idx, design in enumerate(designs):
        pop_mats = np.array([l.pooled_matrix_for_design(design.snr_grid) for l in cohort])
        p_nh_pop = pop_mats[is_nh].mean(axis=0)
        p_hl_pop = pop_mats[~is_nh].mean(axis=0)
        p_sl_pop = pop_mats[is_slight].mean(axis=0)
        pop_h_nh_hl = float(np.mean(np.sqrt(0.5 * np.sum((np.sqrt(p_nh_pop) - np.sqrt(p_hl_pop)) ** 2, axis=1))))
        pop_h_nh_sl = float(np.mean(np.sqrt(0.5 * np.sum((np.sqrt(p_nh_pop) - np.sqrt(p_sl_pop)) ** 2, axis=1))))

        tracker = MultiSNRTracker(design=design, mode="block")
        pooled_by_nt = {nt: np.zeros((n_reps, n_subj, N_STIMULI, N_RESPONSES), dtype=float) for nt in EVAL_POINTS}
        surf_tvd_nt: dict[int, list[float]] = {nt: [] for nt in EVAL_POINTS}
        srt_mae_nt: dict[int, list[float]] = {nt: [] for nt in EVAL_POINTS}

        for rep in range(n_reps):
            fit_psych = rep < 10
            for li, l in enumerate(cohort):
                sseq = np.random.SeedSequence([seed, 20, d_idx, rep, li])
                c_rng, t_seed_rng = [np.random.default_rng(s) for s in sseq.spawn(2)]
                sim_l = l.clone_with_rng(c_rng, draw_session_shift=True)
                t_seed = int(t_seed_rng.integers(0, 2**31 - 1))
                outs = tracker.run_session_at_points(sim_l, EVAL_POINTS, seed=t_seed, fit_psychometric=fit_psych)
                for nt in EVAL_POINTS:
                    pooled_by_nt[nt][rep, li] = outs[nt].cell_counts.sum(axis=1)
                    if fit_psych and nt in (40, 100):
                        pm = outs[nt].psychometric_fit
                        pred_g = pm.predict_grid(eval_grid)
                        surf_tvd_nt[nt].append(float(50.0 * np.mean(np.sum(np.abs(pred_g - true_surf[li]), axis=-1))))
                        srt_mae_nt[nt].append(float(np.mean(np.abs(pm.predict_all_srt50_db() - true_srt50[li]))))

        for nt in EVAL_POINTS:
            aroc_all, aroc_sl, aroc_mm, aroc_stage = [], [], [], []
            pta_mae_aud, pta_mae_hi = [], []
            for rep in range(n_reps):
                pc = pooled_by_nt[nt][rep]
                _, d_stg_diff = loocv_hellinger_scores(pc, is_mild, is_mod)

                aroc_all.append(loocv_hellinger_aroc_unbiased(pc, is_nh, ~is_nh))
                aroc_sl.append(loocv_hellinger_aroc_unbiased(pc, is_nh, is_slight))
                aroc_mm.append(loocv_hellinger_aroc_unbiased(pc, is_nh, is_mm))
                aroc_stage.append(compute_aroc(d_stg_diff[is_mild], d_stg_diff[is_mod]))

                m_aud, m_hi = loocv_ridge_pta_mae(pc, true_pta)
                pta_mae_aud.append(m_aud)
                pta_mae_hi.append(m_hi)

            design_rows.append({
                "design": design.name,
                "description": design.label,
                "n_trials": nt,
                "pop_hellinger_nh_hl": round(pop_h_nh_hl, 4),
                "pop_hellinger_nh_slight": round(pop_h_nh_sl, 4),
                "aroc_nh_vs_all_hl": round(float(np.mean(aroc_all)), 4),
                "aroc_nh_vs_slight_hl": round(float(np.mean(aroc_sl)), 4),
                "aroc_nh_vs_mild_mod_hl": round(float(np.mean(aroc_mm)), 4),
                "aroc_mild_vs_mod_hl": round(float(np.mean(aroc_stage)), 4),
                "loocv_pta_mae_all_db": round(float(np.mean(pta_mae_aud)), 3),
                "loocv_pta_mae_hi_db": round(float(np.mean(pta_mae_hi)), 3),
                "surface_3d_tvd_pct": round(float(np.mean(surf_tvd_nt[nt])), 3) if surf_tvd_nt[nt] else None,
                "srt50_mae_db": round(float(np.mean(srt_mae_nt[nt])), 3) if srt_mae_nt[nt] else None,
            })

    # Compare Multi-SNR selection modes (random vs block vs adaptive_consonant vs adaptive_2d_snr) on rel_pm4
    rel_pm4 = [d for d in designs if d.name == "rel_pm4"][0]
    multisnr_mode_rows: list[dict[str, Any]] = []
    for m_idx, mode in enumerate(("random", "block", "adaptive_consonant", "adaptive_2d_snr")):
        tracker = MultiSNRTracker(design=rel_pm4, mode=mode)
        n_r = 16 if mode == "adaptive_2d_snr" else n_reps
        norm_by_nt = {nt: np.zeros((n_r, n_subj, N_STIMULI, N_RESPONSES), dtype=float) for nt in EVAL_POINTS}
        srt_by_nt: dict[int, list[float]] = {nt: [] for nt in EVAL_POINTS}

        for rep in range(n_r):
            need_fit = (mode == "adaptive_2d_snr") or (rep < 10)
            for li, l in enumerate(cohort):
                sseq = np.random.SeedSequence([seed, 30, m_idx, rep, li])
                c_rng, t_seed_rng = [np.random.default_rng(s) for s in sseq.spawn(2)]
                sim_l = l.clone_with_rng(c_rng, draw_session_shift=True)
                t_seed = int(t_seed_rng.integers(0, 2**31 - 1))
                outs = tracker.run_session_at_points(sim_l, EVAL_POINTS, seed=t_seed, fit_psychometric=need_fit)
                for nt in EVAL_POINTS:
                    if mode == "adaptive_2d_snr":
                        pm = outs[nt].psychometric_fit
                        norm_by_nt[nt][rep, li] = pm.predict_grid(rel_pm4.snr_grid).mean(axis=1) * (nt / 10.0)
                    else:
                        cnt = outs[nt].cell_counts.sum(axis=1)
                        row_n = np.maximum(cnt.sum(axis=1, keepdims=True), 1.0)
                        norm_by_nt[nt][rep, li] = np.where(cnt.sum(axis=1, keepdims=True) > 0, cnt / row_n, outs[nt].pooled_dirichlet) * (nt / 10.0)
                    if rep < 10 and nt in (40, 100):
                        srt_by_nt[nt].append(float(np.mean(np.abs(outs[nt].psychometric_fit.predict_all_srt50_db() - true_srt50[li]))))

        for nt in EVAL_POINTS:
            aroc_all, aroc_sl, pta_mae_all = [], [], []
            for rep in range(n_r):
                nc = norm_by_nt[nt][rep]
                aroc_all.append(loocv_hellinger_aroc_unbiased(nc, is_nh, ~is_nh))
                aroc_sl.append(loocv_hellinger_aroc_unbiased(nc, is_nh, is_slight))
                m_aud, _ = loocv_ridge_pta_mae(nc, true_pta)
                pta_mae_all.append(m_aud)

            multisnr_mode_rows.append({
                "design": "rel_pm4",
                "mode": mode,
                "n_trials": nt,
                "aroc_nh_vs_all_hl": round(float(np.mean(aroc_all)), 4),
                "aroc_nh_vs_slight_hl": round(float(np.mean(aroc_sl)), 4),
                "loocv_pta_mae_all_db": round(float(np.mean(pta_mae_all)), 3),
                "srt50_mae_db": round(float(np.mean(srt_by_nt[nt])), 3) if srt_by_nt[nt] else None,
            })

    extra_info = {
        "dsnr_sweep_db": dsnr_sweep.tolist(),
        "hellinger_nh_hl_by_consonant": hellinger_curves,
        "hellinger_nh_slight_mean": hellinger_slight_mean,
    }
    return design_rows, multisnr_mode_rows, extra_info


def plot_human_calibrated_summary(
    val_report: dict[str, Any],
    fixed_rows: list[dict[str, Any]],
    design_rows: list[dict[str, Any]],
    extra_info: dict[str, Any],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5), dpi=160)
    fig.suptitle(
        "Human-Calibrated qVCV Simulation Study (98 Listeners: 32 NH + 66 HL, 19,600 Empirical Trials)",
        fontsize=13.5,
        fontweight="bold",
        y=0.98,
    )

    # Panel A: Calibration Fidelity
    ax = axes[0, 0]
    grp_names = ["Normal\n(n=32)", "Slight\n(n=23)", "Mild\n(n=26)", "Moderate\n(n=17)"]
    grp_keys = ["Normal", "Slight", "Mild", "Moderate"]
    x = np.arange(len(grp_keys))
    w = 0.19
    gm = val_report["group_means_test_and_practice"]
    ax.bar(x - 1.5 * w, [gm[k]["human_test_pc_mean"] for k in grp_keys], w, label="Human Test (Ref SNR)", color="#1f4e79")
    ax.bar(x - 0.5 * w, [gm[k]["sim_test_pc_mean"] for k in grp_keys], w, label="Calibrated Sim Test", color="#5b9bd5")
    ax.bar(x + 0.5 * w, [gm[k]["human_practice_pc_mean"] for k in grp_keys], w, label="Human Practice (60 dB SNR)", color="#c55a11")
    ax.bar(x + 1.5 * w, [gm[k]["sim_practice_pc_mean"] for k in grp_keys], w, label="Calibrated Sim (60 dB SNR)", color="#f4b183")
    ax.set_xticks(x)
    ax.set_xticklabels(grp_names)
    ax.set_ylim(28, 107)
    ax.set_ylabel("Consonant Identification (%)")
    ax.set_title("A. Calibration vs. 98 Humans (Ref-SNR Test & 60-SNR Practice)", fontweight="bold", fontsize=11)
    rep = val_report["repeatability_100_trials"]
    aroc_v = val_report["loocv_hellinger_aroc_100_trials"]
    txt = (
        f"100-Trial Repeatability & LOOCV A_ROC:\n"
        f"• PC% MAD: Human {rep['human_pc_mad_pct']:.2f}% vs Sim {rep['sim_pc_mad_pct']:.2f}% (r={rep['human_pc_r']:.3f} vs {rep['sim_pc_r']:.3f})\n"
        f"• PPTA MAD: Human {rep['human_ppta_mad_db']:.2f} dB vs Sim {rep['sim_ppta_mad_db']:.2f} dB (r={rep['human_ppta_r']:.3f} vs {rep['sim_ppta_r']:.3f})\n"
        f"• A_ROC (NH vs All HL): Human {aroc_v['human_nh_vs_all_hl']:.3f} vs Sim {aroc_v['sim_nh_vs_all_hl']:.3f}"
    )
    ax.text(0.02, 0.03, txt, transform=ax.transAxes, fontsize=8.2, bbox=dict(boxstyle="round,pad=0.35", fc="#f8f9fa", ec="#cccccc"))
    ax.legend(loc="upper right", fontsize=8.1, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Population Hellinger Distance vs. Delta_SNR
    ax = axes[0, 1]
    dsnr = np.asarray(extra_info["dsnr_sweep_db"], dtype=float)
    h_map = extra_info["hellinger_nh_hl_by_consonant"]
    mean_h_all = np.mean([np.asarray(h_map[c]) for c in STIMULI_CONSONANTS], axis=0)
    mean_h_slight = np.asarray(extra_info["hellinger_nh_slight_mean"], dtype=float)
    for sc, col in [("S", "#d62728"), ("D", "#2ca02c"), ("T", "#9467bd"), ("V", "#ff7f0e"), ("B", "#1f77b4"), ("N", "#17becf")]:
        ax.plot(dsnr, h_map[sc], lw=1.3, alpha=0.75, color=col, label=f"/{sc}/ (Ref {int(CONSONANT_SNR_DB[sc]):+d} dB)")
    ax.plot(dsnr, mean_h_all, lw=2.8, color="#111111", label="Mean H(NH, All HL)")
    ax.plot(dsnr, mean_h_slight, lw=2.3, ls="--", color="#b8860b", label="Mean H(NH, Slight HL)")
    ax.axvline(0.0, color="#333333", ls=":", lw=1.4, label="qVCV Ref SNR (ΔSNR = 0 dB)")
    ax.set_xlabel("SNR Offset Relative to qVCV Reference SNR, ΔSNR (dB)")
    ax.set_ylabel("Population Hellinger Distance H(p_NH, p_HL)")
    ax.set_title("B. Empirical Human Hellinger Separation vs. ΔSNR", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.0, ncol=2)
    ax.grid(alpha=0.25)

    # Panel C: Adaptive Tracking vs. Block vs. Random on Fixed Ref SNR
    ax = axes[1, 0]
    mode_styles = [
        ("random", "Random Selection", "#888888", ":"),
        ("block", "Balanced Block (Current qVCV)", "#111111", "-"),
        ("adaptive_var", "Adaptive (Max Row Variance)", "#1f77b4", "--"),
        ("adaptive_eig", "Adaptive (Expected Info Gain)", "#2ca02c", "-."),
        ("adaptive_feature", "Adaptive (Phonetic Feature)", "#d62728", "--"),
    ]
    for m_key, m_lab, col, ls in mode_styles:
        sub = sorted([r for r in fixed_rows if r["mode"] == m_key], key=lambda r: r["n_trials"])
        ax.plot([r["n_trials"] for r in sub], [r["ppta_retest_mad_db"] for r in sub], marker="o", ms=5, lw=2.0, color=col, ls=ls, label=m_lab)
    ax.set_xlabel("Number of Trials (N)")
    ax.set_ylabel("Predicted PTA Test-Retest MAD (dB HL) ↓")
    ax.set_title("C. Adaptive vs. Block Selection: Clinical PPTA Precision", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", fontsize=8.3)
    ax.grid(alpha=0.25)

    # Panel D: Single SNR vs. 5-SNR Designs at N = 100
    ax = axes[1, 1]
    order = ["single_ref", "rel_pm2", "rel_pm4", "rel_pm6", "rel_down_m6_0", "rel_up_m3_p9", "rel_pm9", "abs_shared"]
    short_labels = ["Single\n(Ref)", "Rel\n±2 dB", "Rel\n±4 dB", "Rel\n±6 dB", "Down\n-6..0", "Up\n-3..+9", "Rel\n±9 dB", "Abs\nShared"]
    d100 = {r["design"]: r for r in design_rows if r["n_trials"] == 100}
    xd = np.arange(len(order))
    bw = 0.25
    ax.bar(xd - bw, [d100[k]["aroc_nh_vs_all_hl"] for k in order], bw, label="A_ROC: NH vs All HL", color="#1f4e79")
    ax.bar(xd, [d100[k]["aroc_nh_vs_slight_hl"] for k in order], bw, label="A_ROC: NH vs Slight HL", color="#2e75b6")
    ax.bar(xd + bw, [d100[k]["aroc_mild_vs_mod_hl"] for k in order], bw, label="A_ROC: Mild vs Mod HL", color="#ed7d31")
    ax.set_xticks(xd)
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_ylim(0.55, 1.02)
    ax.set_ylabel("LOOCV Hellinger A_ROC (N = 100 Trials) ↑")
    ax.set_title("D. Single SNR vs. 5-SNR Designs on 98 Human Listeners", fontweight="bold", fontsize=11)

    ax2 = ax.twinx()
    pta_maes = [d100[k]["loocv_pta_mae_all_db"] for k in order]
    ax2.plot(xd, pta_maes, color="#8b0000", marker="D", lw=2.0, ms=5.5, label="LOOCV PTA MAE (dB) ↓")
    ax2.set_ylabel("LOOCV Predicted-PTA MAE (dB HL) ↓", color="#8b0000")
    ax2.tick_params(axis="y", colors="#8b0000")
    ymin_p = max(2.5, float(min(pta_maes)) - 0.5)
    ymax_p = float(max(pta_maes)) + 0.7
    ax2.set_ylim(ymin_p, ymax_p)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=7.7, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.955])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run human-calibrated qVCV tracking & multi-SNR study.")
    parser.add_argument("--n_reps_fixed", type=int, default=20)
    parser.add_argument("--n_reps_multisnr", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "human_calibrated_results"))
    args = parser.parse_args()

    t0 = time.time()
    sim_dir = Path(__file__).resolve().parent
    human_dir = sim_dir / "human_data"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with (human_dir / "calibration_validation.json").open("r", encoding="utf-8") as f:
        val_report = json.load(f)

    print("[1/3] Running Fixed-SNR Adaptive vs. Block Tracking Study on 98 Human Listeners...", flush=True)
    fixed_rows = run_fixed_snr_tracking_experiment(human_dir, n_reps=args.n_reps_fixed)
    write_dict_rows_csv(fixed_rows, out_dir / "human98_fixed_snr_tracking.csv")

    print("[2/3] Running Multi-SNR Stimulus-Set & Hellinger A_ROC Study on 98 Human Listeners...", flush=True)
    cohort = load_human_multisnr_cohort(human_dir / "human_listeners_98.json")
    designs = build_human_study_designs()
    design_rows, multisnr_mode_rows, extra_info = run_multisnr_human_design_experiment(
        cohort, designs, n_reps=args.n_reps_multisnr
    )
    write_dict_rows_csv(design_rows, out_dir / "human98_multisnr_designs.csv")
    write_dict_rows_csv(multisnr_mode_rows, out_dir / "human98_multisnr_tracking_modes.csv")

    print("[3/3] Generating summary figure & JSON report...")
    fig_path = out_dir / "fig_human98_calibrated_study.png"
    plot_human_calibrated_summary(val_report, fixed_rows, design_rows, extra_info, fig_path)

    summary = {
        "elapsed_sec": round(time.time() - t0, 2),
        "calibration_validation": val_report,
        "fixed_snr_100_trials": [r for r in fixed_rows if r["n_trials"] == 100],
        "multisnr_designs_100_trials": [r for r in design_rows if r["n_trials"] == 100],
        "multisnr_modes_100_trials": [r for r in multisnr_mode_rows if r["n_trials"] == 100],
    }
    with (out_dir / "human98_study_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done in {summary['elapsed_sec']:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
