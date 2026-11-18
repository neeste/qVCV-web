#!/usr/bin/env python3
"""
exp_multisnr_study.py

Comprehensive simulation study for qVCV addressing two core design questions:

1. STIMULUS-SET DESIGN (Single-SNR vs. Multi-SNR Sets at N=100 trials):
   Should qVCV present 5 repetitions of 1 fixed SNR per VCV token (current design),
   or 1 presentation at each of 5 distinct SNRs per VCV token (10 consonants x
   2 talker tokens x 5 SNRs = 100 static .wav files)?
   Compares:
     - single_ref : 5 identical reps at current reference SNRs
     - rel_pm3    : Consonant-relative 5-SNR range (-3..+3 dB, 1.5 dB step)
     - rel_pm6    : Consonant-relative 5-SNR range (-6..+6 dB, 3.0 dB step)
     - rel_pm9    : Consonant-relative 5-SNR range (-9..+9 dB, 4.5 dB step)
     - abs_shared : Shared absolute 5-SNR grid (-6, 0, +6, +12, +18 dB)

2. ADAPTIVE TRACKING vs. STRATIFIED BLOCK vs. RANDOM SELECTION:
   Verifies when adaptive tracking (consonant-only or 2D Consonant x SNR)
   does or does NOT improve over simple random or stratified block selection.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from qvcv_multisnr_designs import (
    FittedPsychometricModel,
    MultiSNRTracker,
    StimulusSetDesign,
    build_stimulus_designs,
)
from qvcv_multisnr_simulator import (
    EVAL_REL_SNRS_DB,
    MultiSNRVirtualListener,
    generate_multisnr_cohort,
    save_multisnr_cohort_json,
)
from qvcv_simulator import (
    CONSONANT_SNR_DB,
    N_STIMULI,
    STIMULI_CONSONANTS,
    relative_information_transmitted,
)
from qvcv_tracking_sim import (
    compute_pair_reliability,
    make_rep_seed,
    write_dict_rows_csv,
)

REF_SNR_VEC = np.array([CONSONANT_SNR_DB[c] for c in STIMULI_CONSONANTS], dtype=float)


# --------------------------------------------------------------------------
# Psychometric & Clinical Cohort Metrics
# --------------------------------------------------------------------------

def compute_icc_and_strata(values_LxR: np.ndarray) -> tuple[float, float, float, float]:
    """
    One-way random-effects ANOVA Intraclass Correlation ICC(1,1):
    Returns (icc, sigma_between, sigma_within, n_discriminable_strata).
    """
    n_listeners, n_reps = values_LxR.shape
    listener_means = np.mean(values_LxR, axis=1)
    grand_mean = float(np.mean(listener_means))

    msb = float(n_reps * np.sum((listener_means - grand_mean) ** 2) / max(n_listeners - 1, 1))
    msw = float(np.sum((values_LxR - listener_means[:, None]) ** 2) / max(n_listeners * (n_reps - 1), 1))

    var_within = msw
    var_between = max(0.0, (msb - msw) / n_reps)
    denom = var_between + var_within
    icc = (var_between / denom) if denom > 1e-12 else 0.0

    sd_between = float(np.sqrt(var_between))
    sd_within = float(np.sqrt(var_within))
    sep_ratio = (sd_between / sd_within) if sd_within > 1e-9 else 0.0
    strata = (4.0 * sep_ratio + 1.0) / 3.0
    return float(icc), sd_between, sd_within, float(strata)


EVAL_GRID_10x7: np.ndarray = REF_SNR_VEC[:, None] + EVAL_REL_SNRS_DB[None, :]


def precompute_listener_truth_cache(
    cohort: list[MultiSNRVirtualListener],
) -> dict[str, dict[str, np.ndarray]]:
    cache: dict[str, dict[str, np.ndarray]] = {}
    low_high_grid = REF_SNR_VEC[:, None] + np.array([[-3.0, 9.0]])  # (10, 2)
    for l in cohort:
        surf = np.zeros((N_STIMULI, len(EVAL_REL_SNRS_DB), 11), dtype=float)
        srts = np.zeros(N_STIMULI, dtype=float)
        lh = np.zeros((N_STIMULI, 2, 11), dtype=float)
        for i in range(N_STIMULI):
            srts[i] = l.true_srt50_db(i)
            for si in range(len(EVAL_REL_SNRS_DB)):
                surf[i, si] = l.response_probs(i, float(EVAL_GRID_10x7[i, si]), token_idx=None)
            lh[i, 0] = l.response_probs(i, float(low_high_grid[i, 0]), token_idx=None)
            lh[i, 1] = l.response_probs(i, float(low_high_grid[i, 1]), token_idx=None)
        # Listener's true recognition gain from Ref-3 dB to Ref+9 dB (separates audibility vs distortion)
        true_gain_pct = 100.0 * float(np.mean([lh[i, 1, i] - lh[i, 0, i] for i in range(N_STIMULI)]))
        cache[l.listener_id] = {
            "surf": surf,
            "srts": srts,
            "true_gain_pct": np.array([true_gain_pct]),
        }
    return cache


def evaluate_3d_surface_and_srt(
    psych_fit: FittedPsychometricModel,
    truth_entry: dict[str, np.ndarray],
) -> tuple[float, float, float]:
    """
    Vectorized comparison of fitted SNR-psychometric surface against listener truth:
    Returns (surface_mean_tvd_pct, srt50_mae_db, snr_gain_abs_err_pct).
    """
    pred_surf = psych_fit.predict_grid(EVAL_GRID_10x7)  # (10, 7, 11)
    tvd_pct = 50.0 * float(np.mean(np.sum(np.abs(pred_surf - truth_entry["surf"]), axis=2)))

    est_srts = psych_fit.predict_all_srt50_db()
    srt_mae = float(np.mean(np.abs(est_srts - truth_entry["srts"])))

    low_high_grid = REF_SNR_VEC[:, None] + np.array([[-3.0, 9.0]])
    pred_lh = psych_fit.predict_grid(low_high_grid)     # (10, 2, 11)
    est_gain_pct = 100.0 * float(np.mean([pred_lh[i, 1, i] - pred_lh[i, 0, i] for i in range(N_STIMULI)]))
    gain_err = abs(est_gain_pct - float(truth_entry["true_gain_pct"][0]))
    return tvd_pct, srt_mae, gain_err


# --------------------------------------------------------------------------
# Experiment 1: Single-SNR vs. Multi-SNR Stimulus Sets (at N=100, Block Mode)
# --------------------------------------------------------------------------

def run_experiment_1_designs(
    cohort: list[MultiSNRVirtualListener],
    designs: dict[str, StimulusSetDesign],
    truth_cache: dict[str, dict[str, np.ndarray]],
    n_reps: int = 30,
    base_seed: int = 100,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    design_names = ["single_ref", "rel_pm3", "rel_pm6", "rel_up_m3_p9", "rel_up_m4_p12", "abs_shared"]
    summary_rows: list[dict[str, Any]] = []
    archetype_rows: list[dict[str, Any]] = []

    broad_grid = REF_SNR_VEC[:, None] + EVAL_REL_SNRS_DB[None, :]
    broad_truths = {l.listener_id: l.pooled_matrix_for_design(broad_grid) for l in cohort}

    for dname in design_names:
        design = designs[dname]
        tracker = MultiSNRTracker(design=design, mode="block")

        n_L = len(cohort)
        pc_mat = np.zeros((n_L, n_reps), dtype=float)
        it_all_mat = np.zeros((n_L, n_reps), dtype=float)
        it_place_mat = np.zeros((n_L, n_reps), dtype=float)
        it_voice_mat = np.zeros((n_L, n_reps), dtype=float)
        it_manner_mat = np.zeros((n_L, n_reps), dtype=float)

        ceil_fracs: list[float] = []
        floor_fracs: list[float] = []
        active_zone_fracs: list[float] = []
        own_tvds: list[float] = []
        broad_tvds: list[float] = []
        retest_tvds: list[float] = []
        surf_tvds: list[float] = []
        srt_maes: list[float] = []
        gain_errs: list[float] = []

        arch_stats: dict[str, dict[str, list[float]]] = {}

        for li, base_listener in enumerate(cohort):
            own_gt = base_listener.pooled_matrix_for_design(design.snr_grid)
            broad_gt = broad_truths[base_listener.listener_id]
            t_entry = truth_cache[base_listener.listener_id]
            emp_fits: list[np.ndarray] = []

            arch_bucket = arch_stats.setdefault(
                base_listener.archetype,
                {"pc": [], "ceil": [], "floor": [], "active": [], "retest_tvd": [], "surf_tvd": [], "srt_mae": [], "gain_err": []},
            )

            for r in range(n_reps):
                seed = make_rep_seed(base_seed, base_listener.listener_id, r)
                vlistener = base_listener.clone_with_rng(np.random.default_rng(seed ^ 0x3C3C3C3C))
                out = tracker.run_session_at_points(vlistener, [100], seed=seed)[100]
                emp = out.pooled_empirical
                emp_fits.append(emp)

                diag = np.diag(emp[:, :N_STIMULI])
                pc = 100.0 * float(np.mean(diag))
                pc_mat[li, r] = pc

                it = relative_information_transmitted(emp)
                it_all_mat[li, r] = it["overall"]
                it_place_mat[li, r] = it["place"]
                it_voice_mat[li, r] = it["voicing"]
                it_manner_mat[li, r] = it["manner"]

                ceil_f = 100.0 * float(np.mean(diag >= 0.90))
                floor_f = 100.0 * float(np.mean(diag <= 0.15))
                act_f = 100.0 * float(np.mean((diag > 0.15) & (diag < 0.90)))
                ceil_fracs.append(ceil_f)
                floor_fracs.append(floor_f)
                active_zone_fracs.append(act_f)

                own_tvd = 50.0 * float(np.mean(np.sum(np.abs(emp - own_gt), axis=1)))
                broad_tvd = 50.0 * float(np.mean(np.sum(np.abs(emp - broad_gt), axis=1)))
                own_tvds.append(own_tvd)
                broad_tvds.append(broad_tvd)

                s_tvd, s_mae, g_err = evaluate_3d_surface_and_srt(out.psychometric_fit, t_entry)
                surf_tvds.append(s_tvd)
                srt_maes.append(s_mae)
                gain_errs.append(g_err)

                arch_bucket["pc"].append(pc)
                arch_bucket["ceil"].append(ceil_f)
                arch_bucket["floor"].append(floor_f)
                arch_bucket["active"].append(act_f)
                arch_bucket["surf_tvd"].append(s_tvd)
                arch_bucket["srt_mae"].append(s_mae)
                arch_bucket["gain_err"].append(g_err)

            for i in range(len(emp_fits)):
                for j in range(i + 1, len(emp_fits)):
                    rt = compute_pair_reliability(emp_fits[i], emp_fits[j])["mean_tvd_pct"]
                    retest_tvds.append(rt)
                    arch_bucket["retest_tvd"].append(rt)

        icc_pc, sdb_pc, sdw_pc, strata_pc = compute_icc_and_strata(pc_mat)
        icc_all, _, _, strata_all = compute_icc_and_strata(it_all_mat)
        icc_pl, _, _, strata_pl = compute_icc_and_strata(it_place_mat)
        icc_vo, _, _, _ = compute_icc_and_strata(it_voice_mat)
        icc_ma, _, _, _ = compute_icc_and_strata(it_manner_mat)

        summary_rows.append({
            "design": dname,
            "label": design.label,
            "active_confusion_zone_pct": float(np.mean(active_zone_fracs)),
            "ceiling_rows_pct": float(np.mean(ceil_fracs)),
            "floor_rows_pct": float(np.mean(floor_fracs)),
            "retest_tvd_pct": float(np.mean(retest_tvds)),
            "retest_pc_sd_pct": sdw_pc,
            "between_pc_sd_pct": sdb_pc,
            "icc_pc": icc_pc,
            "strata_pc": strata_pc,
            "icc_it_overall": icc_all,
            "strata_it_overall": strata_all,
            "icc_it_place": icc_pl,
            "strata_it_place": strata_pl,
            "icc_it_voicing": icc_vo,
            "icc_it_manner": icc_ma,
            "own_design_tvd_pct": float(np.mean(own_tvds)),
            "broad_ecological_tvd_pct": float(np.mean(broad_tvds)),
            "psych_surface_tvd_pct": float(np.mean(surf_tvds)),
            "srt50_mae_db": float(np.mean(srt_maes)),
            "snr_gain_mae_pct": float(np.mean(gain_errs)),
        })

        for arch, b in sorted(arch_stats.items()):
            archetype_rows.append({
                "design": dname,
                "archetype": arch,
                "mean_pc_pct": float(np.mean(b["pc"])),
                "active_zone_pct": float(np.mean(b["active"])),
                "ceiling_rows_pct": float(np.mean(b["ceil"])),
                "floor_rows_pct": float(np.mean(b["floor"])),
                "retest_tvd_pct": float(np.mean(b["retest_tvd"])),
                "psych_surface_tvd_pct": float(np.mean(b["surf_tvd"])),
                "srt50_mae_db": float(np.mean(b["srt_mae"])),
                "snr_gain_mae_pct": float(np.mean(b["gain_err"])),
            })

    return summary_rows, archetype_rows


# --------------------------------------------------------------------------
# Experiment 2: Adaptive vs. Stratified Block vs. Random across SNR Banks
# --------------------------------------------------------------------------

def run_experiment_2_tracking_modes(
    cohort: list[MultiSNRVirtualListener],
    designs: dict[str, StimulusSetDesign],
    truth_cache: dict[str, dict[str, np.ndarray]],
    trial_counts: list[int],
    n_reps: int = 25,
    base_seed: int = 200,
) -> list[dict[str, Any]]:
    target_banks = ["single_ref", "rel_up_m3_p9", "abs_shared", "rel_dense9"]
    modes = ["random", "block", "adaptive_consonant", "adaptive_2d_snr"]
    rows: list[dict[str, Any]] = []

    for dname in target_banks:
        design = designs[dname]
        for mode in modes:
            tracker = MultiSNRTracker(design=design, mode=mode)

            surf_by_nt: dict[int, list[float]] = {nt: [] for nt in trial_counts}
            srt_by_nt: dict[int, list[float]] = {nt: [] for nt in trial_counts}
            pool_dir_by_nt: dict[int, list[float]] = {nt: [] for nt in trial_counts}
            pool_psy_by_nt: dict[int, list[float]] = {nt: [] for nt in trial_counts}

            for base_listener in cohort:
                own_gt = base_listener.pooled_matrix_for_design(design.snr_grid)
                t_entry = truth_cache[base_listener.listener_id]
                for r in range(n_reps):
                    seed = make_rep_seed(base_seed, base_listener.listener_id, r)
                    vlistener = base_listener.clone_with_rng(np.random.default_rng(seed ^ 0x7A7A7A7A))
                    outs = tracker.run_session_at_points(vlistener, trial_counts, seed=seed)
                    for nt in trial_counts:
                        out = outs[nt]
                        s_tvd, s_mae, _ = evaluate_3d_surface_and_srt(out.psychometric_fit, t_entry)
                        surf_by_nt[nt].append(s_tvd)
                        srt_by_nt[nt].append(s_mae)

                        # Identical estimators across ALL 4 modes:
                        # 1) Uncoupled Dirichlet pooled estimate
                        pool_dir_tvd = 50.0 * float(np.mean(np.sum(np.abs(out.pooled_dirichlet - own_gt), axis=1)))
                        pool_dir_by_nt[nt].append(pool_dir_tvd)
                        # 2) Coupled psychometric model's implied pooled confusion matrix
                        pred_grid = out.psychometric_fit.predict_grid(design.snr_grid)  # (10, K, 11)
                        pred_pool = np.mean(pred_grid, axis=1)                          # (10, 11)
                        pool_psy_tvd = 50.0 * float(np.mean(np.sum(np.abs(pred_pool - own_gt), axis=1)))
                        pool_psy_by_nt[nt].append(pool_psy_tvd)

            for nt in trial_counts:
                rows.append({
                    "design": dname,
                    "mode": mode,
                    "n_trials": nt,
                    "surface_tvd_pct": float(np.mean(surf_by_nt[nt])),
                    "srt50_mae_db": float(np.mean(srt_by_nt[nt])),
                    "pooled_dirichlet_tvd_pct": float(np.mean(pool_dir_by_nt[nt])),
                    "pooled_psychfit_tvd_pct": float(np.mean(pool_psy_by_nt[nt])),
                })

    return rows


# --------------------------------------------------------------------------
# Visualization
# --------------------------------------------------------------------------

def save_study_figures(
    exp1_summary: list[dict[str, Any]],
    exp1_arch: list[dict[str, Any]],
    exp2_rows: list[dict[str, Any]],
    cohort: list[MultiSNRVirtualListener],
    designs: dict[str, StimulusSetDesign],
    out_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qvcv_mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    short_labels = {
        "single_ref":    "Single SNR\n(Current)",
        "rel_pm3":       "Rel ±3 dB\n(-3..+3)",
        "rel_pm6":       "Rel ±6 dB\n(-6..+6)",
        "rel_up_m3_p9":  "Rel Clinical\n(-3..+9 dB)",
        "rel_up_m4_p12": "Rel Wide Clin\n(-4..+12 dB)",
        "abs_shared":    "Abs Shared\n(-6..+18 dB)",
    }

    # --- Figure 1: Stimulus-Set Design Trade-offs (N=100, Block Mode) ---
    fig1, axes = plt.subplots(2, 2, figsize=(13.2, 8.8))
    d_order = [r["design"] for r in exp1_summary]
    x = np.arange(len(d_order))
    xlabels = [short_labels[d] for d in d_order]

    # Panel A: Stacked Ceiling / Active Confusion Zone / Floor %
    ax = axes[0, 0]
    act = np.array([r["active_confusion_zone_pct"] for r in exp1_summary])
    ceil = np.array([r["ceiling_rows_pct"] for r in exp1_summary])
    flr = np.array([r["floor_rows_pct"] for r in exp1_summary])
    ax.bar(x, flr, label="Floor (≤15% ID)", color="#ef4444", alpha=0.85)
    ax.bar(x, act, bottom=flr, label="Active Confusion Zone (15–90%)", color="#10b981", alpha=0.90)
    ax.bar(x, ceil, bottom=flr + act, label="Ceiling (≥90% ID)", color="#3b82f6", alpha=0.85)
    for k in range(len(x)):
        ax.text(x[k], flr[k] + 0.5 * act[k], f"{act[k]:.0f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=9.0)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.3)
    ax.set_ylabel("% of Listener × Consonant Rows")
    ax.set_ylim(0, 105)
    ax.set_title("A. Dynamic Range: Avoiding Ceiling & Floor Compression", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    # Panel B: Within-Listener Test-Retest SD vs Broad Ecological Confusion Accuracy
    ax = axes[0, 1]
    rt_sd = [r["retest_pc_sd_pct"] for r in exp1_summary]
    br_tvd = [r["broad_ecological_tvd_pct"] for r in exp1_summary]
    w = 0.36
    ax_r0 = ax.twinx()
    b0a = ax.bar(x - w / 2, rt_sd, width=w, color="#6366f1", label="Test-Retest Pc SD (%) [Stratification]")
    b0b = ax_r0.bar(x + w / 2, br_tvd, width=w, color="#0ea5e9", label="Broad [-9..+9 dB] Confusion TVD (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.3)
    ax.set_ylabel("Within-Listener Test-Retest Pc SD (%)", color="#6366f1")
    ax_r0.set_ylabel("Broad Ecological Confusion TVD (%)", color="#0ea5e9")
    ax.set_ylim(0, max(rt_sd) * 1.28)
    ax_r0.set_ylim(0, max(br_tvd) * 1.25)
    ax.set_title("B. Repeatability (Poisson-Binomial) & Broad Confusion Accuracy", fontsize=10.5, fontweight="bold")
    ax.legend([b0a, b0b], ["Test-Retest Pc SD (%)", "Broad [-9..+9 dB] Confusion TVD (%)"], fontsize=8.0, loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)

    # Panel C: Clinical Cohort Discriminability (Separable Strata)
    ax = axes[1, 0]
    st_pc = [r["strata_pc"] for r in exp1_summary]
    st_it = [r["strata_it_overall"] for r in exp1_summary]
    st_pl = [r["strata_it_place"] for r in exp1_summary]
    w3 = 0.25
    ax.bar(x - w3, st_pc, width=w3, label="Overall % Correct", color="#1e293b")
    ax.bar(x, st_it, width=w3, label="Overall Info Transmitted", color="#d97706")
    ax.bar(x + w3, st_pl, width=w3, label="Place Info Transmitted", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.3)
    ax.set_ylabel("Statistically Discriminable Strata (Higher = Better)")
    ax.set_title("C. Pooled Score Cohort Discriminability (Slope-Width Trade-off)", fontsize=10.5, fontweight="bold")
    ax.legend(fontsize=8.2)
    ax.grid(True, axis="y", alpha=0.25)

    # Panel D: SNR-Psychometric Surface & SNR-Recovery Gain Error
    ax = axes[1, 1]
    surf = [r["psych_surface_tvd_pct"] for r in exp1_summary]
    g_err = [r["snr_gain_mae_pct"] for r in exp1_summary]
    ax_r = ax.twinx()
    b1 = ax.bar(x - w / 2, surf, width=w, color="#059669", label="3D Confusion Surface TVD (%)")
    b2 = ax_r.bar(x + w / 2, g_err, width=w, color="#ea580c", label="SNR-Gain Slope MAE (% pts)")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=8.3)
    ax.set_ylabel("3D Confusion Surface TVD (%)", color="#059669")
    ax_r.set_ylabel("SNR-Gain Error (-3 to +9 dB, % pts)", color="#ea580c")
    ax.set_title("D. Psychometric Curve & Audibility-vs-Distortion Separation", fontsize=10.5, fontweight="bold")
    ax.legend([b1, b2], ["3D Confusion Surface TVD (%)", "SNR-Gain Slope MAE (% pts)"], fontsize=8.0, loc="upper right")
    ax.grid(True, axis="y", alpha=0.22)

    fig1.suptitle("qVCV Stimulus-Set Study: 5 Identical SNR Reps vs. 5-SNR Ranges (N = 100 Trials)", fontsize=13, fontweight="bold")
    fig1.tight_layout()
    fig1.savefig(out_dir / "fig1_snr_design_tradeoffs.png", dpi=170)
    plt.close(fig1)

    # --- Figure 2: Adaptive vs. Stratified Block vs. Random across Stimulus Banks ---
    banks_to_plot = [
        ("single_ref",    "Single SNR (Current 20-file set)"),
        ("rel_up_m3_p9",  "Relative Clinical 5-SNR (Ref -3..+9 dB)"),
        ("abs_shared",    "Shared Absolute 5-SNR (-6..+18 dB)"),
        ("rel_dense9",    "Wide Dense 9-SNR Bank (Ref ±12 dB)"),
    ]
    mode_styles = {
        "random":             ("Random (with replacement)",     "#94a3b8", ":",  "s"),
        "block":              ("Stratified Block (static)",     "#1e293b", "--", "o"),
        "adaptive_consonant": ("Adaptive Consonant Selection",  "#f59e0b", "-.", "^"),
        "adaptive_2d_snr":    ("Adaptive 2D (Consonant × SNR)", "#dc2626", "-",  "D"),
    }

    fig2, axes2 = plt.subplots(2, 2, figsize=(12.0, 8.2))
    for idx, (bname, btitle) in enumerate(banks_to_plot):
        ax = axes2[idx // 2, idx % 2]
        for mkey, (mlabel, col, ls, mk) in mode_styles.items():
            sub = sorted(
                [r for r in exp2_rows if r["design"] == bname and r["mode"] == mkey],
                key=lambda z: z["n_trials"],
            )
            if not sub:
                continue
            xs = [r["n_trials"] for r in sub]
            ykey = "pooled_dirichlet_tvd_pct" if bname == "single_ref" else "surface_tvd_pct"
            ys = [r[ykey] for r in sub]
            ax.plot(xs, ys, color=col, linestyle=ls, marker=mk, linewidth=2.0, markersize=5.5, label=mlabel)
        ax.set_title(btitle, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Number of Trials (N)")
        ax.set_ylabel("Pooled Row TVD (%)" if bname == "single_ref" else "3D Psychometric Surface TVD (%)")
        ax.grid(True, alpha=0.28)
        if idx == 0:
            ax.legend(fontsize=8.2)

    fig2.suptitle("Adaptive Tracking vs. Stratified Block vs. Random Across Stimulus Sets", fontsize=12.8, fontweight="bold")
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig2_adaptive_vs_block_multisnr.png", dpi=170)
    plt.close(fig2)

    # --- Figure 3: Why Clinical Upward-Shifted 5-SNR (-3..+9 dB) Captures Both NH & HI ---
    fig3, axes3 = plt.subplots(1, 2, figsize=(11.6, 4.7))
    snr_sweep = np.linspace(-15.0, 29.0, 150)
    example_archetypes = [
        ("NH_01",           "Normal Hearing",      "#2563eb"),
        ("HF_Mild_01",      "Mild Sloping HF",     "#f59e0b"),
        ("HF_Steep_01",     "Steep Sloping HF",    "#dc2626"),
        ("Elevated_SRT_01", "Poor SIN (+SRT)",     "#7c3aed"),
    ]
    by_id = {l.listener_id: l for l in cohort}

    for panel_idx, (cons_label, c_idx) in enumerate([("Consonant 'B' (/aba/, Ref = +18 dB)", 0), ("Consonant 'S' (/asa/, Ref = 0 dB)", 5)]):
        ax = axes3[panel_idx]
        ref_s = CONSONANT_SNR_DB[STIMULI_CONSONANTS[c_idx]]
        rel_pts = ref_s + np.array([-3.0, 0.0, 3.0, 6.0, 9.0])
        for rp in rel_pts:
            ax.axvline(rp, color="#10b981", linestyle=":", alpha=0.65, linewidth=1.4)
        ax.axvline(ref_s, color="#111827", linestyle="--", linewidth=1.8, label=f"Current Single SNR ({ref_s:+.0f} dB)")
        ax.axvspan(rel_pts[0], rel_pts[-1], color="#10b981", alpha=0.10, label="Rel Clinical 5-SNR (-3..+9 dB)")

        for lid, alabel, col in example_archetypes:
            l = by_id[lid]
            pcs = [100.0 * l.response_probs(c_idx, float(s))[c_idx] for s in snr_sweep]
            ax.plot(snr_sweep, pcs, color=col, linewidth=2.2, label=alabel)

        ax.axhspan(15, 90, color="#fef3c7", alpha=0.22)
        ax.set_title(cons_label, fontsize=10.8, fontweight="bold")
        ax.set_xlabel("Presentation SNR (dB)")
        ax.set_ylabel("Target Identification Probability (%)")
        ax.set_ylim(0, 102)
        ax.grid(True, alpha=0.25)
        if panel_idx == 0:
            ax.legend(fontsize=8.0, loc="lower right")

    fig3.suptitle("Psychometric Coverage: Single Fixed SNR vs. Clinical 5-SNR Range (Ref -3 to +9 dB)", fontsize=12.2, fontweight="bold")
    fig3.tight_layout()
    fig3.savefig(out_dir / "fig3_psychometric_coverage.png", dpi=170)
    plt.close(fig3)


# --------------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-SNR vs single-SNR stimulus design & tracking study.")
    parser.add_argument("--n_reps_exp1", type=int, default=35)
    parser.add_argument("--n_reps_exp2", type=int, default=25)
    parser.add_argument("--trial_counts", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path(__file__).resolve().with_name("multisnr_results"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    cohort = generate_multisnr_cohort(seed=20261115, per_archetype=4)
    save_multisnr_cohort_json(cohort, out_dir / "multisnr_virtual_listeners.json")
    designs = build_stimulus_designs()
    truth_cache = precompute_listener_truth_cache(cohort)

    print(f"Generated {len(cohort)} SNR-dependent virtual listeners across 6 clinical profiles.")
    print("\n--- Running Experiment 1: Single-SNR vs. 5-SNR Stimulus Sets (N=100, Stratified Block) ---")
    exp1_summary, exp1_arch = run_experiment_1_designs(
        cohort, designs, truth_cache=truth_cache, n_reps=args.n_reps_exp1, base_seed=101
    )
    write_dict_rows_csv(exp1_summary, out_dir / "exp1_stimulus_set_comparison.csv")
    write_dict_rows_csv(exp1_arch, out_dir / "exp1_by_archetype.csv")

    print(
        f"\n{'Design':<14s} | {'ActiveZone':>10s} | {'Ceil/Flr':>10s} | {'RetestPcSD':>10s} | "
        f"{'ICC(Pc)':>7s} | {'SurfTVD':>8s} | {'SRT MAE':>8s} | {'GainErr':>8s}"
    )
    print("-" * 100)
    for r in exp1_summary:
        cf = f"{r['ceiling_rows_pct']:4.1f}/{r['floor_rows_pct']:4.1f}%"
        print(
            f"{r['design']:<14s} | {r['active_confusion_zone_pct']:9.1f}% | {cf:>10s} | "
            f"{r['retest_pc_sd_pct']:9.2f}% | {r['icc_pc']:7.3f} | "
            f"{r['psych_surface_tvd_pct']:7.2f}% | {r['srt50_mae_db']:6.2f}dB | {r['snr_gain_mae_pct']:7.2f}%"
        )

    print("\n--- Running Experiment 2: Adaptive vs. Stratified Block vs. Random Selection ---")
    exp2_rows = run_experiment_2_tracking_modes(
        cohort, designs, truth_cache=truth_cache, trial_counts=args.trial_counts, n_reps=args.n_reps_exp2, base_seed=202
    )
    write_dict_rows_csv(exp2_rows, out_dir / "exp2_tracking_vs_block_multisnr.csv")

    print(f"\n{'Bank':<14s} | {'Selection Mode':<20s} | {'PoolDir@100':>11s} | {'PoolPsy@100':>11s} | {'SurfTVD@100':>11s} | {'SRT_MAE@100':>11s}")
    print("-" * 92)
    for bname in ["single_ref", "rel_up_m3_p9", "abs_shared", "rel_dense9"]:
        for mode in ["random", "block", "adaptive_consonant", "adaptive_2d_snr"]:
            r100 = next(r for r in exp2_rows if r["design"] == bname and r["mode"] == mode and r["n_trials"] == 100)
            print(
                f"{bname:<14s} | {mode:<20s} | {r100['pooled_dirichlet_tvd_pct']:10.2f}% | "
                f"{r100['pooled_psychfit_tvd_pct']:10.2f}% | {r100['surface_tvd_pct']:10.2f}% | "
                f"{r100['srt50_mae_db']:9.2f}dB"
            )
        print("-" * 92)

    with open(out_dir / "multisnr_study_summary.json", "w", encoding="utf-8") as f:
        json.dump({"exp1_designs": exp1_summary, "exp2_tracking_at_100": [r for r in exp2_rows if r["n_trials"] == 100]}, f, indent=2)

    save_study_figures(exp1_summary, exp1_arch, exp2_rows, cohort, designs, out_dir)
    print(f"\nCompleted Multi-SNR study in {time.perf_counter() - t0:.1f}s. Outputs saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
