#!/usr/bin/env python3
"""
exp_hellinger_aroc_snr.py

Evaluates single-SNR vs. multi-SNR stimulus sets and adaptive vs. block/random
selection on the EXACT criterion originally used to select qVCV's consonant SNRs:
  Maximizing A_ROC between Normal-Hearing (NH) and Hearing-Impaired (HI)
  listeners using consonant-confusion matrices and Hellinger distance.

Questions tested:
  1. When each consonant's reference SNR (B:18, V:18, N:15, D:4, G:3,
     K/S/SH/T/Z:0 dB) sits at the peak of the NH-vs-HI Hellinger distance curve,
     does replacing 5 identical SNR repetitions with a 5-SNR range help or hurt
     NH-vs-HI A_ROC (across N = 20, 40, 60, 80, 100 trials)?
  2. What happens to Hellinger A_ROC for SECONDARY clinical tasks where the
     optimal SNR differs — specifically:
       (a) Borderline screening: NH vs. Mild HI
       (b) Severity staging within HI: Mild HI vs. Moderate/Severe HI
       (c) Systematic SNR offset / presentation-level mismatch (+5 dB shift)
  3. Does adaptive selection (consonant or 2D Consonant x SNR) improve Hellinger
     A_ROC over static block or random selection?
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
    MultiSNRTracker,
    StimulusSetDesign,
)
from qvcv_multisnr_simulator import (
    MultiSNRArchetypeSpec,
    MultiSNRVirtualListener,
    _build_multisnr_listener,
)
from qvcv_simulator import (
    CONSONANT_SNR_DB,
    N_RESPONSES,
    N_STIMULI,
    PHONETIC_FEATURES,
    RESP_INDEX,
    STIM_INDEX,
    STIMULI_CONSONANTS,
    normalize_rows,
)
from qvcv_tracking_sim import write_dict_rows_csv

REF_SNR_VEC = np.array([CONSONANT_SNR_DB[c] for c in STIMULI_CONSONANTS], dtype=float)


# --------------------------------------------------------------------------
# Hellinger Distance & Exact Mann-Whitney A_ROC
# --------------------------------------------------------------------------

def hellinger_distance_matrix(p: np.ndarray, q: np.ndarray) -> float:
    """
    Mean row-wise Hellinger distance in [0, 1] between two (10, 11)
    row-stochastic confusion matrices:
      H(P, Q) = (1/10) sum_{i=1}^{10} (1/sqrt(2)) || sqrt(P_i) - sqrt(Q_i) ||_2
    """
    p_norm = normalize_rows(p)
    q_norm = normalize_rows(q)
    diff = np.sqrt(p_norm) - np.sqrt(q_norm)
    row_h = np.sqrt(0.5 * np.sum(diff * diff, axis=1))
    return float(np.mean(row_h))


def hellinger_row(p_vec: np.ndarray, q_vec: np.ndarray) -> float:
    diff = np.sqrt(np.maximum(p_vec, 0.0)) - np.sqrt(np.maximum(q_vec, 0.0))
    return float(np.sqrt(0.5 * np.sum(diff * diff)))


def compute_aroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    """Exact Mann-Whitney U / Wilcoxon A_ROC = P(S_pos > S_neg) + 0.5 P(S_pos == S_neg)."""
    neg = np.asarray(scores_neg, dtype=float)[:, None]
    pos = np.asarray(scores_pos, dtype=float)[None, :]
    greater = np.sum(pos > neg)
    ties = np.sum(pos == neg)
    return float((greater + 0.5 * ties) / (neg.shape[0] * pos.shape[1]))


# --------------------------------------------------------------------------
# Population Generator Calibrated so NH-vs-HI Hellinger Peaks at Ref SNR
# --------------------------------------------------------------------------

def build_nh_hi_populations(
    n_nh: int = 240,
    n_mild: int = 120,
    n_mod: int = 120,
    n_sev: int = 120,
    seed: int = 20261115,
) -> dict[str, list[MultiSNRVirtualListener]]:
    """
    Generate realistic NH and HI populations with natural inter-subject variance
    such that for every consonant i, the population Hellinger distance
    H(p_NH(i, SNR), p_HI(i, SNR)) peaks right at CONSONANT_SNR_DB[i].
    """
    rng = np.random.default_rng(seed)

    def _sample_group(
        count: int,
        prefix: str,
        archetype: str,
        lf_shift_mean: float,
        lf_shift_sd: float,
        hf_shift_mean: float,
        hf_shift_sd: float,
        slope_mean: float,
        ceil_lf: float,
        ceil_hf: float,
        w_place: float,
        unk_hi: float,
    ) -> list[MultiSNRVirtualListener]:
        out: list[MultiSNRVirtualListener] = []
        for idx in range(count):
            # Center NH threshold at Ref - 3.2 dB and average HI around Ref + 3.2 dB
            # so the symmetric Hellinger midpoint sits right at Ref_SNR(i)
            tau_50 = np.zeros(N_STIMULI, dtype=float)
            slope_db = np.zeros(N_STIMULI, dtype=float)
            max_self = np.zeros(N_STIMULI, dtype=float)

            subj_lf = rng.normal(lf_shift_mean, lf_shift_sd)
            subj_hf = rng.normal(hf_shift_mean, hf_shift_sd)

            for i, sc in enumerate(STIMULI_CONSONANTS):
                is_hf = bool(PHONETIC_FEATURES[sc]["hf_cue"])
                shift = subj_hf if is_hf else subj_lf
                tau_50[i] = CONSONANT_SNR_DB[sc] + shift + rng.normal(0.0, 1.0)
                slope_db[i] = np.clip(rng.normal(slope_mean, 0.05), 0.16, 0.55)
                max_self[i] = (ceil_hf if is_hf else ceil_lf) + rng.normal(0.0, 0.30)

            feat_w = np.maximum(
                0.0,
                np.array([1.00, 1.10, w_place, 0.80]) + rng.normal(0.0, 0.12, size=4),
            )
            unk_p = np.array([0.65 + rng.normal(0.0, 0.20), unk_hi + rng.normal(0.0, 0.25)])

            asym = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
            for (sc, rc), val in [
                ("G", "D", 0.55) if False else (("G", "D"), 0.55),
                (("D", "G"), 0.30),
                (("V", "B"), 0.60),
                (("B", "V"), 0.30),
                (("T", "K"), 0.50),
                (("K", "T"), 0.40),
                (("S", "SH"), 0.45),
                (("SH", "S"), 0.35),
                (("Z", "S"), 0.35),
                (("Z", "D"), 0.35),
            ]:
                asym[STIM_INDEX[sc], RESP_INDEX[rc]] += val
            asym[:, :N_STIMULI] += rng.normal(0.0, 0.20, size=(N_STIMULI, N_STIMULI))
            np.fill_diagonal(asym[:, :N_STIMULI], 0.0)

            tok_delta = rng.normal(0.0, 0.12, size=(N_STIMULI, N_RESPONSES))
            out.append(
                MultiSNRVirtualListener(
                    listener_id=f"{prefix}_{idx:03d}",
                    archetype=archetype,
                    description=archetype,
                    tau_50_db=tau_50,
                    slope_db=slope_db,
                    max_self_logit=max_self,
                    feat_weights=feat_w,
                    unk_params=unk_p,
                    asymmetry_matrix=asym,
                    token_logit_offsets=np.stack([tok_delta, -tok_delta], axis=0),
                )
            )
        return out

    # Note: NH tau_50 centered at Ref - 3.3 dB (giving ~72-80% ID at Ref SNR with natural spread),
    # Mild HI centered near Ref + 0.8 dB (LF) / +2.6 dB (HF) -> strong overlap with NH for realistic A_ROC ~0.85-0.93!
    # Moderate HI centered near Ref + 3.5 dB (LF) / +6.0 dB (HF)
    # Severe HI centered near Ref + 7.0 dB (LF) / +10.5 dB (HF)
    nh = _sample_group(
        n_nh, "NH", "NH",
        lf_shift_mean=-3.3, lf_shift_sd=2.2,
        hf_shift_mean=-3.3, hf_shift_sd=2.4,
        slope_mean=0.35, ceil_lf=4.0, ceil_hf=3.9, w_place=0.65, unk_hi=-3.3,
    )
    mild = _sample_group(
        n_mild, "HI_Mild", "HI_Mild",
        lf_shift_mean=0.6, lf_shift_sd=2.3,
        hf_shift_mean=2.4, hf_shift_sd=2.5,
        slope_mean=0.30, ceil_lf=3.5, ceil_hf=2.6, w_place=0.35, unk_hi=-2.4,
    )
    mod = _sample_group(
        n_mod, "HI_Mod", "HI_Mod",
        lf_shift_mean=3.5, lf_shift_sd=2.5,
        hf_shift_mean=6.2, hf_shift_sd=2.7,
        slope_mean=0.27, ceil_lf=3.0, ceil_hf=1.9, w_place=0.15, unk_hi=-1.7,
    )
    sev = _sample_group(
        n_sev, "HI_Sev", "HI_Sev",
        lf_shift_mean=7.2, lf_shift_sd=2.7,
        hf_shift_mean=10.5, hf_shift_sd=3.0,
        slope_mean=0.24, ceil_lf=2.3, ceil_hf=1.3, w_place=0.05, unk_hi=-1.0,
    )
    return {"NH": nh, "HI_Mild": mild, "HI_Mod": mod, "HI_Sev": sev}


def build_aroc_designs(global_snr_bias_db: float = 0.0) -> list[StimulusSetDesign]:
    """Candidate 5-slot stimulus designs (with optional global calibration offset)."""
    ref = (REF_SNR_VEC + global_snr_bias_db)[:, None]
    return [
        StimulusSetDesign("single_ref",    "Single SNR (5x @ Optimal Ref)",       ref + np.array([[0.0, 0.0, 0.0, 0.0, 0.0]])),
        StimulusSetDesign("rel_pm2",       "Rel Narrow (Ref -2..+2 dB, 1 dB)",    ref + np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])),
        StimulusSetDesign("rel_pm4",       "Rel Mod-Narrow (Ref -4..+4 dB, 2 dB)",ref + np.array([[-4.0, -2.0, 0.0, 2.0, 4.0]])),
        StimulusSetDesign("rel_pm6",       "Rel Medium (Ref -6..+6 dB, 3 dB)",    ref + np.array([[-6.0, -3.0, 0.0, 3.0, 6.0]])),
        StimulusSetDesign("rel_pm9",       "Rel Wide (Ref -9..+9 dB, 4.5 dB)",    ref + np.array([[-9.0, -4.5, 0.0, 4.5, 9.0]])),
        StimulusSetDesign("rel_up_m3_p9",  "Rel Upward (Ref -3, 0, +3, +6, +9)",  ref + np.array([[-3.0, 0.0, 3.0, 6.0, 9.0]])),
        StimulusSetDesign("abs_shared",    "Abs Shared (-6, 0, +6, +12, +18 dB)", np.repeat(np.array([[-6.0, 0.0, 6.0, 12.0, 18.0]]) + global_snr_bias_db, N_STIMULI, axis=0)),
    ]


# --------------------------------------------------------------------------
# Fast Vectorized Block & Mode Simulation for Large-Cohort A_ROC
# --------------------------------------------------------------------------

def compute_group_template(
    listeners: list[MultiSNRVirtualListener],
    snr_grid: np.ndarray,
) -> np.ndarray:
    """Exact population-mean (10, 11) confusion matrix for a group under a stimulus design."""
    acc = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
    for l in listeners:
        acc += l.pooled_matrix_for_design(snr_grid)
    return normalize_rows(acc / len(listeners))


def evaluate_design_arocs(
    pops: dict[str, list[MultiSNRVirtualListener]],
    templates_pop: dict[str, list[MultiSNRVirtualListener]],
    design: StimulusSetDesign,
    trial_counts: list[int],
    mode: str = "block",
    seed: int = 500,
) -> list[dict[str, Any]]:
    """
    Simulate 1 session per listener across the independent test populations
    and compute Hellinger-based A_ROC for:
      - NH vs. All HI        (primary screening criterion)
      - NH vs. Mild HI       (hardest borderline screening criterion)
      - Mild HI vs. Mod+Sev  (severity staging within HI)
    Uses two standard Hellinger decision statistics:
      - d_NH(P_hat)    = H(P_hat, Template_NH)
      - delta_H(P_hat) = H(P_hat, Template_NH) - H(P_hat, Template_TargetPos)
    """
    tmpl_nh = compute_group_template(templates_pop["NH"], design.snr_grid)
    tmpl_hi_all = compute_group_template(
        templates_pop["HI_Mild"] + templates_pop["HI_Mod"] + templates_pop["HI_Sev"],
        design.snr_grid,
    )
    tmpl_mild = compute_group_template(templates_pop["HI_Mild"], design.snr_grid)
    tmpl_modsev = compute_group_template(
        templates_pop["HI_Mod"] + templates_pop["HI_Sev"],
        design.snr_grid,
    )

    tracker = MultiSNRTracker(design=design, mode=mode, prior_conc=1.0)

    # Collect estimated matrices per group & trial count
    est_mats: dict[tuple[str, int], list[np.ndarray]] = {
        (gname, nt): [] for gname in pops for nt in trial_counts
    }
    base_rng = np.random.default_rng(seed)

    for gname, glist in pops.items():
        for l in glist:
            s_val = int(base_rng.integers(1, 2**30))
            vl = l.clone_with_rng(np.random.default_rng(s_val ^ 0x13579BDF))
            outs = tracker.run_session_at_points(vl, trial_counts, seed=s_val, fit_psychometric=False)
            for nt in trial_counts:
                # Use mild Laplace/Dirichlet smoothing (prior_conc=1.0) so zero counts
                # at small N don't distort sqrt(p) in Hellinger distance
                est_mats[(gname, nt)].append(outs[nt].pooled_dirichlet)

    rows: list[dict[str, Any]] = []
    for nt in trial_counts:
        nh_mats = est_mats[("NH", nt)]
        mild_mats = est_mats[("HI_Mild", nt)]
        mod_mats = est_mats[("HI_Mod", nt)]
        sev_mats = est_mats[("HI_Sev", nt)]
        all_hi_mats = mild_mats + mod_mats + sev_mats
        modsev_mats = mod_mats + sev_mats

        # 1. Pure Hellinger distance from NH template: H(P_hat, P_NH)
        d_nh_for_nh = np.array([hellinger_distance_matrix(m, tmpl_nh) for m in nh_mats])
        d_nh_for_hi = np.array([hellinger_distance_matrix(m, tmpl_nh) for m in all_hi_mats])
        d_nh_for_mild = np.array([hellinger_distance_matrix(m, tmpl_nh) for m in mild_mats])
        d_nh_for_modsev = np.array([hellinger_distance_matrix(m, tmpl_nh) for m in modsev_mats])

        aroc_nh_hi_oneclass = compute_aroc(d_nh_for_nh, d_nh_for_hi)
        aroc_nh_mild_oneclass = compute_aroc(d_nh_for_nh, d_nh_for_mild)

        # 2. Differential Hellinger classifier: H(P_hat, P_0) - H(P_hat, P_1)
        diff_nh = np.array([
            hellinger_distance_matrix(m, tmpl_nh) - hellinger_distance_matrix(m, tmpl_hi_all)
            for m in nh_mats
        ])
        diff_hi = np.array([
            hellinger_distance_matrix(m, tmpl_nh) - hellinger_distance_matrix(m, tmpl_hi_all)
            for m in all_hi_mats
        ])
        diff_nh_m = np.array([
            hellinger_distance_matrix(m, tmpl_nh) - hellinger_distance_matrix(m, tmpl_mild)
            for m in nh_mats
        ])
        diff_mild_m = np.array([
            hellinger_distance_matrix(m, tmpl_nh) - hellinger_distance_matrix(m, tmpl_mild)
            for m in mild_mats
        ])
        diff_mild_stage = np.array([
            hellinger_distance_matrix(m, tmpl_mild) - hellinger_distance_matrix(m, tmpl_modsev)
            for m in mild_mats
        ])
        diff_modsev_stage = np.array([
            hellinger_distance_matrix(m, tmpl_mild) - hellinger_distance_matrix(m, tmpl_modsev)
            for m in modsev_mats
        ])

        aroc_nh_hi_diff = compute_aroc(diff_nh, diff_hi)
        aroc_nh_mild_diff = compute_aroc(diff_nh_m, diff_mild_m)
        aroc_mild_modsev_diff = compute_aroc(diff_mild_stage, diff_modsev_stage)
        aroc_mild_modsev_dnh = compute_aroc(d_nh_for_mild, d_nh_for_modsev)

        # Also record population theoretical Hellinger separation H(tmpl_NH, tmpl_HI)
        pop_hellinger_nh_hi = hellinger_distance_matrix(tmpl_nh, tmpl_hi_all)
        pop_hellinger_mild_modsev = hellinger_distance_matrix(tmpl_mild, tmpl_modsev)

        rows.append({
            "design": design.name,
            "label": design.label,
            "mode": mode,
            "n_trials": nt,
            "pop_hellinger_nh_hi": pop_hellinger_nh_hi,
            "pop_hellinger_mild_modsev": pop_hellinger_mild_modsev,
            "aroc_nh_vs_all_hi_dNH": aroc_nh_hi_oneclass,
            "aroc_nh_vs_all_hi_diffH": aroc_nh_hi_diff,
            "aroc_nh_vs_mild_hi_dNH": aroc_nh_mild_oneclass,
            "aroc_nh_vs_mild_hi_diffH": aroc_nh_mild_diff,
            "aroc_mild_vs_modsev_dNH": aroc_mild_modsev_dnh,
            "aroc_mild_vs_modsev_diffH": aroc_mild_modsev_diff,
        })
    return rows


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def save_hellinger_figures(
    tmpl_pops: dict[str, list[MultiSNRVirtualListener]],
    design_rows: list[dict[str, Any]],
    shifted_rows: list[dict[str, Any]],
    mode_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qvcv_mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 9.0))

    # Panel A: Population Hellinger Distance H(p_NH, p_HI) vs SNR per Consonant Group
    ax = axes[0, 0]
    snr_axis = np.linspace(-15.0, 30.0, 91)
    nh_list = tmpl_pops["NH"]
    hi_list = tmpl_pops["HI_Mild"] + tmpl_pops["HI_Mod"] + tmpl_pops["HI_Sev"]

    rep_consonants = [
        ("B (+18 dB Ref)", 0, "#2563eb"),
        ("N (+15 dB Ref)", 4, "#0ea5e9"),
        ("D (+4 dB Ref)",  1, "#10b981"),
        ("G (+3 dB Ref)",  2, "#f59e0b"),
        ("S (0 dB Ref)",   5, "#dc2626"),
        ("T (0 dB Ref)",   7, "#9333ea"),
    ]
    for clabel, c_idx, col in rep_consonants:
        h_vals = []
        for s in snr_axis:
            p_nh = np.mean([l.response_probs(c_idx, float(s)) for l in nh_list], axis=0)
            p_hi = np.mean([l.response_probs(c_idx, float(s)) for l in hi_list], axis=0)
            h_vals.append(hellinger_row(p_nh, p_hi))
        ax.plot(snr_axis, h_vals, color=col, linewidth=2.1, label=clabel)
        ref_s = CONSONANT_SNR_DB[STIMULI_CONSONANTS[c_idx]]
        peak_idx = int(np.argmin(np.abs(snr_axis - ref_s)))
        ax.scatter([ref_s], [h_vals[peak_idx]], color=col, s=42, zorder=5, edgecolor="black", linewidth=0.8)

    ax.set_title("A. Population NH-vs-HI Hellinger Distance vs. SNR (Dots = qVCV Ref SNR)", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Presentation SNR (dB)")
    ax.set_ylabel("Hellinger Distance H(p_NH, p_HI)")
    ax.legend(fontsize=8.0, loc="upper right")
    ax.grid(True, alpha=0.26)

    # Panel B: NH vs. Mild HI (Borderline Screening A_ROC) across Trial Counts
    ax = axes[0, 1]
    design_styles = {
        "single_ref":   ("Single SNR (5x @ Optimal Ref)", "#111827", "--", "o"),
        "rel_pm2":      ("Rel ±2 dB (1 dB step)",         "#2563eb", "-",  "^"),
        "rel_pm4":      ("Rel ±4 dB (2 dB step)",         "#0ea5e9", "-",  "s"),
        "rel_pm6":      ("Rel ±6 dB (3 dB step)",         "#10b981", "-",  "D"),
        "rel_pm9":      ("Rel ±9 dB (4.5 dB step)",       "#f59e0b", "-.", "v"),
        "rel_up_m3_p9": ("Rel Upward (-3..+9 dB)",        "#dc2626", "-",  "P"),
        "abs_shared":   ("Abs Shared (-6..+18 dB)",       "#9ca3af", ":",  "x"),
    }
    for dkey, (dlabel, col, ls, mk) in design_styles.items():
        sub = sorted([r for r in design_rows if r["design"] == dkey], key=lambda z: z["n_trials"])
        ax.plot(
            [r["n_trials"] for r in sub],
            [r["aroc_nh_vs_mild_hi_diffH"] for r in sub],
            color=col, linestyle=ls, marker=mk, linewidth=2.0, markersize=5.2, label=dlabel,
        )
    ax.set_title("B. Primary Screening A_ROC (NH vs. Mild HI) via Hellinger Distance", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Number of Trials (N)")
    ax.set_ylabel("A_ROC (NH vs. Mild HI)")
    ax.legend(fontsize=7.8, loc="lower right")
    ax.grid(True, alpha=0.26)

    # Panel C: Trade-off at N=100 across 3 Clinical Discrimination Tasks!
    ax = axes[1, 0]
    d_order = ["single_ref", "rel_pm2", "rel_pm4", "rel_pm6", "rel_up_m3_p9", "rel_pm9", "abs_shared"]
    short_n = ["Single\n(Ref)", "Rel\n±2 dB", "Rel\n±4 dB", "Rel\n±6 dB", "Rel Up\n-3..+9", "Rel\n±9 dB", "Abs\nShared"]
    x = np.arange(len(d_order))
    w = 0.26

    r100_map = {r["design"]: r for r in design_rows if r["n_trials"] == 100}
    sh100_map = {r["design"]: r for r in shifted_rows if r["n_trials"] == 100}

    y_nh_mild = [r100_map[d]["aroc_nh_vs_mild_hi_diffH"] for d in d_order]
    y_stage   = [r100_map[d]["aroc_mild_vs_modsev_diffH"] for d in d_order]
    y_shift   = [sh100_map[d]["aroc_nh_vs_mild_hi_diffH"] for d in d_order]

    ax.bar(x - w, y_nh_mild, width=w, color="#1e293b", label="NH vs. Mild HI (Aligned Ref)")
    ax.bar(x,     y_shift,   width=w, color="#ea580c", label="NH vs. Mild HI (+5 dB Level Mismatch)")
    ax.bar(x + w, y_stage,   width=w, color="#059669", label="HI Severity Staging (Mild vs. Mod/Sev)")
    ax.set_xticks(x)
    ax.set_xticklabels(short_n, fontsize=8.5)
    ax.set_ylim(0.70, 0.98)
    ax.set_ylabel("Hellinger A_ROC at N = 100")
    ax.set_title("C. Screening vs. Calibration-Mismatch vs. HI Staging (N = 100)", fontsize=10.3, fontweight="bold")
    ax.legend(fontsize=7.9, loc="lower left")
    ax.grid(True, axis="y", alpha=0.26)

    # Panel D: Adaptive vs. Block vs. Random on Hellinger A_ROC (NH vs. Mild HI)
    ax = axes[1, 1]
    mode_plot_styles = {
        ("single_ref", "random"):             ("Single SNR — Random",           "#94a3b8", ":",  "s"),
        ("single_ref", "block"):              ("Single SNR — Block",            "#111827", "--", "o"),
        ("single_ref", "adaptive_consonant"): ("Single SNR — Adaptive Cons.",   "#f59e0b", "-.", "^"),
        ("rel_pm4", "random"):                ("Rel ±4 dB — Random",            "#93c5fd", ":",  "s"),
        ("rel_pm4", "block"):                 ("Rel ±4 dB — Stratified Block",  "#2563eb", "--", "o"),
        ("rel_pm4", "adaptive_2d_snr"):       ("Rel ±4 dB — Adaptive 2D SNR",   "#dc2626", "-",  "D"),
    }
    for (dk, mk_mode), (mlab, col, ls, mk) in mode_plot_styles.items():
        sub = sorted(
            [r for r in mode_rows if r["design"] == dk and r["mode"] == mk_mode],
            key=lambda z: z["n_trials"],
        )
        ax.plot(
            [r["n_trials"] for r in sub],
            [r["aroc_nh_vs_mild_hi_diffH"] for r in sub],
            color=col, linestyle=ls, marker=mk, linewidth=2.0, markersize=5.2, label=mlab,
        )
    ax.set_title("D. Adaptive Tracking vs. Block vs. Random on Hellinger A_ROC", fontsize=10.3, fontweight="bold")
    ax.set_xlabel("Number of Trials (N)")
    ax.set_ylabel("A_ROC (NH vs. Mild HI)")
    ax.legend(fontsize=7.8, loc="lower right")
    ax.grid(True, alpha=0.26)

    fig.suptitle("qVCV Hellinger-Distance & A_ROC Analysis: Single Optimal SNR vs. 5-SNR Stimulus Sets", fontsize=13.0, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "fig4_hellinger_aroc_analysis.png", dpi=175)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Hellinger distance & NH-vs-HI A_ROC across SNR designs.")
    parser.add_argument("--trial_counts", type=int, nargs="+", default=[20, 40, 60, 80, 100])
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path(__file__).resolve().with_name("multisnr_results"),
    )
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    # Independent template training cohort & test evaluation cohort (zero data leakage)
    tmpl_pops = build_nh_hi_populations(n_nh=200, n_mild=100, n_mod=100, n_sev=100, seed=11111)
    test_pops = build_nh_hi_populations(n_nh=300, n_mild=150, n_mod=150, n_sev=150, seed=22222)

    print("1. Evaluating 7 SNR designs under Stratified Block across N =", args.trial_counts)
    designs_aligned = build_aroc_designs(global_snr_bias_db=0.0)
    design_rows: list[dict[str, Any]] = []
    for d in designs_aligned:
        design_rows.extend(
            evaluate_design_arocs(test_pops, tmpl_pops, d, args.trial_counts, mode="block", seed=701)
        )
    write_dict_rows_csv(design_rows, out_dir / "hellinger_aroc_by_design.csv")

    print("\n=== ALIGNED REFERENCE SNR: HELLINGER SEPARATION & A_ROC AT N = 100 ===")
    print(
        f"{'Design':<14s} | {'Pop H(NH,HI)':>12s} | {'A_ROC(NH vs All HI)':>19s} | "
        f"{'A_ROC(NH vs Mild)':>17s} | {'A_ROC(Mild vs ModSev)':>21s}"
    )
    print("-" * 96)
    for r in [x for x in design_rows if x["n_trials"] == 100]:
        print(
            f"{r['design']:<14s} | {r['pop_hellinger_nh_hi']:12.4f} | "
            f"{r['aroc_nh_vs_all_hi_diffH']:19.4f} | "
            f"{r['aroc_nh_vs_mild_hi_diffH']:17.4f} | "
            f"{r['aroc_mild_vs_modsev_diffH']:21.4f}"
        )

    print("\n2. Evaluating Robustness under +5 dB Calibration / Presentation-SNR Mismatch...")
    designs_shifted = build_aroc_designs(global_snr_bias_db=5.0)
    # Note: templates remain the nominal clinical reference templates (designs_aligned),
    # while actual acoustic presentation is shifted by +5 dB!
    shifted_rows: list[dict[str, Any]] = []
    for d_nom, d_shift in zip(designs_aligned, designs_shifted):
        # Evaluate test listeners presented with d_shift against nominal templates from d_nom
        tmpl_nh = compute_group_template(tmpl_pops["NH"], d_nom.snr_grid)
        tmpl_mild = compute_group_template(tmpl_pops["HI_Mild"], d_nom.snr_grid)
        tracker = MultiSNRTracker(design=d_shift, mode="block", prior_conc=1.0)

        rng = np.random.default_rng(808)
        nh_scores, mild_scores = [], []
        for g_scores, glist in [(nh_scores, test_pops["NH"]), (mild_scores, test_pops["HI_Mild"])]:
            for l in glist:
                sv = int(rng.integers(1, 2**30))
                vl = l.clone_with_rng(np.random.default_rng(sv ^ 0x2468ACE0))
                m100 = tracker.run_session_at_points(vl, [100], seed=sv, fit_psychometric=False)[100].pooled_dirichlet
                g_scores.append(hellinger_distance_matrix(m100, tmpl_nh) - hellinger_distance_matrix(m100, tmpl_mild))
        shifted_rows.append({
            "design": d_nom.name,
            "n_trials": 100,
            "aroc_nh_vs_mild_hi_diffH": compute_aroc(np.array(nh_scores), np.array(mild_scores)),
        })
    write_dict_rows_csv(shifted_rows, out_dir / "hellinger_aroc_shifted5dB.csv")

    print("\n=== +5 dB CALIBRATION / LEVEL MISMATCH: NH vs. MILD HI A_ROC AT N = 100 ===")
    for r_al, r_sh in zip([x for x in design_rows if x["n_trials"] == 100], shifted_rows):
        print(
            f"  {r_al['design']:<14s}: Aligned={r_al['aroc_nh_vs_mild_hi_diffH']:.4f}  ->  "
            f"+5dB Mismatch={r_sh['aroc_nh_vs_mild_hi_diffH']:.4f}  "
            f"(Delta={r_sh['aroc_nh_vs_mild_hi_diffH'] - r_al['aroc_nh_vs_mild_hi_diffH']:+.4f})"
        )

    print("\n3. Verifying Adaptive vs. Block vs. Random on Hellinger A_ROC...")
    mode_rows: list[dict[str, Any]] = []
    d_map = {d.name: d for d in designs_aligned}
    for dname, modes in [
        ("single_ref", ["random", "block", "adaptive_consonant"]),
        ("rel_pm4",    ["random", "block", "adaptive_2d_snr"]),
    ]:
        for m in modes:
            mode_rows.extend(
                evaluate_design_arocs(test_pops, tmpl_pops, d_map[dname], args.trial_counts, mode=m, seed=909)
            )
    write_dict_rows_csv(mode_rows, out_dir / "hellinger_aroc_tracking_modes.csv")

    print("\n=== ADAPTIVE vs. BLOCK vs. RANDOM ON HELLINGER A_ROC (N = 100) ===")
    for r in [x for x in mode_rows if x["n_trials"] == 100]:
        print(
            f"  {r['design']:<12s} | {r['mode']:<20s} | "
            f"A_ROC(NH vs All HI)={r['aroc_nh_vs_all_hi_diffH']:.4f} | "
            f"A_ROC(NH vs Mild HI)={r['aroc_nh_vs_mild_hi_diffH']:.4f}"
        )

    save_hellinger_figures(tmpl_pops, design_rows, shifted_rows, mode_rows, out_dir)
    with open(out_dir / "hellinger_aroc_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "aligned_N100": [x for x in design_rows if x["n_trials"] == 100],
                "shifted5dB_N100": shifted_rows,
                "modes_N100": [x for x in mode_rows if x["n_trials"] == 100],
            },
            f,
            indent=2,
        )
    print(f"\nCompleted Hellinger A_ROC study in {time.perf_counter() - t0:.1f}s. Saved fig4_hellinger_aroc_analysis.png")


if __name__ == "__main__":
    main()
