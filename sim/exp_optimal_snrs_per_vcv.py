#!/usr/bin/env python3
"""
exp_optimal_snrs_per_vcv.py

Identifies and compares for each of the 10 qVCV consonants (ABA, ADA, AGA, AKA,
ANA, ASA, ASHA, ATA, AVA, AZA) on the 98 human-calibrated virtual listeners
(32 NH, 66 HL: 23 Slight, 26 Mild, 17 Moderate):
  1. Optimal "Single SNR" (5 repetitions x 2 tokens = 10 trials at 1 SNR),
  2. Optimal "Set of 5 SNRs" (1 repetition x 2 tokens at each of 5 distinct SNRs:
     both minimum spacing >= 1 dB and practical spread >= 2 dB / >= 8 dB span),
  3. Exact head-to-head comparison of NH-vs-HL separation (A_ROC) between the
     best "Set of 5 SNRs" and the best "Single SNR" — both per VCV (10 trials)
     and across the full 100-trial qVCV test, evaluated on an independent
     held-out Monte Carlo ensemble (M_eval = 1,500 sessions) with paired
     Common Random Numbers and subject-bootstrap 95% confidence intervals,
     under both:
       (a) Standard pooled confusion-row LOOCV Hellinger distance, and
       (b) SNR-aware slot-specific LOOCV Hellinger & Neyman-Pearson LLR scoring.
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

from qvcv_multisnr_simulator import (
    HumanCalibratedMultiSNRListener,
    load_human_multisnr_cohort,
)
from qvcv_simulator import (
    CONSONANT_SNR_DB,
    N_RESPONSES,
    N_STIMULI,
    STIMULI_CONSONANTS,
)
from qvcv_tracking_sim import write_dict_rows_csv

SNR_GRID_DB = np.arange(-15, 31, 1, dtype=float)  # 46 integer SNRs: -15..+30 dB
N_SNRS = len(SNR_GRID_DB)
SNR_TO_IDX = {int(s): idx for idx, s in enumerate(SNR_GRID_DB)}


def fast_loocv_hellinger_aroc_batch(
    counts_m_n_k_d: np.ndarray,
    neg_mask: np.ndarray,
    pos_mask: np.ndarray,
) -> np.ndarray:
    """
    Exact sample-size-unbiased Leave-One-NH-Out Hellinger Mann-Whitney A_ROC
    for each Monte Carlo draw m in 0..M-1, returning shape (M,).
    Accepts shape (M, N, D) or (M, N, K, D).
    """
    arr = np.asarray(counts_m_n_k_d, dtype=np.float32)
    if arr.ndim == 3:
        arr = arr[:, :, None, :]
    obs = arr / np.maximum(arr.sum(axis=-1, keepdims=True), np.float32(1e-12))
    sqrt_obs = np.sqrt(obs)

    neg_idx = np.where(neg_mask)[0]
    pos_idx = np.where(pos_mask)[0]
    n_neg = len(neg_idx)
    n_pos = len(pos_idx)

    neg_obs = obs[:, neg_idx, :, :]
    neg_sum = neg_obs.sum(axis=1)
    sqrt_neg = sqrt_obs[:, neg_idx, :, :]
    sqrt_pos = sqrt_obs[:, pos_idx, :, :]

    u_total = np.zeros(arr.shape[0], dtype=np.float64)
    inv_nm1 = np.float32(1.0 / float(n_neg - 1))
    half = np.float32(0.5)

    for ii in range(n_neg):
        t_minus_i = np.maximum((neg_sum - neg_obs[:, ii, :, :]) * inv_nm1, np.float32(0.0))
        st = np.sqrt(t_minus_i)
        h_i = np.mean(np.sqrt(half * np.sum((sqrt_neg[:, ii, :, :] - st) ** 2, axis=-1)), axis=-1)
        h_pos = np.mean(np.sqrt(half * np.sum((sqrt_pos - st[:, None, :, :]) ** 2, axis=-1)), axis=-1)
        diff = h_pos - h_i[:, None]
        u_total += np.sum(diff > 1e-6, axis=1) + 0.5 * np.sum(np.abs(diff) <= 1e-6, axis=1)

    return u_total / float(n_neg * n_pos)


def fast_loocv_slot_llr_aroc_batch(
    slot_counts_m_n_k_d: np.ndarray,
    neg_mask: np.ndarray,
    pos_mask: np.ndarray,
    eps: float = 0.005,
) -> np.ndarray:
    """
    SNR-aware Leave-One-Out Log-Likelihood Ratio (Neyman-Pearson Bayes factor)
    Mann-Whitney A_ROC across K SNR slots: weights errors at high SNR more
    strongly than errors at low SNR using empirical out-of-sample slot profiles.
    slot_counts_m_n_k_d: (M, N, K_slots, D=11).
    """
    arr = np.asarray(slot_counts_m_n_k_d, dtype=np.float32)
    props = arr / np.maximum(arr.sum(axis=-1, keepdims=True), np.float32(1e-12))

    neg_idx = np.where(neg_mask)[0]
    pos_idx = np.where(pos_mask)[0]
    n_neg = len(neg_idx)
    n_pos = len(pos_idx)

    neg_props = props[:, neg_idx, :, :]
    pos_props = props[:, pos_idx, :, :]
    neg_sum = neg_props.sum(axis=1)
    pos_sum = pos_props.sum(axis=1)

    cnt_neg = arr[:, neg_idx, :, :]
    cnt_pos = arr[:, pos_idx, :, :]

    u_total = np.zeros(arr.shape[0], dtype=np.float64)
    # Precompute leave-one-out log(p_hl_{-jj}) for all jj in 0..n_pos-1: (M, n_pos, K, 11)
    p_hl_loo = (pos_sum[:, None, :, :] - pos_props + eps) / (float(n_pos - 1) + 11.0 * eps)
    log_hl_loo = np.log(p_hl_loo)  # (M, n_pos, K, 11)
    s_j_hl_part = np.sum(cnt_pos * log_hl_loo, axis=(2, 3))  # (M, n_pos)

    for ii in range(n_neg):
        p_nh = (neg_sum - neg_props[:, ii, :, :] + eps) / (float(n_neg - 1) + 11.0 * eps)  # (M, K, 11)
        log_nh = np.log(p_nh)  # (M, K, 11)
        # For pair (ii, jj): w_{ii,jj} = log_hl_loo[:, jj] - log_nh
        # s_i(ii, jj) = sum_{k,d} cnt_neg[:, ii, k, d] * (log_hl_loo[:, jj, k, d] - log_nh[:, k, d])
        c_i = cnt_neg[:, ii, :, :]  # (M, K, 11)
        s_i_nh_part = np.sum(c_i * log_nh, axis=(1, 2))[:, None]  # (M, 1)
        s_i = np.einsum("mkd,mjkd->mj", c_i, log_hl_loo) - s_i_nh_part  # (M, n_pos)
        s_j = s_j_hl_part - np.einsum("mjkd,mkd->mj", cnt_pos, log_nh)  # (M, n_pos)
        diff = s_j - s_i
        u_total += np.sum(diff > 1e-6, axis=1) + 0.5 * np.sum(np.abs(diff) <= 1e-6, axis=1)

    return u_total / float(n_neg * n_pos)


def sample_crn_two_token_counts(
    cohort: list[HumanCalibratedMultiSNRListener],
    stim_idx: int,
    n_mc: int,
    seed: int,
    include_session_shift: bool = True,
    z_sess_m_n: np.ndarray | None = None,
) -> np.ndarray:
    """
    Sample paired Common-Random-Number 2-token response counts of shape
    (N_SNRS=46, 5 slots, n_mc, 98, 11) in uint8 for consonant ``stim_idx``.
    Using the exact same uniform draws U[n_mc, 98, 2, 5] across all candidate
    SNRs guarantees variance-reduced paired comparisons between any single SNR
    and any 5-SNR tuple.
    """
    n_subj = len(cohort)
    rng = np.random.default_rng(seed)
    if z_sess_m_n is None:
        if include_session_shift:
            z_sess_m_n = rng.normal(0.0, cohort[0].session_logit_sd, size=(n_mc, n_subj)).astype(np.float32)
        else:
            z_sess_m_n = np.zeros((n_mc, n_subj), dtype=np.float32)

    # Pre-draw uniform quantiles (5 slots, 2 tokens, n_mc, n_subj)
    u_draws = rng.random((5, 2, n_mc, n_subj), dtype=np.float32)
    out = np.zeros((N_SNRS, 5, n_mc, n_subj, N_RESPONSES), dtype=np.uint8)

    gamma = 1.0 / float(N_RESPONSES)
    ref_snr = float(CONSONANT_SNR_DB[STIMULI_CONSONANTS[stim_idx]])
    slope = float(cohort[0].slope_per_db)
    snr_q = np.array([l.snr_quiet_db[stim_idx] for l in cohort], dtype=np.float64)
    seff_ref = HumanCalibratedMultiSNRListener._calc_seff_vec(ref_snr, snr_q)
    eta0 = np.array([l.eta0_logit[stim_idx] for l in cohort], dtype=np.float64)
    c_max = np.array([l.ceil_prob[stim_idx] for l in cohort], dtype=np.float64)
    tok_hd = float(cohort[0].token_diag_logit_halfdiff[stim_idx])
    ref_od = np.array([l._ref_offdiag[stim_idx] for l in cohort], dtype=np.float64)   # (98, 11)
    err_pf = np.array([l.error_profile[stim_idx] for l in cohort], dtype=np.float64)  # (98, 11)
    guess_off = cohort[0].marginal_low_snr_profile.copy()
    guess_off[stim_idx] = 0.0
    guess_off /= max(float(guess_off.sum()), 1e-12)

    eye11 = np.eye(N_RESPONSES, dtype=np.uint8)

    for s_idx, snr_val in enumerate(SNR_GRID_DB):
        seff = HumanCalibratedMultiSNRListener._calc_seff_vec(float(snr_val), snr_q)
        d_seff = seff - seff_ref
        dsnr_nom = float(snr_val) - ref_snr

        w_away = 1.0 - np.exp(-0.25 * (d_seff * d_seff))[:, None]
        phon_err = (1.0 - w_away) * ref_od + w_away * err_pf
        w_noise = 1.0 / (1.0 + np.exp(0.45 * (dsnr_nom + 10.0)))
        off_dist = (1.0 - w_noise) * phon_err + w_noise * guess_off[None, :]
        off_dist[:, stim_idx] = 0.0
        off_dist /= np.maximum(off_dist.sum(axis=1, keepdims=True), 1e-12)  # (98, 11)

        for tok_idx, tok_sign in ((0, 1.0), (1, -1.0)):
            eta_base = (eta0 + slope * d_seff + tok_sign * tok_hd)[None, :] + z_sess_m_n  # (n_mc, 98)
            u_act = 1.0 / (1.0 + np.exp(-np.clip(eta_base, -35.0, 35.0)))
            p_corr = np.clip(gamma + (c_max[None, :] - gamma) * u_act, 1e-6, 1.0 - 1e-6)  # (n_mc, 98)

            probs = (1.0 - p_corr[:, :, None]) * off_dist[None, :, :]  # (n_mc, 98, 11)
            probs[:, :, stim_idx] = p_corr
            cdf = np.cumsum(probs, axis=2, dtype=np.float32)
            cdf[:, :, -1] = 1.0001

            for slot_k in range(5):
                u_k = u_draws[slot_k, tok_idx, :, :, None]  # (n_mc, 98, 1)
                resp = np.argmax(u_k < cdf, axis=2)         # (n_mc, 98)
                out[s_idx, slot_k] += eye11[resp]

    return out


def generate_regular_5snr_candidates(min_step: int = 1) -> list[tuple[int, int, int, int, int]]:
    """Generate distinct sorted integer 5-tuples across regular symmetric & skewed grids."""
    cands: set[tuple[int, int, int, int, int]] = set()
    for step in range(min_step, 7):
        for center in range(-15 + 2 * step, 31 - 2 * step):
            tup = (center - 2 * step, center - step, center, center + step, center + 2 * step)
            cands.add(tup)
    # Also include half-skewed integer progressions (e.g. [-3, -1, 0, +2, +5], [-6, -3, -1, +1, +3])
    skew_patterns = [
        (-5, -3, -1, 1, 4),
        (-4, -2, -1, 1, 3),
        (-6, -3, -1, 1, 3),
        (-3, -1, 1, 3, 6),
        (-8, -4, -1, 2, 5),
        (-5, -2, 1, 4, 8),
    ]
    for pat in skew_patterns:
        diffs = [pat[k + 1] - pat[k] for k in range(4)]
        if min(diffs) < min_step:
            continue
        for center in range(-15 - pat[0], 31 - pat[-1]):
            cands.add(tuple(center + p for p in pat))  # type: ignore[arg-type]
    return sorted(cands)


def optimize_5snr_tuple_coord_ascent(
    two_tok_counts: np.ndarray,
    neg_mask: np.ndarray,
    pos_mask: np.ndarray,
    min_step: int = 1,
    n_starts: int = 6,
) -> tuple[tuple[int, int, int, int, int], float]:
    """
    Evaluate all structured 5-SNR grids with adjacent spacing >= ``min_step`` dB,
    then run coordinate ascent from the top ``n_starts`` candidates over all
    valid sorted integer 5-tuples s_1 < s_2 < s_3 < s_4 < s_5 in [-15, +30] dB
    satisfying s_{k+1} - s_k >= ``min_step``.
    """
    cache: dict[tuple[int, int, int, int, int], float] = {}

    def eval_tup(tup: tuple[int, int, int, int, int]) -> float:
        if tup in cache:
            return cache[tup]
        idxs = [SNR_TO_IDX[s] for s in tup]
        c_sum = (
            two_tok_counts[idxs[0], 0]
            + two_tok_counts[idxs[1], 1]
            + two_tok_counts[idxs[2], 2]
            + two_tok_counts[idxs[3], 3]
            + two_tok_counts[idxs[4], 4]
        )
        val = float(fast_loocv_hellinger_aroc_batch(c_sum, neg_mask, pos_mask).mean())
        cache[tup] = val
        return val

    reg_cands = generate_regular_5snr_candidates(min_step=min_step)
    scored = [(eval_tup(t), t) for t in reg_cands]
    scored.sort(reverse=True)

    best_val, best_tup = scored[0]
    for _, start_tup in scored[:n_starts]:
        cur = list(start_tup)
        cur_val = eval_tup(tuple(cur))  # type: ignore[arg-type]
        for _ in range(4):
            improved = False
            for k in range(5):
                lo = -15 if k == 0 else cur[k - 1] + min_step
                hi = 30 if k == 4 else cur[k + 1] - min_step
                for cand_s in range(lo, hi + 1):
                    if cand_s == cur[k]:
                        continue
                    trial = cur.copy()
                    trial[k] = cand_s
                    t_tup = (trial[0], trial[1], trial[2], trial[3], trial[4])
                    v = eval_tup(t_tup)
                    if v > cur_val + 1e-7:
                        cur = trial
                        cur_val = v
                        improved = True
            if not improved:
                break
        if cur_val > best_val:
            best_val = cur_val
            best_tup = (cur[0], cur[1], cur[2], cur[3], cur[4])

    return best_tup, best_val


def subject_bootstrap_delta_ci(
    eval_counts_a: np.ndarray,
    eval_counts_b: np.ndarray,
    is_nh: np.ndarray,
    pos_mask: np.ndarray,
    n_boot: int = 400,
    seed: int = 20261199,
) -> tuple[float, float]:
    """
    Bootstrap 95% CI on Delta A_ROC = A_ROC(B) - A_ROC(A) by resampling the
    32 NH and positive HL human participants with replacement (stratified).
    """
    rng = np.random.default_rng(seed)
    nh_idx = np.where(is_nh)[0]
    pos_idx = np.where(pos_mask)[0]
    n_nh = len(nh_idx)
    n_pos = len(pos_idx)
    m_sub = min(eval_counts_a.shape[0], 250)
    ca = eval_counts_a[:m_sub]
    cb = eval_counts_b[:m_sub]

    deltas = np.zeros(n_boot, dtype=float)
    b_nh_mask = np.zeros(n_nh + n_pos, dtype=bool)
    b_nh_mask[:n_nh] = True
    b_pos_mask = ~b_nh_mask

    for b in range(n_boot):
        s_nh = rng.choice(nh_idx, size=n_nh, replace=True)
        s_pos = rng.choice(pos_idx, size=n_pos, replace=True)
        sel = np.concatenate([s_nh, s_pos])
        ar_a = float(fast_loocv_hellinger_aroc_batch(ca[:, sel], b_nh_mask, b_pos_mask).mean())
        ar_b = float(fast_loocv_hellinger_aroc_batch(cb[:, sel], b_nh_mask, b_pos_mask).mean())
        deltas[b] = ar_b - ar_a

    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def plot_per_vcv_optimization(
    per_vcv_rows: list[dict[str, Any]],
    snr_curves_nh_hl: dict[str, list[float]],
    snr_curves_nh_sl: dict[str, list[float]],
    full_test_rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(17.5, 11.8), dpi=160)
    fig.suptitle(
        "Optimal Single SNR vs. Optimal Set of 5 SNRs per VCV (Calibrated to 98 Human Listeners: 32 NH, 66 HL)",
        fontsize=14.0,
        fontweight="bold",
        y=0.985,
    )

    flat_axes = axes.ravel()
    for ci, row in enumerate(per_vcv_rows):
        ax = flat_axes[ci]
        sc = row["consonant"]
        vcv = row["vcv"]
        cur_s = int(row["current_snr_db"])
        best1_s = int(row["best_single_snr_db"])
        tup5_1 = [int(x) for x in row["best_5snr_min1db_str"].split(",")]
        tup5_2 = [int(x) for x in row["best_5snr_min2db_str"].split(",")]

        y_all = np.asarray(snr_curves_nh_hl[sc], dtype=float)
        y_sl = np.asarray(snr_curves_nh_sl[sc], dtype=float)

        ax.plot(SNR_GRID_DB, y_all, lw=2.2, color="#1f4e79", label="Single SNR: NH vs All HL")
        ax.plot(SNR_GRID_DB, y_sl, lw=1.5, ls="--", color="#2e75b6", alpha=0.85, label="Single SNR: NH vs Slight HL")

        ax.axvline(cur_s, color="#666666", ls=":", lw=1.4, label=f"Current ({cur_s:+d} dB)")
        ax.scatter([best1_s], [row["eval_aroc_best_single"]], color="#d62728", s=52, zorder=5, label=f"Best 1-SNR ({best1_s:+d} dB)")

        # Plot horizontal span bar for Best 5-SNR (step >= 2 dB) and (step >= 1 dB)
        y5_2 = float(row["eval_aroc_best_5snr_min2db"])
        ax.plot([tup5_2[0], tup5_2[-1]], [y5_2, y5_2], color="#2ca02c", lw=2.6, zorder=4)
        ax.scatter(tup5_2, [y5_2] * 5, color="#2ca02c", s=24, zorder=4, label=f"Best 5-SNR ≥2dB ({y5_2:.3f})")

        ax.set_title(
            f"{vcv}  |  1-SNR: {best1_s:+d} dB ({row['eval_aroc_best_single']:.3f})  vs  5-SNR: [{tup5_2[0]:+d}..{tup5_2[-1]:+d}]",
            fontsize=9.3,
            fontweight="bold",
        )
        ax.set_xlim(-15, 30)
        ymin = max(0.45, float(min(y_all.min(), y_sl.min())) - 0.03)
        ymax = min(0.92, float(max(y_all.max(), row["eval_aroc_best_single"], y5_2)) + 0.05)
        ax.set_ylim(ymin, ymax)
        if ci % 4 == 0:
            ax.set_ylabel("10-Trial LOOCV A_ROC")
        if ci >= 8:
            ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.25)
        if ci == 0:
            ax.legend(loc="lower right", fontsize=7.0)

    # Panel 11: Per-VCV Delta A_ROC (Best 5-SNR minus Best Single SNR)
    ax10 = flat_axes[10]
    vcvs = [r["vcv"] for r in per_vcv_rows]
    d_min1 = [100.0 * (r["eval_aroc_best_5snr_min1db"] - r["eval_aroc_best_single"]) for r in per_vcv_rows]
    d_min2 = [100.0 * (r["eval_aroc_best_5snr_min2db"] - r["eval_aroc_best_single"]) for r in per_vcv_rows]
    d_llr2 = [100.0 * (r["eval_aroc_snr_llr_5snr_min2db"] - r["eval_aroc_best_single"]) for r in per_vcv_rows]
    xv = np.arange(len(vcvs))
    bw = 0.27
    ax10.bar(xv - bw, d_min1, bw, color="#5b9bd5", label="5-SNR (≥1 dB step) − Best 1-SNR")
    ax10.bar(xv, d_min2, bw, color="#2ca02c", label="5-SNR (≥2 dB step) − Best 1-SNR")
    ax10.bar(xv + bw, d_llr2, bw, color="#ed7d31", label="5-SNR (SNR-Aware LLR) − Best 1-SNR")
    ax10.axhline(0.0, color="#111111", lw=1.1)
    ax10.set_xticks(xv)
    ax10.set_xticklabels([v[1:-1] for v in vcvs], fontsize=8.5)
    ax10.set_ylabel("ΔA_ROC (% points: 5-SNR − 1-SNR)")
    ax10.set_title("Per-VCV Gain of Best 5-SNR over Best 1-SNR", fontsize=9.5, fontweight="bold")
    ax10.legend(loc="lower left", fontsize=7.0)
    ax10.grid(axis="y", alpha=0.25)

    # Panel 12: Full 100-Trial Test Comparison
    ax11 = flat_axes[11]
    labels_f = [
        "Current\n1-SNR",
        "Optimal\n1-SNR",
        "Optimal\n5-SNR\n(≥1 dB)",
        "Optimal\n5-SNR\n(≥2 dB)",
    ]
    xf = np.arange(len(full_test_rows))
    w = 0.26
    ax11.bar(xf - w, [r["aroc_nh_vs_all_hl"] for r in full_test_rows], w, color="#1f4e79", label="NH vs All HL")
    ax11.bar(xf, [r["aroc_nh_vs_slight_hl"] for r in full_test_rows], w, color="#2e75b6", label="NH vs Slight HL")
    ax11.bar(xf + w, [r["aroc_nh_vs_mild_mod_hl"] for r in full_test_rows], w, color="#70ad47", label="NH vs Mild+Mod HL")
    for idx_f, r in enumerate(full_test_rows):
        ax11.text(idx_f - w, r["aroc_nh_vs_all_hl"] + 0.006, f"{r['aroc_nh_vs_all_hl']:.3f}", ha="center", fontsize=7.3, fontweight="bold")
        ax11.text(idx_f, r["aroc_nh_vs_slight_hl"] + 0.006, f"{r['aroc_nh_vs_slight_hl']:.3f}", ha="center", fontsize=7.3)
    ax11.set_xticks(xf)
    ax11.set_xticklabels(labels_f, fontsize=8.2)
    ax11.set_ylim(0.70, 1.01)
    ax11.set_ylabel("Full 100-Trial LOOCV Hellinger A_ROC")
    ax11.set_title("Full 100-Trial Test: Best 1-SNR vs. Best 5-SNR", fontsize=9.5, fontweight="bold")
    ax11.legend(loc="lower right", fontsize=7.3)
    ax11.grid(axis="y", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize single SNR vs. set of 5 SNRs per VCV.")
    parser.add_argument("--m_search", type=int, default=800, help="Monte Carlo sessions for optimization")
    parser.add_argument("--m_eval", type=int, default=1500, help="Independent held-out Monte Carlo sessions for evaluation")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "human_calibrated_results"))
    args = parser.parse_args()

    t0 = time.time()
    sim_dir = Path(__file__).resolve().parent
    cohort = load_human_multisnr_cohort(sim_dir / "human_data" / "human_listeners_98.json")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    n_subj = len(cohort)
    is_nh = np.array([l.is_nh for l in cohort], dtype=bool)
    is_hl = ~is_nh
    is_slight = np.array([l.archetype == "Slight" for l in cohort], dtype=bool)
    is_mm = np.array([l.archetype in ("Mild", "Moderate") for l in cohort], dtype=bool)

    # Shared session-level vigilance shifts across consonants for coherent 100-trial sessions!
    rng_s = np.random.default_rng(20261120)
    rng_e = np.random.default_rng(20261121)
    z_search = rng_s.normal(0.0, cohort[0].session_logit_sd, size=(args.m_search, n_subj)).astype(np.float32)
    z_eval = rng_e.normal(0.0, cohort[0].session_logit_sd, size=(args.m_eval, n_subj)).astype(np.float32)

    per_vcv_rows: list[dict[str, Any]] = []
    snr_curves_nh_hl: dict[str, list[float]] = {}
    snr_curves_nh_sl: dict[str, list[float]] = {}

    # Accumulate held-out (m_eval, 98, 10, 11) full-test matrices for the 4 global designs
    full_cur_1snr = np.zeros((args.m_eval, n_subj, N_STIMULI, N_RESPONSES), dtype=np.float32)
    full_best_1snr = np.zeros((args.m_eval, n_subj, N_STIMULI, N_RESPONSES), dtype=np.float32)
    full_best_5snr_1db = np.zeros((args.m_eval, n_subj, N_STIMULI, N_RESPONSES), dtype=np.float32)
    full_best_5snr_2db = np.zeros((args.m_eval, n_subj, N_STIMULI, N_RESPONSES), dtype=np.float32)

    # Also accumulate SNR-aware slot counts (m_eval, 98, 50, 11) for full-test SNR-aware evaluation
    full_slots_best_1snr = np.zeros((args.m_eval, n_subj, N_STIMULI * 5, N_RESPONSES), dtype=np.float32)
    full_slots_5snr_1db = np.zeros((args.m_eval, n_subj, N_STIMULI * 5, N_RESPONSES), dtype=np.float32)
    full_slots_5snr_2db = np.zeros((args.m_eval, n_subj, N_STIMULI * 5, N_RESPONSES), dtype=np.float32)

    for ci, sc in enumerate(STIMULI_CONSONANTS):
        vcv_name = f"A{sc}A"
        cur_snr = int(CONSONANT_SNR_DB[sc])
        print(f"[{ci+1:2d}/10] Optimizing {vcv_name:4s} (Current SNR = {cur_snr:+3d} dB)...", flush=True)

        ttc_search = sample_crn_two_token_counts(
            cohort, ci, args.m_search, seed=20261200 + ci, z_sess_m_n=z_search
        )

        # 1. Sweep all 46 Single SNRs on search ensemble
        search_1snr_all = np.zeros(N_SNRS, dtype=float)
        for s_idx in range(N_SNRS):
            c10 = ttc_search[s_idx].sum(axis=0)  # (m_search, 98, 11)
            search_1snr_all[s_idx] = float(fast_loocv_hellinger_aroc_batch(c10, is_nh, is_hl).mean())

        best1_idx = int(np.argmax(search_1snr_all))
        best1_snr = int(SNR_GRID_DB[best1_idx])

        # Also find Slight-HL single-SNR optimum for diagnostic insight
        search_1snr_sl = np.array([
            float(fast_loocv_hellinger_aroc_batch(ttc_search[si].sum(axis=0), is_nh, is_slight).mean())
            for si in range(N_SNRS)
        ])
        best1_sl_snr = int(SNR_GRID_DB[int(np.argmax(search_1snr_sl))])

        # 2. Optimize Set of 5 Distinct SNRs (min spacing >= 1 dB, and practical spread >= 2 dB)
        best5_1db, _ = optimize_5snr_tuple_coord_ascent(ttc_search, is_nh, is_hl, min_step=1, n_starts=6)
        best5_2db, _ = optimize_5snr_tuple_coord_ascent(ttc_search, is_nh, is_hl, min_step=2, n_starts=6)
        del ttc_search

        # 3. Evaluate unbiasedly on independent held-out ensemble (m_eval = 1,500)!
        ttc_eval = sample_crn_two_token_counts(
            cohort, ci, args.m_eval, seed=20262200 + ci, z_sess_m_n=z_eval
        )

        eval_curve_all = []
        eval_curve_sl = []
        for s_idx in range(N_SNRS):
            c10_e = ttc_eval[s_idx].sum(axis=0)
            eval_curve_all.append(round(float(fast_loocv_hellinger_aroc_batch(c10_e, is_nh, is_hl).mean()), 5))
            eval_curve_sl.append(round(float(fast_loocv_hellinger_aroc_batch(c10_e, is_nh, is_slight).mean()), 5))
        snr_curves_nh_hl[sc] = eval_curve_all
        snr_curves_nh_sl[sc] = eval_curve_sl

        cur_idx = SNR_TO_IDX[cur_snr]
        c_cur = ttc_eval[cur_idx].sum(axis=0)
        c_b1 = ttc_eval[best1_idx].sum(axis=0)

        idx5_1 = [SNR_TO_IDX[s] for s in best5_1db]
        slots_5_1 = np.stack([ttc_eval[idx5_1[k], k] for k in range(5)], axis=2)  # (m_eval, 98, 5, 11)
        c_b5_1 = slots_5_1.sum(axis=2)

        idx5_2 = [SNR_TO_IDX[s] for s in best5_2db]
        slots_5_2 = np.stack([ttc_eval[idx5_2[k], k] for k in range(5)], axis=2)  # (m_eval, 98, 5, 11)
        c_b5_2 = slots_5_2.sum(axis=2)

        slots_b1 = np.transpose(ttc_eval[best1_idx], (1, 2, 0, 3))  # (m_eval, 98, 5, 11)

        # Save into full 100-trial accumulators
        full_cur_1snr[:, :, ci, :] = c_cur
        full_best_1snr[:, :, ci, :] = c_b1
        full_best_5snr_1db[:, :, ci, :] = c_b5_1
        full_best_5snr_2db[:, :, ci, :] = c_b5_2

        full_slots_best_1snr[:, :, ci * 5 : (ci + 1) * 5, :] = slots_b1
        full_slots_5snr_1db[:, :, ci * 5 : (ci + 1) * 5, :] = slots_5_1
        full_slots_5snr_2db[:, :, ci * 5 : (ci + 1) * 5, :] = slots_5_2

        ar_cur = float(fast_loocv_hellinger_aroc_batch(c_cur, is_nh, is_hl).mean())
        ar_b1 = float(fast_loocv_hellinger_aroc_batch(c_b1, is_nh, is_hl).mean())
        ar_b5_1 = float(fast_loocv_hellinger_aroc_batch(c_b5_1, is_nh, is_hl).mean())
        ar_b5_2 = float(fast_loocv_hellinger_aroc_batch(c_b5_2, is_nh, is_hl).mean())

        ar_sl_cur = float(fast_loocv_hellinger_aroc_batch(c_cur, is_nh, is_slight).mean())
        ar_sl_b1 = float(fast_loocv_hellinger_aroc_batch(c_b1, is_nh, is_slight).mean())
        ar_sl_b5_1 = float(fast_loocv_hellinger_aroc_batch(c_b5_1, is_nh, is_slight).mean())
        ar_sl_b5_2 = float(fast_loocv_hellinger_aroc_batch(c_b5_2, is_nh, is_slight).mean())

        # SNR-aware slot-specific Hellinger & Neyman-Pearson LLR on held-out subset (first 400 draws for speed)
        m_sub = min(args.m_eval, 400)
        ar_slot_h_b5_2 = float(fast_loocv_hellinger_aroc_batch(slots_5_2[:m_sub], is_nh, is_hl).mean())
        ar_llr_b1 = float(fast_loocv_slot_llr_aroc_batch(slots_b1[:m_sub], is_nh, is_hl).mean())
        ar_llr_b5_1 = float(fast_loocv_slot_llr_aroc_batch(slots_5_1[:m_sub], is_nh, is_hl).mean())
        ar_llr_b5_2 = float(fast_loocv_slot_llr_aroc_batch(slots_5_2[:m_sub], is_nh, is_hl).mean())

        ci_lo_1db, ci_hi_1db = subject_bootstrap_delta_ci(c_b1, c_b5_1, is_nh, is_hl, n_boot=300, seed=20263000 + ci)
        ci_lo_2db, ci_hi_2db = subject_bootstrap_delta_ci(c_b1, c_b5_2, is_nh, is_hl, n_boot=300, seed=20264000 + ci)

        # Expected NH and HL PC% at Best Single SNR
        nh_pc_b1 = float(100.0 * np.mean([cohort[i].response_probs(ci, float(best1_snr))[ci] for i in np.where(is_nh)[0]]))
        hl_pc_b1 = float(100.0 * np.mean([cohort[i].response_probs(ci, float(best1_snr))[ci] for i in np.where(is_hl)[0]]))

        # Near-optimal plateau range (within 0.005 A_ROC of peak single SNR)
        near_opt_snrs = [int(SNR_GRID_DB[k]) for k, v in enumerate(eval_curve_all) if v >= ar_b1 - 0.005]

        per_vcv_rows.append({
            "vcv": vcv_name,
            "consonant": sc,
            "current_snr_db": cur_snr,
            "best_single_snr_db": best1_snr,
            "near_optimal_single_range_db": f"{min(near_opt_snrs):+d}..{max(near_opt_snrs):+d}",
            "best_single_snr_slight_db": best1_sl_snr,
            "nh_pc_at_best_single_pct": round(nh_pc_b1, 1),
            "hl_pc_at_best_single_pct": round(hl_pc_b1, 1),
            "best_5snr_min1db_str": ",".join(f"{s:+d}" for s in best5_1db),
            "best_5snr_min2db_str": ",".join(f"{s:+d}" for s in best5_2db),
            "eval_aroc_current_single": round(ar_cur, 4),
            "eval_aroc_best_single": round(ar_b1, 4),
            "eval_aroc_best_5snr_min1db": round(ar_b5_1, 4),
            "eval_aroc_best_5snr_min2db": round(ar_b5_2, 4),
            "delta_aroc_5snr1db_minus_best1": round(ar_b5_1 - ar_b1, 4),
            "delta_aroc_5snr1db_ci95": f"[{ci_lo_1db:+.4f}, {ci_hi_1db:+.4f}]",
            "delta_aroc_5snr2db_minus_best1": round(ar_b5_2 - ar_b1, 4),
            "delta_aroc_5snr2db_ci95": f"[{ci_lo_2db:+.4f}, {ci_hi_2db:+.4f}]",
            "eval_aroc_slight_current": round(ar_sl_cur, 4),
            "eval_aroc_slight_best_single": round(ar_sl_b1, 4),
            "eval_aroc_slight_5snr_min1db": round(ar_sl_b5_1, 4),
            "eval_aroc_slight_5snr_min2db": round(ar_sl_b5_2, 4),
            "eval_aroc_slot_hellinger_5snr_min2db": round(ar_slot_h_b5_2, 4),
            "eval_aroc_snr_llr_best_single": round(ar_llr_b1, 4),
            "eval_aroc_snr_llr_5snr_min1db": round(ar_llr_b5_1, 4),
            "eval_aroc_snr_llr_5snr_min2db": round(ar_llr_b5_2, 4),
        })
        del ttc_eval

    # Full 100-Trial Test Evaluation across all 10 VCVs
    print("Evaluating full 100-trial test designs & bootstrap 95% CIs...", flush=True)
    full_specs = [
        ("current_qVCV_single_snr", "Current Published qVCV Single SNRs (5x each)", full_cur_1snr, None),
        ("optimal_single_snr", "Best Single SNR per VCV (5x each)", full_best_1snr, full_slots_best_1snr),
        ("optimal_5snr_min1db", "Best 5-SNR Set per VCV (step >= 1 dB)", full_best_5snr_1db, full_slots_5snr_1db),
        ("optimal_5snr_min2db", "Best Spread 5-SNR Set per VCV (step >= 2 dB)", full_best_5snr_2db, full_slots_5snr_2db),
    ]

    full_test_rows: list[dict[str, Any]] = []
    m_sub_full = min(args.m_eval, 300)
    for d_key, d_label, mat_4d, slots_4d in full_specs:
        ar_all = float(fast_loocv_hellinger_aroc_batch(mat_4d, is_nh, is_hl).mean())
        ar_sl = float(fast_loocv_hellinger_aroc_batch(mat_4d, is_nh, is_slight).mean())
        ar_mm = float(fast_loocv_hellinger_aroc_batch(mat_4d, is_nh, is_mm).mean())

        ci_lo, ci_hi = subject_bootstrap_delta_ci(full_best_1snr, mat_4d, is_nh, is_hl, n_boot=400, seed=20265000)
        ci_sl_lo, ci_sl_hi = subject_bootstrap_delta_ci(full_best_1snr, mat_4d, is_nh, is_slight, n_boot=400, seed=20265100)

        ar_llr = (
            round(float(fast_loocv_slot_llr_aroc_batch(slots_4d[:m_sub_full], is_nh, is_hl).mean()), 4)
            if slots_4d is not None
            else None
        )
        ar_llr_sl = (
            round(float(fast_loocv_slot_llr_aroc_batch(slots_4d[:m_sub_full], is_nh, is_slight).mean()), 4)
            if slots_4d is not None
            else None
        )

        full_test_rows.append({
            "design": d_key,
            "description": d_label,
            "aroc_nh_vs_all_hl": round(ar_all, 4),
            "delta_vs_opt_single_all_hl": round(ar_all - float(fast_loocv_hellinger_aroc_batch(full_best_1snr, is_nh, is_hl).mean()), 4),
            "delta_vs_opt_single_all_hl_ci95": f"[{ci_lo:+.4f}, {ci_hi:+.4f}]",
            "aroc_nh_vs_slight_hl": round(ar_sl, 4),
            "delta_vs_opt_single_slight_hl": round(ar_sl - float(fast_loocv_hellinger_aroc_batch(full_best_1snr, is_nh, is_slight).mean()), 4),
            "delta_vs_opt_single_slight_hl_ci95": f"[{ci_sl_lo:+.4f}, {ci_sl_hi:+.4f}]",
            "aroc_nh_vs_mild_mod_hl": round(ar_mm, 4),
            "snr_aware_llr_aroc_nh_vs_all_hl": ar_llr,
            "snr_aware_llr_aroc_nh_vs_slight_hl": ar_llr_sl,
        })

    write_dict_rows_csv(per_vcv_rows, out_dir / "optimal_snrs_per_vcv.csv")
    write_dict_rows_csv(full_test_rows, out_dir / "full_100trial_single_vs_5snr.csv")

    fig_path = out_dir / "fig_optimal_snrs_per_vcv.png"
    plot_per_vcv_optimization(per_vcv_rows, snr_curves_nh_hl, snr_curves_nh_sl, full_test_rows, fig_path)

    summary = {
        "elapsed_sec": round(time.time() - t0, 2),
        "m_search": args.m_search,
        "m_eval": args.m_eval,
        "per_vcv_results": per_vcv_rows,
        "full_100trial_comparison": full_test_rows,
    }
    with (out_dir / "optimal_snrs_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Done in {summary['elapsed_sec']:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
