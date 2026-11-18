#!/usr/bin/env python3
"""
exp_suprathreshold_efficiency.py

Evaluates the relative efficiency of:
  1. Single SNR per VCV — Current published qVCV SNRs (NH-vs-HL screening SNRs)
  2. Single SNR per VCV — Bayes A-Optimal Single SNRs for Suprathreshold Deficit
  3. Set of 5 SNRs per VCV — Centered around Current Ref (Ref + [-8, -4, 0, +4, +8] dB)
  4. Set of 5 SNRs per VCV — Suprathreshold-Balanced 5-SNR Span (Ref + [-12, -7, -2, +3, +8] dB)
  5. Adaptive Trial-by-Trial SNR — Uncoupled Per-VCV Bayesian Tracker (1-dB bank, audiogram-blind)
  6. Adaptive Trial-by-Trial SNR — Hierarchical Coupled Tracker on Wide 5-SNR Bank (audiogram-blind)
  7. Adaptive Trial-by-Trial SNR — Hierarchical Coupled Tracker on 1-dB Bank (audiogram-blind)
  8. Adaptive Joint (Consonant + SNR) — Audiogram-Informed Hierarchical Tracker (1-dB bank)

for quantifying SUPRATHRESHOLD HEARING DEFICITS across the 98 human-calibrated
listeners (32 Normal, 23 Slight, 26 Mild, 17 Moderate) over N = 10..100 trials.
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

EVAL_TRIALS = [10, 15, 20, 30, 40, 60, 80, 100]
SNR_BANK_1DB = np.arange(-18.0, 31.0, 1.0, dtype=float)  # 49 SNRs: -18..+30 dB


def seff_vec(snr_db: np.ndarray | float, sq_db: np.ndarray | float) -> np.ndarray:
    ln10_10 = np.log(10.0) / 10.0
    a = -np.asarray(snr_db, dtype=float) * ln10_10
    b = -np.asarray(sq_db, dtype=float) * ln10_10
    m = np.maximum(a, b)
    return -10.0 * (m + np.log(np.exp(a - m) + np.exp(b - m))) / np.log(10.0)


def build_balanced_consonant_order(rng: np.random.Generator, b_sessions: int, max_trials: int) -> tuple[np.ndarray, np.ndarray]:
    """Balanced 10-consonant blocks (no boundary repeat) + alternating token indices (0 vs 1)."""
    n_blocks = (max_trials + 9) // 10
    c_sched = np.zeros((b_sessions, n_blocks * 10), dtype=np.int32)
    t_sched = np.zeros((b_sessions, n_blocks * 10), dtype=np.int32)
    tok_parity = rng.integers(0, 2, size=(b_sessions, 10), dtype=np.int32)

    for s in range(b_sessions):
        prev_last = -1
        for blk in range(n_blocks):
            perm = rng.permutation(10)
            if prev_last >= 0 and perm[0] == prev_last:
                perm[0], perm[1] = perm[1], perm[0]
            c_sched[s, blk * 10 : (blk + 1) * 10] = perm
            for pos, ci in enumerate(perm):
                t_sched[s, blk * 10 + pos] = (tok_parity[s, ci] + blk) % 2
            prev_last = int(perm[-1])

    return c_sched[:, :max_trials], t_sched[:, :max_trials]


def build_stratified_5slot_schedule(
    rng: np.random.Generator, b_sessions: int, max_trials: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Stratified Latin-square schedule over (10 consonants x 2 tokens x 5 SNR slots)
    so every 20-trial block visits all 10 consonants, both tokens, and all 5 SNR
    slots 4 times each, and 100 trials visits all 100 stimuli exactly once.
    """
    n_blocks = (max_trials + 19) // 20
    c_sched = np.zeros((b_sessions, n_blocks * 20), dtype=np.int32)
    t_sched = np.zeros((b_sessions, n_blocks * 20), dtype=np.int32)
    k_sched = np.zeros((b_sessions, n_blocks * 20), dtype=np.int32)

    for s in range(b_sessions):
        perm_c = rng.permutation(10)
        slot_perm = rng.permutation(5)
        prev_c = -1
        for blk in range(n_blocks):
            items = []
            for rank, ci in enumerate(perm_c):
                for tok in (0, 1):
                    sl = int(slot_perm[(rank + 2 * blk + tok) % 5])
                    items.append((int(ci), tok, sl))
            rng.shuffle(items)
            if prev_c >= 0 and items[0][0] == prev_c:
                sw = next((m for m in range(1, len(items)) if items[m][0] != prev_c), 1)
                items[0], items[sw] = items[sw], items[0]
            for pos, (ci, tok, sl) in enumerate(items):
                idx = blk * 20 + pos
                c_sched[s, idx] = ci
                t_sched[s, idx] = tok
                k_sched[s, idx] = sl
            prev_c = items[-1][0]

    return c_sched[:, :max_trials], t_sched[:, :max_trials], k_sched[:, :max_trials]


def compute_bayes_optimal_single_snrs(
    alpha_98x10: np.ndarray,
    sq_98x10: np.ndarray,
    ref_snr: np.ndarray,
    slope: float,
    gamma: float,
    ceil_p: float,
    sd_g: float,
    sd_d: float,
) -> np.ndarray:
    """
    Coordinate-descent Bayes A-optimal single SNR per consonant minimizing expected
    posterior RMSE across the 98 human listeners (avoiding harmonic-mean starvation).
    """
    span = ceil_p - gamma
    cand = np.arange(-16.0, 25.0, 1.0)
    cov0 = (sd_g ** 2) * np.ones((10, 10)) + (sd_d ** 2) * np.eye(10)
    lam0 = np.linalg.inv(cov0)
    one10 = np.full((10, 1), 0.1)

    # Precompute FI[98, 10, len(cand)]
    eta = alpha_98x10[:, :, None] + slope * seff_vec(cand[None, None, :], sq_98x10[:, :, None])
    u = 1.0 / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
    p = np.clip(gamma + span * u, 1e-7, 1.0 - 1e-7)
    dp = -slope * span * u * (1.0 - u)
    fi_table = (dp * dp) / (p * (1.0 - p))  # (98, 10, M)

    best_idx = np.array([int(np.argmin(np.abs(cand - r))) for r in ref_snr], dtype=int)
    for _ in range(3):
        for c in range(10):
            best_sc = 1e9
            best_m = best_idx[c]
            for m in range(len(cand)):
                idx_try = best_idx.copy()
                idx_try[c] = m
                w = 10.0 * fi_table[:, np.arange(10), idx_try]  # (98, 10)
                H = np.broadcast_to(lam0, (98, 10, 10)).copy()
                H[:, np.arange(10), np.arange(10)] += w
                pcov = np.linalg.inv(H)
                vg = np.einsum("ia,iab,ib->i", np.broadcast_to(one10.ravel(), (98, 10)), pcov, np.broadcast_to(one10.ravel(), (98, 10)))
                vc = np.mean(np.diagonal(pcov, axis1=1, axis2=2), axis=1)
                sc = float(np.sqrt(np.mean(vg)) + 0.25 * np.sqrt(np.mean(vc)))
                if sc < best_sc:
                    best_sc = sc
                    best_m = m
            best_idx[c] = best_m
    return cand[best_idx]


def decode_hierarchical_suprathreshold_map(
    trial_c: np.ndarray,       # (B, N) int32
    trial_tok: np.ndarray,     # (B, N) int32 (0 or 1)
    trial_snr: np.ndarray,     # (B, N) float
    trial_corr: np.ndarray,    # (B, N) float (0.0 or 1.0)
    sq_b_10: np.ndarray,       # (B, 10)
    alpha0: np.ndarray,        # (10,)
    tok_hd: np.ndarray,        # (10,)
    slope: float,
    gamma: float,
    ceil_p: float,
    prior_mu_g: float,
    prior_sd_g: float,
    prior_sd_delta: float,
    n_iter: int = 12,
) -> np.ndarray:
    """Exact vectorized Newton-Raphson MAP decoder for 10-consonant suprathreshold D_{i,c} (dB)."""
    b_tot, _ = trial_c.shape
    span = ceil_p - gamma

    b_idx = np.arange(b_tot)[:, None]
    sq_bt = sq_b_10[b_idx, trial_c]
    seff_bt = seff_vec(trial_snr, sq_bt)
    tok_sign = np.where(trial_tok == 0, 1.0, -1.0)
    base_eta = alpha0[trial_c] + slope * seff_bt + tok_sign * tok_hd[trial_c]

    cov0 = (prior_sd_g ** 2) * np.ones((10, 10), dtype=float) + (prior_sd_delta ** 2) * np.eye(10, dtype=float)
    lam0 = np.linalg.inv(cov0)
    mu0 = np.full(10, prior_mu_g, dtype=float)

    d = np.full((b_tot, 10), prior_mu_g, dtype=float)
    one_hot_c = np.eye(10, dtype=float)[trial_c]

    for _ in range(n_iter):
        d_bt = d[b_idx, trial_c]
        eta = np.clip(base_eta - slope * d_bt, -32.0, 32.0)
        u = 1.0 / (1.0 + np.exp(-eta))
        p = np.clip(gamma + span * u, 1e-7, 1.0 - 1e-7)

        dp_dd = -slope * span * u * (1.0 - u)
        d2p_dd2 = (slope ** 2) * span * u * (1.0 - u) * (1.0 - 2.0 * u)

        dL_dp = trial_corr / p - (1.0 - trial_corr) / (1.0 - p)
        d2L_dp2 = -trial_corr / (p * p) - (1.0 - trial_corr) / ((1.0 - p) ** 2)

        g_bt = dL_dp * dp_dd
        neg_h_bt = np.maximum(-(d2L_dp2 * (dp_dd ** 2) + dL_dp * d2p_dd2), 0.0)

        grad_like = np.einsum("bn,bnc->bc", g_bt, one_hot_c)
        diag_h = np.einsum("bn,bnc->bc", neg_h_bt, one_hot_c)

        grad = grad_like - (d - mu0[None, :]) @ lam0
        H = np.broadcast_to(lam0, (b_tot, 10, 10)).copy()
        H[:, np.arange(10), np.arange(10)] += diag_h
        step = np.clip(np.linalg.solve(H, grad[:, :, None])[:, :, 0], -3.0, 3.0)
        d = d + step
        if float(np.max(np.abs(step))) < 1e-4:
            break

    return d


def simulate_arm_trials(
    arm_name: str,
    n_reps: int,
    max_trials: int,
    seed: int,
    alpha_true: np.ndarray,     # (98, 10)
    snr_q_true: np.ndarray,     # (98, 10)
    alpha0: np.ndarray,         # (10,)
    tok_hd: np.ndarray,         # (10,)
    slope: float,
    gamma: float,
    ceil_p: float,
    prior_mu_d: float,
    prior_sd_g: float,
    prior_sd_d: float,
    static_grid_10xK: np.ndarray | None = None,
    adaptive_bank_10xM: np.ndarray | None = None,
    hierarchical_coupling: bool = True,
    audiogram_informed_joint: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Simulate n_reps x 98 sessions of max_trials trials."""
    n_subj = alpha_true.shape[0]
    b_tot = n_reps * n_subj
    subj_of_b = np.tile(np.arange(n_subj), n_reps)

    rng = np.random.default_rng(seed)
    u_resp = rng.random((b_tot, max_trials))
    span = ceil_p - gamma

    alpha_b = alpha_true[subj_of_b]
    sq_b = snr_q_true[subj_of_b]
    b_idx = np.arange(b_tot)

    trial_c = np.zeros((b_tot, max_trials), dtype=np.int32)
    trial_tok = np.zeros((b_tot, max_trials), dtype=np.int32)
    trial_snr = np.zeros((b_tot, max_trials), dtype=float)
    trial_corr = np.zeros((b_tot, max_trials), dtype=float)

    if static_grid_10xK is not None:
        k_slots = static_grid_10xK.shape[1]
        if k_slots == 1:
            c_sch, t_sch = build_balanced_consonant_order(rng, b_tot, max_trials)
            trial_c[:] = c_sch
            trial_tok[:] = t_sch
            trial_snr[:] = static_grid_10xK[c_sch, 0]
        else:
            c_sch, t_sch, k_sch = build_stratified_5slot_schedule(rng, b_tot, max_trials)
            trial_c[:] = c_sch
            trial_tok[:] = t_sch
            trial_snr[:] = static_grid_10xK[c_sch, k_sch]

        for t in range(max_trials):
            ci = trial_c[:, t]
            ti = trial_tok[:, t]
            sn = trial_snr[:, t]
            tsign = np.where(ti == 0, 1.0, -1.0)
            eta = alpha_b[b_idx, ci] + slope * seff_vec(sn, sq_b[b_idx, ci]) + tsign * tok_hd[ci]
            p = gamma + span / (1.0 + np.exp(-np.clip(eta, -35.0, 35.0)))
            trial_corr[:, t] = (u_resp[:, t] < p).astype(float)

        return trial_c, trial_tok, trial_snr, trial_corr

    assert adaptive_bank_10xM is not None
    m_bank = adaptive_bank_10xM.shape[1]

    if not audiogram_informed_joint:
        # 100% AUDIOGRAM-BLIND balanced-consonant adaptive SNR tracker:
        c_sch, t_sch = build_balanced_consonant_order(rng, b_tot, max_trials)
        trial_c[:] = c_sch
        trial_tok[:] = t_sch

        tau0 = -alpha0 / slope
        mu_theta = 4.0
        sd_g = 6.5 if hierarchical_coupling else 1e-3
        sd_d = 4.5 if hierarchical_coupling else 7.8

        cov_th = (sd_g ** 2) * np.ones((10, 10), dtype=float) + (sd_d ** 2) * np.eye(10, dtype=float)
        lam_th = np.linalg.inv(cov_th)
        theta_hat = np.full((b_tot, 10), mu_theta, dtype=float)
        theta_sd = np.full((b_tot, 10), np.sqrt(sd_g ** 2 + sd_d ** 2), dtype=float)

        gh_nodes = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)], dtype=float)
        gh_weights = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=float)

        for t in range(max_trials):
            ci = trial_c[:, t]
            ti = trial_tok[:, t]
            cand_snrs = adaptive_bank_10xM[ci]
            th_cur = theta_hat[b_idx, ci][:, None]
            sd_cur = theta_sd[b_idx, ci][:, None]
            tau0_cur = tau0[ci][:, None]

            efi = np.zeros((b_tot, m_bank), dtype=float)
            for q_node, q_wt in zip(gh_nodes, gh_weights):
                th_q = th_cur + q_node * sd_cur
                eta_q = np.clip(slope * (cand_snrs - (tau0_cur + th_q)), -30.0, 30.0)
                uq = 1.0 / (1.0 + np.exp(-eta_q))
                pq = np.clip(gamma + span * uq, 1e-6, 1.0 - 1e-6)
                dpq = slope * span * uq * (1.0 - uq)
                efi += q_wt * (dpq * dpq) / (pq * (1.0 - pq))

            best_m = np.argmax(efi, axis=1)
            sn = cand_snrs[b_idx, best_m]
            trial_snr[:, t] = sn

            tsign = np.where(ti == 0, 1.0, -1.0)
            eta_true = alpha_b[b_idx, ci] + slope * seff_vec(sn, sq_b[b_idx, ci]) + tsign * tok_hd[ci]
            p_true = gamma + span / (1.0 + np.exp(-np.clip(eta_true, -35.0, 35.0)))
            trial_corr[:, t] = (u_resp[:, t] < p_true).astype(float)

            t_done = t + 1
            c_sub = trial_c[:, :t_done]
            tok_sub = trial_tok[:, :t_done]
            snr_sub = trial_snr[:, :t_done]
            cor_sub = trial_corr[:, :t_done]

            b_col = b_idx[:, None]
            ts_sub = np.where(tok_sub == 0, 1.0, -1.0)
            base_blind = slope * (snr_sub - tau0[c_sub]) + ts_sub * tok_hd[c_sub]
            oh_sub = np.eye(10, dtype=float)[c_sub]

            for _ in range(2):
                th_bt = theta_hat[b_col, c_sub]
                eta_b = np.clip(base_blind - slope * th_bt, -30.0, 30.0)
                ub = 1.0 / (1.0 + np.exp(-eta_b))
                pb = np.clip(gamma + span * ub, 1e-7, 1.0 - 1e-7)
                dp_dth = -slope * span * ub * (1.0 - ub)
                dL_dp = cor_sub / pb - (1.0 - cor_sub) / (1.0 - pb)
                d2L_dp2 = -cor_sub / (pb * pb) - (1.0 - cor_sub) / ((1.0 - pb) ** 2)
                g_bt = dL_dp * dp_dth
                w_bt = np.maximum(-d2L_dp2 * (dp_dth ** 2), 0.0)

                g_like = np.einsum("bn,bnc->bc", g_bt, oh_sub)
                w_diag = np.einsum("bn,bnc->bc", w_bt, oh_sub)
                grad = g_like - (theta_hat - mu_theta) @ lam_th
                H = np.broadcast_to(lam_th, (b_tot, 10, 10)).copy()
                H[:, np.arange(10), np.arange(10)] += w_diag
                step = np.clip(np.linalg.solve(H, grad[:, :, None])[:, :, 0], -3.5, 3.5)
                theta_hat = theta_hat + step

            if hierarchical_coupling:
                inv_vd = 1.0 / (sd_d ** 2)
                w_tilde = w_diag / (1.0 + (sd_d ** 2) * w_diag)
                var_g = 1.0 / ((1.0 / (sd_g ** 2)) + np.sum(w_tilde, axis=1, keepdims=True))
                var_c = 1.0 / (w_diag + inv_vd) + ((inv_vd / (w_diag + inv_vd)) ** 2) * var_g
                theta_sd = np.sqrt(np.maximum(var_c, 0.05))
            else:
                theta_sd = np.sqrt(1.0 / (w_diag + 1.0 / (sd_d ** 2)))

        return trial_c, trial_tok, trial_snr, trial_corr

    # AUDIOGRAM-INFORMED JOINT (CONSONANT + SNR) ADAPTIVE TRACKER:
    # Uses known pure-tone audibility ceiling sq_b[b, c] to avoid wasting trials on
    # consonants that are inaudible at 60 dB SPL, choosing (consonant c, SNR s) to
    # maximize expected reduction in posterior suprathreshold variance!
    cov0 = (prior_sd_g ** 2) * np.ones((10, 10), dtype=float) + (prior_sd_d ** 2) * np.eye(10, dtype=float)
    lam0 = np.linalg.inv(cov0)
    d_hat = np.full((b_tot, 10), prior_mu_d, dtype=float)
    post_cov = np.broadcast_to(cov0, (b_tot, 10, 10)).copy()
    tok_counts = rng.integers(0, 2, size=(b_tot, 10), dtype=np.int32)
    prev_c = np.full(b_tot, -1, dtype=np.int32)

    # Precompute seff_bank[B, 10, M]:
    seff_bank = seff_vec(adaptive_bank_10xM[None, :, :], sq_b[:, :, None])  # (B, 10, M)
    gh_nodes = np.array([-np.sqrt(3.0), 0.0, np.sqrt(3.0)], dtype=float)
    gh_weights = np.array([1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0], dtype=float)
    one10 = np.full(10, 0.1, dtype=float)

    for t in range(max_trials):
        sd_cur = np.sqrt(np.maximum(np.diagonal(post_cov, axis1=1, axis2=2), 0.05))  # (B, 10)
        efi_bcm = np.zeros((b_tot, 10, m_bank), dtype=float)
        for q_node, q_wt in zip(gh_nodes, gh_weights):
            d_q = d_hat + q_node * sd_cur  # (B, 10)
            eta_q = np.clip(alpha0[None, :, None] + slope * (seff_bank - d_q[:, :, None]), -30.0, 30.0)
            uq = 1.0 / (1.0 + np.exp(-eta_q))
            pq = np.clip(gamma + span * uq, 1e-6, 1.0 - 1e-6)
            dpq = -slope * span * uq * (1.0 - uq)
            efi_bcm += q_wt * (dpq * dpq) / (pq * (1.0 - pq))

        best_m_bc = np.argmax(efi_bcm, axis=2)                      # (B, 10)
        max_fi_bc = np.take_along_axis(efi_bcm, best_m_bc[:, :, None], axis=2)[:, :, 0]  # (B, 10)

        # Expected rank-1 variance reduction on composite objective: Var(D_global) + 0.15 * mean_c Var(D_c)
        # If we add information fi on consonant c, ΔCov = -(Cov[:, :, c] Cov[:, c, :]) * fi / (1 + fi * Cov[:, c, c])
        cov_1 = post_cov @ one10                                    # (B, 10): Cov_c dot (1/10)
        diag_cov = np.diagonal(post_cov, axis1=1, axis2=2)          # (B, 10)
        sq_norm_col = np.sum(post_cov ** 2, axis=1)                 # (B, 10): sum_j Cov_{j,c}^2
        denom = 1.0 + max_fi_bc * diag_cov
        gain_glob = (max_fi_bc * (cov_1 ** 2)) / denom
        gain_cons = (max_fi_bc * (sq_norm_col / 10.0)) / denom
        util_bc = gain_glob + 0.15 * gain_cons
        # Prevent immediate consecutive consonant repeats (matching qVCV clinical constraint):
        if t > 0:
            util_bc[b_idx, prev_c] = -1.0

        ci = np.argmax(util_bc, axis=1).astype(np.int32)
        prev_c = ci
        ti = (tok_counts[b_idx, ci] % 2).astype(np.int32)
        tok_counts[b_idx, ci] += 1
        mi = best_m_bc[b_idx, ci]
        sn = adaptive_bank_10xM[ci, mi]

        trial_c[:, t] = ci
        trial_tok[:, t] = ti
        trial_snr[:, t] = sn

        tsign = np.where(ti == 0, 1.0, -1.0)
        eta_true = alpha_b[b_idx, ci] + slope * seff_vec(sn, sq_b[b_idx, ci]) + tsign * tok_hd[ci]
        p_true = gamma + span / (1.0 + np.exp(-np.clip(eta_true, -35.0, 35.0)))
        trial_corr[:, t] = (u_resp[:, t] < p_true).astype(float)

        # Online 2-step Newton update of d_hat and post_cov:
        t_done = t + 1
        c_sub = trial_c[:, :t_done]
        tok_sub = trial_tok[:, :t_done]
        snr_sub = trial_snr[:, :t_done]
        cor_sub = trial_corr[:, :t_done]
        b_col = b_idx[:, None]
        ts_sub = np.where(tok_sub == 0, 1.0, -1.0)
        base_kn = alpha0[c_sub] + slope * seff_vec(snr_sub, sq_b[b_col, c_sub]) + ts_sub * tok_hd[c_sub]
        oh_sub = np.eye(10, dtype=float)[c_sub]

        for _ in range(2):
            d_bt = d_hat[b_col, c_sub]
            eta_k = np.clip(base_kn - slope * d_bt, -30.0, 30.0)
            uk = 1.0 / (1.0 + np.exp(-eta_k))
            pk = np.clip(gamma + span * uk, 1e-7, 1.0 - 1e-7)
            dp_dd = -slope * span * uk * (1.0 - uk)
            dL_dp = cor_sub / pk - (1.0 - cor_sub) / (1.0 - pk)
            d2L_dp2 = -cor_sub / (pk * pk) - (1.0 - cor_sub) / ((1.0 - pk) ** 2)
            g_bt = dL_dp * dp_dd
            w_bt = np.maximum(-d2L_dp2 * (dp_dd ** 2), 0.0)

            g_like = np.einsum("bn,bnc->bc", g_bt, oh_sub)
            w_diag = np.einsum("bn,bnc->bc", w_bt, oh_sub)
            grad = g_like - (d_hat - prior_mu_d) @ lam0
            H = np.broadcast_to(lam0, (b_tot, 10, 10)).copy()
            H[:, np.arange(10), np.arange(10)] += w_diag
            step = np.clip(np.linalg.solve(H, grad[:, :, None])[:, :, 0], -3.0, 3.0)
            d_hat = d_hat + step

        H_fin = np.broadcast_to(lam0, (b_tot, 10, 10)).copy()
        H_fin[:, np.arange(10), np.arange(10)] += w_diag
        post_cov = np.linalg.inv(H_fin)

    return trial_c, trial_tok, trial_snr, trial_corr


def interpolate_equiv_trials(n_grid: list[int], rmse_curve: list[float], target_rmse: float) -> float | None:
    if rmse_curve[0] <= target_rmse:
        return float(n_grid[0])
    if rmse_curve[-1] > target_rmse:
        return None
    for k in range(len(n_grid) - 1):
        r0, r1 = rmse_curve[k], rmse_curve[k + 1]
        if r0 >= target_rmse >= r1:
            frac = (r0 - target_rmse) / max(r0 - r1, 1e-12)
            return round(float(n_grid[k] + frac * (n_grid[k + 1] - n_grid[k])), 1)
    return None


def plot_suprathreshold_efficiency(
    trial_rows: list[dict[str, Any]],
    group_rows_100: list[dict[str, Any]],
    arm_meta: list[tuple[str, str, str, str]],
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15.6, 11.0), dpi=160)
    fig.suptitle(
        "Quantifying Suprathreshold Hearing Deficits: Single SNR vs. 5-SNR Sets vs. Adaptive Trial-by-Trial SNR Tracking\n"
        "(Calibrated to 98 Human Listeners: 32 Normal, 23 Slight, 26 Mild, 17 Moderate; 40 Sessions/Listener)",
        fontsize=12.6,
        fontweight="bold",
        y=0.985,
    )

    # Panel A: All-98 Global Suprathreshold Deficit D_i RMSE vs. N
    ax = axes[0, 0]
    for key, label, col, ls in arm_meta:
        sub = sorted([r for r in trial_rows if r["arm"] == key], key=lambda r: r["n_trials"])
        ax.plot([r["n_trials"] for r in sub], [r["global_D_rmse_db"] for r in sub], marker="o", ms=4.5, lw=2.0, color=col, ls=ls, label=label)
    cur100 = [r["global_D_rmse_db"] for r in trial_rows if r["arm"] == "single_current" and r["n_trials"] == 100][0]
    ax.axhline(cur100, color="#666666", ls=":", lw=1.2, label=f"Current qVCV @ 100 trials ({cur100:.2f} dB)")
    ax.set_xlabel("Number of Trials (N)")
    ax.set_ylabel("Global Suprathreshold Deficit RMSE (dB) ↓")
    ax.set_title("A. All-Cohort (n = 98) Suprathreshold Precision vs. Trial Budget", fontweight="bold", fontsize=10.6)
    ax.legend(loc="upper right", fontsize=7.3)
    ax.grid(alpha=0.25)

    # Panel B: Normal + Slight HL (PTA <= 25 dB HL, n = 55) "Hidden" Suprathreshold Loss Precision vs. N
    ax = axes[0, 1]
    for key, label, col, ls in arm_meta:
        sub = sorted([r for r in trial_rows if r["arm"] == key], key=lambda r: r["n_trials"])
        ax.plot([r["n_trials"] for r in sub], [r["nh_slight_D_rmse_db"] for r in sub], marker="s", ms=4.5, lw=2.0, color=col, ls=ls, label=label)
    cur_nh100 = [r["nh_slight_D_rmse_db"] for r in trial_rows if r["arm"] == "single_current" and r["n_trials"] == 100][0]
    ax.axhline(cur_nh100, color="#666666", ls=":", lw=1.2, label=f"Current qVCV @ 100 trials ({cur_nh100:.2f} dB)")
    ax.set_xlabel("Number of Trials (N)")
    ax.set_ylabel("Normal + Slight Suprathreshold RMSE (dB) ↓")
    ax.set_title("B. Near-Normal Audiogram (PTA ≤ 25 dB HL, n = 55) Efficiency", fontweight="bold", fontsize=10.6)
    ax.legend(loc="upper right", fontsize=7.3)
    ax.grid(alpha=0.25)

    # Panel C: Clinical Group & Worst-10% Breakdown at N = 100 Trials
    ax = axes[1, 0]
    g_map = {r["arm"]: r for r in group_rows_100}
    cats = ["Normal\n(n=32)", "Slight\n(n=23)", "Mild\n(n=26)", "Moderate\n(n=17)", "Worst 10%\nListeners"]
    cat_cols = ["rmse_normal_db", "rmse_slight_db", "rmse_mild_db", "rmse_moderate_db", "rmse_p90_worst_db"]
    show_arms = [
        ("single_current",       "Current 1-SNR (Ref)",         "#7f7f7f"),
        ("single_bayes_opt",     "Bayes-Opt 1-SNR",             "#1f77b4"),
        ("set5_balanced_span",   "Balanced 5-SNR Set",          "#2ca02c"),
        ("adaptive_5snr_hier",   "Adaptive SNR (5-SNR Bank)",   "#ff7f0e"),
        ("adaptive_1db_hier",    "Adaptive SNR (1-dB Blind)",   "#d62728"),
        ("adaptive_joint_audio", "Adaptive (Cons+SNR, Audio)",  "#8c564b"),
    ]
    xg = np.arange(len(cats))
    bw = 0.135
    for idx_a, (akey, alab, acol) in enumerate(show_arms):
        vals = [g_map[akey][ck] for ck in cat_cols]
        ax.bar(xg + (idx_a - 2.5) * bw, vals, bw, color=acol, label=alab)
    ax.set_xticks(xg)
    ax.set_xticklabels(cats, fontsize=8.8)
    ax.set_ylabel("Global Suprathreshold Deficit RMSE at N=100 (dB) ↓")
    ax.set_title("C. Clinical Group & Worst-Case Robustness (N = 100 Trials)", fontweight="bold", fontsize=10.6)
    ax.legend(loc="upper left", fontsize=7.4, ncol=2)
    ax.grid(axis="y", alpha=0.25)

    # Panel D: Per-Consonant Suprathreshold Profile D_{i,c} RMSE vs. N
    ax = axes[1, 1]
    for key, label, col, ls in arm_meta:
        sub = sorted([r for r in trial_rows if r["arm"] == key], key=lambda r: r["n_trials"])
        ax.plot([r["n_trials"] for r in sub], [r["per_cons_D_rmse_db"] for r in sub], marker="^", ms=4.5, lw=2.0, color=col, ls=ls, label=label)
    ax.set_xlabel("Number of Trials (N)")
    ax.set_ylabel("10-Consonant Suprathreshold Profile RMSE (dB) ↓")
    ax.set_title("D. Consonant-Specific Suprathreshold Profile Precision", fontweight="bold", fontsize=10.6)
    ax.legend(loc="upper right", fontsize=7.3)
    ax.grid(alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.945])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Suprathreshold efficiency study: 1-SNR vs 5-SNR vs Adaptive SNR.")
    parser.add_argument("--n_reps", type=int, default=40, help="Monte Carlo sessions per human listener")
    parser.add_argument("--out_dir", type=str, default=str(Path(__file__).resolve().parent / "human_calibrated_results"))
    args = parser.parse_args()

    t0 = time.time()
    sim_dir = Path(__file__).resolve().parent
    with (sim_dir / "human_data" / "human_listeners_98.json").open("r", encoding="utf-8") as f:
        data = json.load(f)

    hp = data["calibration_hyperparams"]
    slope = float(hp["slope_per_db"])
    ceil_p = float(hp["universal_ceil_test"])
    gamma = 1.0 / 11.0
    stims = data["stimuli"]
    ref_snr = np.array([float(data["ref_snr_db"][c]) for c in stims], dtype=float)
    alpha0 = np.array([float(hp["alpha_0_by_consonant"][c]) for c in stims], dtype=float)

    listeners = data["listeners"]
    n_subj = len(listeners)
    groups = np.array([r["archetype"] for r in listeners])
    nh_slight_mask = (groups == "Normal") | (groups == "Slight")
    alpha_true = np.array([r["multisnr_params"]["alpha_plomp_logit"] for r in listeners], dtype=float)
    snr_q_true = np.array([r["multisnr_params"]["snr_quiet_db"] for r in listeners], dtype=float)
    tok_hd = np.array(listeners[0]["multisnr_params"]["token_diag_logit_halfdiff"], dtype=float)

    D_ic_true = (alpha0[None, :] - alpha_true) / slope
    D_global_true = D_ic_true.mean(axis=1)

    prior_mu_D = float(D_global_true.mean())
    prior_sd_G = float(D_global_true.std())
    prior_sd_delta = float((D_ic_true - D_global_true[:, None]).std())

    bayes_opt_1snr = compute_bayes_optimal_single_snrs(
        alpha_true, snr_q_true, ref_snr, slope, gamma, ceil_p, prior_sd_G, prior_sd_delta
    )

    grid_single_cur = ref_snr[:, None]
    grid_single_bayes = bayes_opt_1snr[:, None]
    grid_set5_ref = ref_snr[:, None] + np.array([[-8.0, -4.0, 0.0, 4.0, 8.0]])
    grid_set5_span = ref_snr[:, None] + np.array([[-12.0, -7.0, -2.0, 3.0, 8.0]])
    # Wide 5-SNR adaptive bank (5 pre-baked WAVs per VCV covering crisp NH to Moderate HL):
    bank_5snr_wide = ref_snr[:, None] + np.array([[-14.0, -8.0, -2.0, 4.0, 10.0]])
    bank_1db = np.repeat(SNR_BANK_1DB[None, :], 10, axis=0)

    arms_cfg = [
        ("single_current",       "Current qVCV 1-SNR (5x @ Ref)",            "#7f7f7f", ":",  grid_single_cur,   None,           True,  False),
        ("single_bayes_opt",     "Bayes-Opt 1-SNR (5x)",                     "#1f77b4", "-.", grid_single_bayes, None,           True,  False),
        ("set5_around_ref",      "5-SNR Centered on Ref (Ref ±8 dB)",         "#9467bd", "--", grid_set5_ref,     None,           True,  False),
        ("set5_balanced_span",   "Balanced 5-SNR Set (Ref -12..+8 dB)",       "#2ca02c", "-",  grid_set5_span,    None,           True,  False),
        ("adaptive_1db_vcv",     "Adaptive SNR (Uncoupled Per-VCV, Blind)",   "#17becf", "--", None,              bank_1db,       False, False),
        ("adaptive_5snr_hier",   "Adaptive SNR (5-SNR Bank, Hier. Blind)",    "#ff7f0e", "-.", None,              bank_5snr_wide, True,  False),
        ("adaptive_1db_hier",    "Adaptive SNR (1-dB Bank, Hier. Blind)",     "#d62728", "-",  None,              bank_1db,       True,  False),
        ("adaptive_joint_audio", "Adaptive Cons+SNR (Audiogram-Informed)",    "#8c564b", "-",  None,              bank_1db,       True,  True),
    ]

    max_tr = max(EVAL_TRIALS)
    subj_of_b = np.tile(np.arange(n_subj), args.n_reps)
    sq_b_known = snr_q_true[subj_of_b]

    trial_rows: list[dict[str, Any]] = []
    group_rows_100: list[dict[str, Any]] = []

    for a_idx, (a_key, a_lab, _, _, s_grid, a_bank, hier, joint_aud) in enumerate(arms_cfg):
        print(f"[{a_idx+1}/{len(arms_cfg)}] Simulating {a_lab} ({args.n_reps} reps x 98 listeners)...", flush=True)
        tc, ttok, tsnr, tcor = simulate_arm_trials(
            arm_name=a_key,
            n_reps=args.n_reps,
            max_trials=max_tr,
            seed=20269000 + a_idx * 100,
            alpha_true=alpha_true,
            snr_q_true=snr_q_true,
            alpha0=alpha0,
            tok_hd=tok_hd,
            slope=slope,
            gamma=gamma,
            ceil_p=ceil_p,
            prior_mu_d=prior_mu_D,
            prior_sd_g=prior_sd_G,
            prior_sd_d=prior_sd_delta,
            static_grid_10xK=s_grid,
            adaptive_bank_10xM=a_bank,
            hierarchical_coupling=hier,
            audiogram_informed_joint=joint_aud,
        )

        # Informative error count per 100 trials on Normal listeners (for confusion profiling!):
        cor_reshaped_100 = tcor.reshape(args.n_reps, n_subj, max_tr)
        mean_errors_normal_100 = float(np.mean(100.0 - np.sum(cor_reshaped_100[:, groups == "Normal", :], axis=2)))
        mean_errors_all98_100 = float(np.mean(100.0 - np.sum(cor_reshaped_100, axis=2)))

        for nt in EVAL_TRIALS:
            d_est = decode_hierarchical_suprathreshold_map(
                tc[:, :nt], ttok[:, :nt], tsnr[:, :nt], tcor[:, :nt],
                sq_b_10=sq_b_known,
                alpha0=alpha0,
                tok_hd=tok_hd,
                slope=slope,
                gamma=gamma,
                ceil_p=ceil_p,
                prior_mu_g=prior_mu_D,
                prior_sd_g=prior_sd_G,
                prior_sd_delta=prior_sd_delta,
            ).reshape(args.n_reps, n_subj, 10)

            d_glob_est = d_est.mean(axis=2)
            err_glob = d_glob_est - D_global_true[None, :]
            err_ic = d_est - D_ic_true[None, :, :]

            rmse_glob = float(np.sqrt(np.mean(err_glob ** 2)))
            rmse_nh_sl = float(np.sqrt(np.mean(err_glob[:, nh_slight_mask] ** 2)))
            mae_glob = float(np.mean(np.abs(err_glob)))
            rmse_ic = float(np.sqrt(np.mean(err_ic ** 2)))

            half_r = args.n_reps // 2
            retest_mad = float(np.mean(np.abs(d_glob_est[:half_r] - d_glob_est[half_r : 2 * half_r])))
            r_vals = [float(np.corrcoef(d_glob_est[r], D_global_true)[0, 1]) for r in range(args.n_reps)]
            r2_true = float(np.mean(np.array(r_vals) ** 2))

            trial_rows.append({
                "arm": a_key,
                "description": a_lab,
                "n_trials": nt,
                "global_D_rmse_db": round(rmse_glob, 3),
                "nh_slight_D_rmse_db": round(rmse_nh_sl, 3),
                "global_D_mae_db": round(mae_glob, 3),
                "global_D_r2": round(r2_true, 4),
                "global_D_retest_mad_db": round(retest_mad, 3),
                "per_cons_D_rmse_db": round(rmse_ic, 3),
            })

            if nt == 100:
                per_listener_rmse = np.sqrt(np.mean(err_glob ** 2, axis=0))
                group_rows_100.append({
                    "arm": a_key,
                    "description": a_lab,
                    "rmse_all98_db": round(rmse_glob, 3),
                    "rmse_nh_slight55_db": round(rmse_nh_sl, 3),
                    "rmse_normal_db": round(float(np.sqrt(np.mean(err_glob[:, groups == "Normal"] ** 2))), 3),
                    "rmse_slight_db": round(float(np.sqrt(np.mean(err_glob[:, groups == "Slight"] ** 2))), 3),
                    "rmse_mild_db": round(float(np.sqrt(np.mean(err_glob[:, groups == "Mild"] ** 2))), 3),
                    "rmse_moderate_db": round(float(np.sqrt(np.mean(err_glob[:, groups == "Moderate"] ** 2))), 3),
                    "rmse_p90_worst_db": round(float(np.percentile(per_listener_rmse, 90.0)), 3),
                    "retest_mad_db": round(retest_mad, 3),
                    "r2_global_D": round(r2_true, 4),
                    "per_cons_D_rmse_db": round(rmse_ic, 3),
                    "mean_errors_normal_100": round(mean_errors_normal_100, 1),
                    "mean_errors_all98_100": round(mean_errors_all98_100, 1),
                })

    cur_100_glob = [r["global_D_rmse_db"] for r in trial_rows if r["arm"] == "single_current" and r["n_trials"] == 100][0]
    cur_100_nhsl = [r["nh_slight_D_rmse_db"] for r in trial_rows if r["arm"] == "single_current" and r["n_trials"] == 100][0]
    cur_100_cons = [r["per_cons_D_rmse_db"] for r in trial_rows if r["arm"] == "single_current" and r["n_trials"] == 100][0]

    efficiency_table: list[dict[str, Any]] = []
    for a_key, *_ in arms_cfg:
        sub = sorted([r for r in trial_rows if r["arm"] == a_key], key=lambda r: r["n_trials"])
        g100 = [r for r in group_rows_100 if r["arm"] == a_key][0]
        neq_glob = interpolate_equiv_trials(EVAL_TRIALS, [r["global_D_rmse_db"] for r in sub], cur_100_glob)
        neq_nhsl = interpolate_equiv_trials(EVAL_TRIALS, [r["nh_slight_D_rmse_db"] for r in sub], cur_100_nhsl)
        neq_cons = interpolate_equiv_trials(EVAL_TRIALS, [r["per_cons_D_rmse_db"] for r in sub], cur_100_cons)
        efficiency_table.append({
            **g100,
            "n_eq_all98_global_D": neq_glob,
            "speedup_all98_global_D": round(100.0 / neq_glob, 2) if neq_glob is not None else None,
            "n_eq_nh_slight_D": neq_nhsl,
            "speedup_nh_slight_D": round(100.0 / neq_nhsl, 2) if neq_nhsl is not None else None,
            "n_eq_per_cons_D": neq_cons,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    from qvcv_tracking_sim import write_dict_rows_csv
    write_dict_rows_csv(trial_rows, out_dir / "suprathreshold_efficiency_by_trials.csv")
    write_dict_rows_csv(efficiency_table, out_dir / "suprathreshold_group_breakdown_100trials.csv")

    arm_meta = [(k, lab, col, ls) for k, lab, col, ls, *_ in arms_cfg]
    fig_path = out_dir / "fig_suprathreshold_efficiency.png"
    plot_suprathreshold_efficiency(trial_rows, group_rows_100, arm_meta, fig_path)

    summary = {
        "elapsed_sec": round(time.time() - t0, 2),
        "n_reps": args.n_reps,
        "current_ref_snrs_db": {stims[c]: int(ref_snr[c]) for c in range(10)},
        "bayes_optimal_single_snrs_db": {stims[c]: int(bayes_opt_1snr[c]) for c in range(10)},
        "balanced_5snr_grid_db": {stims[c]: [int(x) for x in grid_set5_span[c]] for c in range(10)},
        "adaptive_wide_5snr_bank_db": {stims[c]: [int(x) for x in bank_5snr_wide[c]] for c in range(10)},
        "efficiency_summary_at_100_trials": efficiency_table,
    }
    with (out_dir / "suprathreshold_efficiency_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
