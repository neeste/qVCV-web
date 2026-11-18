#!/usr/bin/env python3
"""
export_human_listeners.py

Calibrates qVCV virtual listeners (both fixed-SNR and continuous multi-SNR)
directly to the empirical dataset of 98 human listeners (32 NH + 66 HI across
Normal, Slight, Mild, and Moderate groups; 2 unaided sessions x 100 trials =
19,600 trials, plus 60-dB-SNR Practice block scores and 10-frequency audiograms)
in ``~/STN/2026_Quick_Tests/qtest_dataset``.

Physical Multi-SNR Calibration (Joint Empirical-Bayes Plomp Band-Masking Model):
  - Directly measures the 0-dB-SNR LTASS speech-shaped noise ERB band masking
    levels M0_c (dB HL) from the qVCV ``Stimuli/`` .wav waveforms across each
    consonant's primary acoustic/audiometric cue band.
  - Combines external band noise (M0_c - SNR) with listener i's audiometric
    band threshold T_{i,c} via Plomp (1978) / SII power-sum masking, and fits
    joint Empirical-Bayes MAP supra-threshold coding intercepts alpha_{i,c}
    across all 98 listeners against both Test (Ref SNR) and Practice (+60 dB SNR).

Exports self-contained calibration files to ``qVCV/sim/human_data/``:
  - human_listeners_98.csv      (compatible with qvcv_simulator.py --ground_truth)
  - human_listeners_98.json     (full fixed-SNR + multi-SNR + token parameters)
  - mnr_vcv_pta_model.json      (exact clinical PPTA regression B, C, ca, pa)
  - calibration_validation.json (human vs. simulated validation benchmarks)
"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, wavfile
from scipy.optimize import minimize

from qvcv_simulator import (
    CONSONANT_SNR_DB,
    DISPLAY_CONSONANTS,
    N_RESPONSES,
    N_STIMULI,
    STIMULI_CONSONANTS,
    VirtualListener,
    normalize_rows,
    save_cohort_csv,
)

DEFAULT_DATASET_DIR = Path("/Users/neely/STN/2026_Quick_Tests/qtest_dataset")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "human_data"
STIMULI_WAV_DIR = Path(__file__).resolve().parent.parent / "Stimuli"

UCNS_LOWER = [c.lower() for c in STIMULI_CONSONANTS]
URSP_LOWER = [c.lower() for c in STIMULI_CONSONANTS] + ["other"]
REF_SNR_ARR = np.array([CONSONANT_SNR_DB[c] for c in STIMULI_CONSONANTS], dtype=float)
PRACTICE_SNR_DB = 60.0

AUD_FREQS_HZ = np.array([250, 500, 750, 1000, 1500, 2000, 3000, 4000, 6000, 8000], dtype=float)
ANSI_RETSPL_DB = np.array([18.0, 11.0, 8.5, 7.5, 7.5, 9.0, 10.0, 9.5, 15.5, 13.0], dtype=float)
# Fallback measured 0-dB-SNR LTASS ERB band masking levels (dB HL) from Stimuli/*.wav:
FALLBACK_LTASS_ERB_HL = np.array(
    [28.83, 37.62, 37.42, 37.53, 35.12, 31.71, 28.80, 28.03, 20.30, 21.30],
    dtype=float,
)

# Consonant-specific primary acoustic & empirical audiogram cue bands (indices into AUD_FREQS_HZ):
CONSONANT_CUE_BAND_IDX: dict[str, list[int]] = {
    "B":  [1, 2, 3],       # 0.5, 0.75, 1.0 kHz (labial burst & low F2 locus)
    "D":  [5, 6, 7],       # 2.0, 3.0, 4.0 kHz (alveolar burst & F2/F3 transition)
    "G":  [3, 4, 5],       # 1.0, 1.5, 2.0 kHz (velar F2-F3 pinch in /a_a/)
    "K":  [4, 7, 8],       # 1.5, 4.0, 6.0 kHz (velar burst + HF aspiration)
    "N":  [0, 1, 3, 4],    # 0.25, 0.5, 1.0, 1.5 kHz (nasal murmur + F2 transition)
    "S":  [7, 8, 9],       # 4.0, 6.0, 8.0 kHz (alveolar sibilant frication)
    "SH": [5, 6, 7],       # 2.0, 3.0, 4.0 kHz (postalveolar sibilant frication)
    "T":  [7, 8, 9],       # 4.0, 6.0, 8.0 kHz (alveolar burst & HF aspiration)
    "V":  [2, 4, 5, 7],    # 0.75, 1.5, 2.0, 4.0 kHz (voiced labiodental frication)
    "Z":  [6, 7, 8, 9],    # 3.0, 4.0, 6.0, 8.0 kHz (voiced alveolar sibilant)
}


def measure_ltass_0snr_erb_hl(stim_dir: Path = STIMULI_WAV_DIR) -> np.ndarray:
    """
    Measure the exact 0-dB-SNR speech-shaped noise ERB band level (in dB HL)
    across the 10 audiometric frequencies by subtracting 60-SNR clear speech
    waveforms from 0-SNR test waveforms (where calibration RMS=0.01 == 60 dB SPL).
    """
    if not (stim_dir / "practice").exists() or not (stim_dir / "test").exists():
        return FALLBACK_LTASS_ERB_HL.copy()
    band_spls: list[list[float]] = []
    for c in ("k", "s", "sh", "t", "z"):
        for tok in (1, 2):
            p_path = stim_dir / "practice" / f"VCV_a{c}a_{tok}_60SNR.wav"
            t_path = stim_dir / "test" / f"VCV_a{c}a_{tok}_0SNR.wav"
            if not p_path.exists() or not t_path.exists():
                continue
            sr, p_dat = wavfile.read(str(p_path))
            _, t_dat = wavfile.read(str(t_path))
            scale = float(np.iinfo(p_dat.dtype).max)
            n_arr = (t_dat.astype(float) - p_dat.astype(float)) / scale
            freqs = np.fft.rfftfreq(len(n_arr), 1.0 / float(sr))
            psd = (np.abs(np.fft.rfft(n_arr)) ** 2) / (len(n_arr) ** 2) * 2.0
            b_spl = []
            for f0 in AUD_FREQS_HZ:
                erb = 24.7 * (4.37 * (f0 / 1000.0) + 1.0)
                m = (freqs >= f0 - 0.5 * erb) & (freqs <= f0 + 0.5 * erb)
                pwr = max(float(np.sum(psd[m])), 1e-20)
                b_spl.append(60.0 + 10.0 * np.log10(pwr / (0.01 ** 2)))
            band_spls.append(b_spl)
    if not band_spls:
        return FALLBACK_LTASS_ERB_HL.copy()
    return np.mean(band_spls, axis=0) - ANSI_RETSPL_DB


def effective_audible_snr(snr_db: np.ndarray | float, snr_quiet_db: np.ndarray | float) -> np.ndarray:
    """
    Power-sum masking model (Plomp 1978 / SII): combines external speech-shaped
    noise at nominal ``snr_db`` with internal threshold limit ``snr_quiet_db``
    when speech is presented at a fixed 60 dB SPL:
      SNR_eff = -10 * log10( 10^(-snr_db/10) + 10^(-snr_quiet_db/10) )
    """
    ln10_10 = np.log(10.0) / 10.0
    a = -np.asarray(snr_db, dtype=float) * ln10_10
    b = -np.asarray(snr_quiet_db, dtype=float) * ln10_10
    m = np.maximum(a, b)
    lse = m + np.log(np.exp(a - m) + np.exp(b - m))
    return -10.0 * lse / np.log(10.0)


def _read_xlsx_sheet(path: Path, sheet_index: int) -> list[list[str]]:
    """Dependency-free .xlsx worksheet reader using standard zipfile + ElementTree."""
    ns = {"main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(path, "r") as z:
        strings: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            sst = ET.fromstring(z.read("xl/sharedStrings.xml"))
            for si in sst.findall("main:si", ns):
                strings.append("".join(t.text or "" for t in si.findall(".//main:t", ns)))
        ws = ET.fromstring(z.read(f"xl/worksheets/sheet{sheet_index}.xml"))

        def col_to_idx(cell_ref: str) -> int:
            letters = "".join(ch for ch in cell_ref if ch.isalpha())
            idx = 0
            for ch in letters:
                idx = idx * 26 + (ord(ch) - ord("A") + 1)
            return idx - 1

        max_col = 0
        parsed: list[dict[int, str]] = []
        for row in ws.findall(".//main:row", ns):
            row_dict: dict[int, str] = {}
            for c in row.findall("main:c", ns):
                idx = col_to_idx(c.attrib["r"])
                max_col = max(max_col, idx + 1)
                t = c.attrib.get("t")
                v_el = c.find("main:v", ns)
                if v_el is None or v_el.text is None:
                    val = ""
                elif t == "s":
                    val = strings[int(v_el.text)]
                else:
                    val = v_el.text
                row_dict[idx] = val
            parsed.append(row_dict)
        return [[rd.get(i, "") for i in range(max_col)] for rd in parsed]


def predict_ppta_mnr(
    ccms_n_10_11: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    ca: np.ndarray,
    pa: np.ndarray | None = None,
) -> np.ndarray | float:
    """
    Exact Python port of Summarize.m / Residual_Loss.m ``predict_pta`` +
    ``audibility_adjustment`` (verified to <1e-14 dB against mnr_vcv_pta.mat).
    """
    arr = np.asarray(ccms_n_10_11, dtype=float)
    single = arr.ndim == 2
    if single:
        arr = arr[None, :, :]
    if arr.shape[2] == 10:
        arr = np.concatenate([arr, np.zeros((arr.shape[0], 10, 1), dtype=float)], axis=2)

    n_subj = arr.shape[0]
    totals = np.maximum(arr.sum(axis=(1, 2), keepdims=True), 1e-12)
    normed = arr * (100.0 / totals)
    X = normed.reshape(n_subj, 110, order="F")
    Z = np.column_stack([np.ones(n_subj, dtype=float), X @ C]) @ B
    P = 1.0 / (1.0 + np.exp(-np.clip(Z, -40.0, 40.0)))
    phat = np.ones((n_subj, 1), dtype=float)
    for k in range(B.shape[1]):
        pk = P[:, k : k + 1]
        phat = np.hstack([pk, (1.0 - pk) * phat])
    raw_a = (phat @ np.asarray(ca, dtype=float).reshape(-1, 1)).ravel()
    if pa is not None:
        raw_a = raw_a + float(pa[1]) - float(pa[0])
    return float(raw_a[0]) if single else raw_a


def compute_aroc(scores_neg: np.ndarray, scores_pos: np.ndarray) -> float:
    neg = np.asarray(scores_neg, dtype=float)[:, None]
    pos = np.asarray(scores_pos, dtype=float)[None, :]
    return float(
        (np.sum(pos - neg > 1e-12) + 0.5 * np.sum(np.abs(pos - neg) <= 1e-12))
        / (neg.shape[0] * pos.shape[1])
    )


def loocv_hellinger_aroc_unbiased(
    counts_n_k_d: np.ndarray,
    neg_mask: np.ndarray,
    pos_mask: np.ndarray,
) -> float:
    """
    Exact sample-size-unbiased Leave-One-Out Hellinger Mann-Whitney A_ROC:
    For every pair (i in neg, j in pos), both listener i and listener j are
    scored against the exact same out-of-sample negative reference distribution
    T_{-i} (mean of neg \\ {i}), guaranteeing zero in-sample leakage and exact
    0.5 tie-handling on identical confusion rows (with alpha = 0).

    Accepts shape (N, D) for a single VCV, (N, K, D) for a K-row matrix, or
    (M, N, K, D) for M Monte Carlo draws (returning mean A_ROC across M).
    """
    arr = np.asarray(counts_n_k_d, dtype=float)
    if arr.ndim == 2:
        arr = arr[None, :, None, :]
    elif arr.ndim == 3:
        arr = arr[None, :, :, :]

    obs = arr / np.maximum(arr.sum(axis=-1, keepdims=True), 1e-12)
    sqrt_obs = np.sqrt(obs)

    neg_idx = np.where(neg_mask)[0]
    pos_idx = np.where(pos_mask)[0]
    n_neg = len(neg_idx)
    n_pos = len(pos_idx)

    neg_obs = obs[:, neg_idx, :, :]
    neg_sum = neg_obs.sum(axis=1)
    sqrt_neg = sqrt_obs[:, neg_idx, :, :]
    sqrt_pos = sqrt_obs[:, pos_idx, :, :]

    u_total = np.zeros(arr.shape[0], dtype=float)
    for ii in range(n_neg):
        t_minus_i = np.maximum((neg_sum - neg_obs[:, ii, :, :]) / float(n_neg - 1), 0.0)
        st = np.sqrt(t_minus_i)
        h_i = np.mean(np.sqrt(0.5 * np.sum((sqrt_neg[:, ii, :, :] - st) ** 2, axis=-1)), axis=-1)
        h_pos = np.mean(np.sqrt(0.5 * np.sum((sqrt_pos - st[:, None, :, :]) ** 2, axis=-1)), axis=-1)
        diff = h_pos - h_i[:, None]
        u_total += np.sum(diff > 1e-12, axis=1) + 0.5 * np.sum(np.abs(diff) <= 1e-12, axis=1)

    return float(np.mean(u_total / float(n_neg * n_pos)))


def loocv_hellinger_scores(
    counts_n_10_11: np.ndarray,
    neg_mask: np.ndarray,
    pos_mask: np.ndarray,
    alpha: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-listener LOOCV Hellinger distance scores (normalized per listener before smoothing)."""
    arr = np.asarray(counts_n_10_11, dtype=float)
    row_props = arr / np.maximum(arr.sum(axis=-1, keepdims=True), 1e-12)
    n = arr.shape[0]
    d_neg = np.zeros(n, dtype=float)
    d_diff = np.zeros(n, dtype=float)
    idx_all = np.arange(n)
    for i in range(n):
        neg_idx = idx_all[neg_mask & (idx_all != i)]
        pos_idx = idx_all[pos_mask & (idx_all != i)]
        t_neg = row_props[neg_idx].mean(axis=0) + alpha
        t_neg /= t_neg.sum(axis=1, keepdims=True)
        t_pos = row_props[pos_idx].mean(axis=0) + alpha
        t_pos /= t_pos.sum(axis=1, keepdims=True)
        obs = row_props[i] + alpha
        obs /= obs.sum(axis=1, keepdims=True)
        h_neg = float(np.mean(np.sqrt(0.5 * np.sum((np.sqrt(obs) - np.sqrt(t_neg)) ** 2, axis=1))))
        h_pos = float(np.mean(np.sqrt(0.5 * np.sum((np.sqrt(obs) - np.sqrt(t_pos)) ** 2, axis=1))))
        d_neg[i] = h_neg
        d_diff[i] = h_neg - h_pos
    return d_neg, d_diff


def _fit_joint_plomp_map(
    k_obs_98_10: np.ndarray,
    m0_c: np.ndarray,
    t_ic_98_10: np.ndarray,
    delta_m: float,
    beta_aud: float,
    slope_db: float,
    beta_dist: float,
    ceil_test: float = 0.996,
    sigma_alpha: float = 1.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Joint Empirical-Bayes MAP calibration across all 98 listeners and 10 consonants:
      SNR_quiet(i, c) = (M0_c + delta_m) - beta_aud * T_{i,c}
      eta(i, c, SNR)  = alpha_{i,c} + slope_db * SNR_eff(SNR, SNR_quiet(i, c))
      alpha_{i,c} ~ N(alpha_0[c] - beta_dist * max(0, T_{i,c} - 15), sigma_alpha^2)
    """
    gamma = 1.0 / float(N_RESPONSES)
    span = ceil_test - gamma
    snr_q = (m0_c[None, :] + delta_m) - beta_aud * t_ic_98_10
    seff_ref = effective_audible_snr(REF_SNR_ARR[None, :], snr_q)
    dist_term = beta_dist * np.maximum(0.0, t_ic_98_10 - 15.0)
    inv_var = 1.0 / (sigma_alpha ** 2)

    alpha_0 = -slope_db * (REF_SNR_ARR - 8.0)
    a = alpha_0[None, :] - dist_term
    n_subj = float(k_obs_98_10.shape[0])

    for _ in range(8):
        mu_alpha = alpha_0[None, :] - dist_term
        for _ in range(10):
            eta = np.clip(a + slope_db * seff_ref, -30.0, 30.0)
            u = 1.0 / (1.0 + np.exp(-eta))
            p = np.clip(gamma + span * u, 1e-7, 1.0 - 1e-7)
            dp_deta = span * u * (1.0 - u)
            d2p_deta2 = dp_deta * (1.0 - 2.0 * u)
            dL_dp = k_obs_98_10 / p - (20.0 - k_obs_98_10) / (1.0 - p)
            d2L_dp2 = -k_obs_98_10 / (p * p) - (20.0 - k_obs_98_10) / ((1.0 - p) ** 2)
            grad = dL_dp * dp_deta - (a - mu_alpha) * inv_var
            hess = d2L_dp2 * (dp_deta ** 2) + dL_dp * d2p_deta2 - inv_var
            a = a - np.clip(grad / np.minimum(hess, -1e-6), -2.5, 2.5)
        alpha_0 = np.sum((a + dist_term) * inv_var, axis=0) / (n_subj * inv_var + 1.0 / 25.0)

    return snr_q, alpha_0, a


def export_calibrated_cohort(
    dataset_dir: Path = DEFAULT_DATASET_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = dataset_dir / "Data" / "QuickTests_DATA.xlsx"
    aud_rows = _read_xlsx_sheet(xlsx_path, 1)
    qvcv_rows = _read_xlsx_sheet(xlsx_path, 3)

    meta: dict[str, dict[str, Any]] = {}
    for r in aud_rows[1:]:
        if not r or not r[0].strip():
            continue
        sid = r[0].strip()
        meta[sid] = {
            "age": float(r[1]) if (len(r) > 1 and r[1] != "") else None,
            "sex": r[2].strip() if len(r) > 2 else "",
            "ear": r[3].strip() if len(r) > 3 else "",
            "audiogram_khz": [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0],
            "audiogram_db": [float(r[c]) if (len(r) > c and r[c] != "") else 0.0 for c in range(4, 14)],
            "pta_124_db": float(r[14]) if (len(r) > 14 and r[14] != "") else 0.0,
            "group": r[15].strip().title() if len(r) > 15 else "Unknown",
        }
    for r in qvcv_rows[1:]:
        if not r or not r[0].strip():
            continue
        sid = r[0].strip()
        if sid in meta:
            meta[sid]["practice_pc"] = float(r[1]) if (len(r) > 1 and r[1] != "") else None

    m_comb = loadmat(str(dataset_dir / "qVCV_analysis" / "QVC_Combined.mat"), squeeze_me=True, struct_as_record=False)
    m_perf = loadmat(str(dataset_dir / "qVCV_analysis" / "QVC_perf.mat"), squeeze_me=True)
    m_mnr = loadmat(str(dataset_dir / "qVCV_analysis" / "mnr_vcv_pta.mat"), squeeze_me=True)

    ids_98 = [str(x).strip() for x in m_comb["ID"]]
    t_comb = m_comb["T_Combined"]
    c_map = {c: idx for idx, c in enumerate(UCNS_LOWER)}
    r_map = {r: idx for idx, r in enumerate(URSP_LOWER)}

    ccms_sess = np.zeros((len(ids_98), 2, N_STIMULI, N_RESPONSES), dtype=float)
    ccms_tok = np.zeros((len(ids_98), 2, N_STIMULI, 2, N_RESPONSES), dtype=float)
    for pk in range(len(ids_98)):
        for sk in range(2):
            row_idx = sk + pk * 2
            for tk in range(t_comb.shape[1]):
                tr = t_comb[row_idx, tk]
                ci = c_map[str(tr.consonant).strip().lower()]
                ri = r_map[str(tr.response).strip().lower()]
                tni = int(tr.token_number) - 1
                ccms_sess[pk, sk, ci, ri] += 1.0
                ccms_tok[pk, sk, ci, tni, ri] += 1.0

    pooled_200 = ccms_sess.sum(axis=1)
    pooled_tok = ccms_tok.sum(axis=1)
    k_obs = np.diagonal(pooled_200[:, :, :N_STIMULI], axis1=1, axis2=2)  # (98, 10)

    B = np.asarray(m_mnr["B"], dtype=float)
    C = np.asarray(m_mnr["C"], dtype=float)
    ca = np.asarray(m_mnr["ca"], dtype=float).ravel()
    pa = np.asarray(m_mnr["pa"], dtype=float).ravel()
    rhl_98 = np.asarray(m_mnr["rhl"], dtype=float).ravel()
    ptawt = np.asarray(m_perf["ptawt"], dtype=float).ravel()

    # Population token-specific log-odds offset for _1_ vs _2_ recordings
    pop_tok_counts = pooled_tok.sum(axis=0)
    tok_diag_logit_halfdiff = np.zeros(N_STIMULI, dtype=float)
    for ci in range(N_STIMULI):
        p_t0 = (pop_tok_counts[ci, 0, ci] + 1.0) / (pop_tok_counts[ci, 0].sum() + 2.0)
        p_t1 = (pop_tok_counts[ci, 1, ci] + 1.0) / (pop_tok_counts[ci, 1].sum() + 2.0)
        lo0 = np.log(p_t0 / (1.0 - p_t0))
        lo1 = np.log(p_t1 / (1.0 - p_t1))
        tok_diag_logit_halfdiff[ci] = 0.5 * (lo0 - lo1)

    # Empirical off-diagonal error distribution per listener & consonant (strictly group-blind)
    pop_err_profile = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
    for ci in range(N_STIMULI):
        ec = pooled_200[:, ci, :].sum(axis=0).copy()
        ec[ci] = 0.0
        ec = ec + 0.10
        ec[ci] = 0.0
        pop_err_profile[ci] = ec / ec.sum()

    indiv_err_profile = np.zeros((len(ids_98), N_STIMULI, N_RESPONSES), dtype=float)
    raw_emp_offdiag = np.zeros((len(ids_98), N_STIMULI, N_RESPONSES), dtype=float)
    for i in range(len(ids_98)):
        # Group-blind LOO off-diagonal prior for listeners with 0 empirical errors on consonant ci:
        others_ec = pooled_200[[j for j in range(len(ids_98)) if j != i]].sum(axis=0)
        for ci in range(N_STIMULI):
            obs_e = pooled_200[i, ci, :].copy()
            obs_e[ci] = 0.0
            loo_e = others_ec[ci].copy()
            loo_e[ci] = 0.0
            loo_e /= max(float(loo_e.sum()), 1e-12)
            s_obs = float(obs_e.sum())
            if s_obs > 0.0:
                raw_od = obs_e + 0.25 * loo_e
                raw_od[ci] = 0.0
                raw_emp_offdiag[i, ci] = raw_od / raw_od.sum()
            else:
                raw_emp_offdiag[i, ci] = loo_e
            prof = obs_e + 2.0 * pop_err_profile[ci]
            prof[ci] = 0.0
            indiv_err_profile[i, ci] = prof / prof.sum()

    marginal_resp = pooled_200.sum(axis=(0, 1))
    marginal_resp = marginal_resp / marginal_resp.sum()

    # Measure physical 0-dB-SNR LTASS ERB band masking levels (dB HL) per consonant cue band
    m0_freq_hl = measure_ltass_0snr_erb_hl()
    m0_c = np.array([float(m0_freq_hl[CONSONANT_CUE_BAND_IDX[c]].mean()) for c in STIMULI_CONSONANTS], dtype=float)

    groups_arr = np.array([meta[sid]["group"] for sid in ids_98])
    aud_mat = np.array([meta[sid]["audiogram_db"] for sid in ids_98], dtype=float)  # (98, 10)
    t_ic = np.column_stack([aud_mat[:, CONSONANT_CUE_BAND_IDX[c]].mean(axis=1) for c in STIMULI_CONSONANTS])

    prac_pc = np.array(
        [meta[sid]["practice_pc"] if meta[sid]["practice_pc"] is not None else np.nan for sid in ids_98],
        dtype=float,
    )
    valid_prac = ~np.isnan(prac_pc)
    hum_test_pc = 0.5 * (
        np.trace(ccms_sess[:, 0, :, :N_STIMULI], axis1=1, axis2=2)
        + np.trace(ccms_sess[:, 1, :, :N_STIMULI], axis1=1, axis2=2)
    )

    gamma = 1.0 / float(N_RESPONSES)
    ceil_test = 0.996

    def _eval_plomp_hyper(params: np.ndarray) -> float:
        delta_m, beta_aud, slope_db, beta_dist, prac_lapse = params
        snr_q, _, alpha_map = _fit_joint_plomp_map(
            k_obs, m0_c, t_ic, delta_m, beta_aud, slope_db, beta_dist, ceil_test=ceil_test
        )
        seff_60 = effective_audible_snr(PRACTICE_SNR_DB, snr_q)
        u60 = 1.0 / (1.0 + np.exp(-np.clip(alpha_map + slope_db * seff_60, -35.0, 35.0)))
        p60 = (1.0 - prac_lapse) * (gamma + (ceil_test - gamma) * u60) + prac_lapse * gamma
        pr_prac = 10.0 * np.sum(p60, axis=1)

        seff_ref = effective_audible_snr(REF_SNR_ARR[None, :], snr_q)
        uref = 1.0 / (1.0 + np.exp(-np.clip(alpha_map + slope_db * seff_ref, -35.0, 35.0)))
        pr_test = 10.0 * np.sum(gamma + (ceil_test - gamma) * uref, axis=1)

        loss = 0.20 * float(np.mean((pr_prac[valid_prac] - prac_pc[valid_prac]) ** 2))
        loss += 0.25 * float(np.mean((pr_test - hum_test_pc) ** 2))
        for g in ("Normal", "Slight", "Mild", "Moderate"):
            mp = (groups_arr == g) & valid_prac
            mt = groups_arr == g
            if np.any(mp):
                loss += 15.0 * float((pr_prac[mp].mean() - prac_pc[mp].mean()) ** 2)
            if np.any(mt):
                loss += 15.0 * float((pr_test[mt].mean() - hum_test_pc[mt].mean()) ** 2)
        return loss

    fit_res = minimize(
        _eval_plomp_hyper,
        x0=np.array([6.0, 0.85, 0.38, 0.03, 0.022]),
        bounds=[(-6.0, 12.0), (0.55, 1.20), (0.24, 0.48), (0.0, 0.12), (0.005, 0.045)],
    )
    delta_m, beta_aud, slope_db, beta_dist, prac_lapse = [float(x) for x in fit_res.x]
    snr_quiet_mat, alpha_0_vec, alpha_map = _fit_joint_plomp_map(
        k_obs, m0_c, t_ic, delta_m, beta_aud, slope_db, beta_dist, ceil_test=ceil_test
    )
    seff_ref_mat = effective_audible_snr(REF_SNR_ARR[None, :], snr_quiet_mat)
    eta0_mat = alpha_map + slope_db * seff_ref_mat  # (98, 10)

    # Construct calibrated reference confusion matrices p_ref_cal consistent with eta0_mat
    p_ref_cal = np.zeros((len(ids_98), N_STIMULI, N_RESPONSES), dtype=float)
    for i in range(len(ids_98)):
        for ci in range(N_STIMULI):
            u0 = 1.0 / (1.0 + np.exp(-np.clip(eta0_mat[i, ci], -35.0, 35.0)))
            p_corr = float(np.clip(gamma + (ceil_test - gamma) * u0, 1e-6, 1.0 - 1e-6))
            p_ref_cal[i, ci] = (1.0 - p_corr) * raw_emp_offdiag[i, ci]
            p_ref_cal[i, ci, ci] = p_corr
    p_ref_cal = normalize_rows(p_ref_cal.reshape(-1, N_RESPONSES)).reshape(len(ids_98), N_STIMULI, N_RESPONSES)

    session_logit_sd = 0.33
    listener_records: list[dict[str, Any]] = []
    vlisteners: list[VirtualListener] = []

    for i, sid in enumerate(ids_98):
        m_info = meta[sid]
        aud_vec = np.asarray(m_info["audiogram_db"], dtype=float)
        pta_w = float(aud_vec @ (ptawt / ptawt.sum()))

        ceil_vec = np.full(N_STIMULI, ceil_test, dtype=float)
        eta0_vec = eta0_mat[i].copy()
        tok_mats = np.zeros((2, N_STIMULI, N_RESPONSES), dtype=float)
        for ci in range(N_STIMULI):
            for t_idx, t_sign in ((0, 1.0), (1, -1.0)):
                u_t = 1.0 / (1.0 + np.exp(-np.clip(eta0_vec[ci] + t_sign * tok_diag_logit_halfdiff[ci], -35.0, 35.0)))
                p_t = float(np.clip(gamma + (ceil_test - gamma) * u_t, 1e-6, 1.0 - 1e-6))
                off = p_ref_cal[i, ci].copy()
                off[ci] = 0.0
                off /= max(float(off.sum()), 1e-12)
                row_t = (1.0 - p_t) * off
                row_t[ci] = p_t
                tok_mats[t_idx, ci] = row_t / row_t.sum()
            mean_pair = 0.5 * (tok_mats[0, ci] + tok_mats[1, ci])
            delta_pair = tok_mats[0, ci] - mean_pair
            max_scale = 1.0
            for ri in range(N_RESPONSES):
                if abs(delta_pair[ri]) > 1e-12:
                    max_scale = min(
                        max_scale,
                        p_ref_cal[i, ci, ri] / abs(delta_pair[ri]),
                        (1.0 - p_ref_cal[i, ci, ri]) / abs(delta_pair[ri]),
                    )
            delta_pair *= 0.98 * max_scale
            tok_mats[0, ci] = p_ref_cal[i, ci] + delta_pair
            tok_mats[1, ci] = p_ref_cal[i, ci] - delta_pair

        s1_pc = float(np.trace(ccms_sess[i, 0, :, :N_STIMULI]))
        s2_pc = float(np.trace(ccms_sess[i, 1, :, :N_STIMULI]))
        s1_ppta = float(predict_ppta_mnr(ccms_sess[i, 0], B, C, ca, pa))
        s2_ppta = float(predict_ppta_mnr(ccms_sess[i, 1], B, C, ca, pa))
        pool_ppta = float(predict_ppta_mnr(pooled_200[i], B, C, ca, pa))

        desc = (
            f"Human {sid} ({m_info['group']}, PTA124={m_info['pta_124_db']:.1f} dB HL, "
            f"PC={0.5*(s1_pc+s2_pc):.1f}%)"
        )
        vlisteners.append(
            VirtualListener(
                listener_id=sid,
                gt_matrix=p_ref_cal[i],
                archetype=m_info["group"],
                description=desc,
                token_matrices=tok_mats,
            )
        )
        listener_records.append({
            "listener_id": sid,
            "archetype": m_info["group"],
            "is_nh": bool(sid.startswith("NH")),
            "description": desc,
            "age": m_info["age"],
            "sex": m_info["sex"],
            "ear": m_info["ear"],
            "pta_124_db": round(float(m_info["pta_124_db"]), 2),
            "pta_weighted_db": round(pta_w, 2),
            "rhl_db": round(float(rhl_98[i]), 2),
            "audiogram_db": [round(float(x), 1) for x in m_info["audiogram_db"]],
            "cue_band_thresh_db": np.round(t_ic[i], 2).tolist(),
            "practice_pc": m_info["practice_pc"],
            "human_s1_pc": s1_pc,
            "human_s2_pc": s2_pc,
            "human_s1_ppta_db": round(s1_ppta, 3),
            "human_s2_ppta_db": round(s2_ppta, 3),
            "human_pooled_ppta_db": round(pool_ppta, 3),
            "empirical_s1_counts": ccms_sess[i, 0].astype(int).tolist(),
            "empirical_s2_counts": ccms_sess[i, 1].astype(int).tolist(),
            "empirical_pooled_counts": pooled_200[i].astype(int).tolist(),
            "confusion_matrix": np.round(p_ref_cal[i], 6).tolist(),
            "multisnr_params": {
                "slope_per_db": round(slope_db, 5),
                "snr_quiet_db": np.round(snr_quiet_mat[i], 4).tolist(),
                "alpha_plomp_logit": np.round(alpha_map[i], 5).tolist(),
                "ceil_prob": np.round(ceil_vec, 6).tolist(),
                "eta0_logit": np.round(eta0_vec, 6).tolist(),
                "error_profile": np.round(indiv_err_profile[i], 6).tolist(),
                "token_diag_logit_halfdiff": np.round(tok_diag_logit_halfdiff, 5).tolist(),
                "marginal_low_snr_profile": np.round(marginal_resp, 6).tolist(),
                "session_logit_sd": session_logit_sd,
            },
        })

    csv_path = out_dir / "human_listeners_98.csv"
    save_cohort_csv(vlisteners, csv_path)

    json_path = out_dir / "human_listeners_98.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_dataset": str(dataset_dir),
                "n_listeners": len(listener_records),
                "n_nh": int(sum(1 for r in listener_records if r["is_nh"])),
                "n_hl": int(sum(1 for r in listener_records if not r["is_nh"])),
                "stimuli": STIMULI_CONSONANTS,
                "responses": DISPLAY_CONSONANTS,
                "ref_snr_db": CONSONANT_SNR_DB,
                "practice_snr_db": PRACTICE_SNR_DB,
                "ltass_0snr_erb_band_hl_by_consonant": {
                    c: round(float(m0_c[idx]), 2) for idx, c in enumerate(STIMULI_CONSONANTS)
                },
                "calibration_hyperparams": {
                    "model": "joint_empirical_bayes_plomp_band_masking",
                    "ltass_delta_m_db": round(delta_m, 4),
                    "beta_audiogram": round(beta_aud, 4),
                    "slope_per_db": round(slope_db, 4),
                    "beta_supra_distortion": round(beta_dist, 4),
                    "practice_familiarization_lapse": round(prac_lapse, 4),
                    "universal_ceil_test": ceil_test,
                    "alpha_0_by_consonant": {
                        c: round(float(alpha_0_vec[idx]), 4) for idx, c in enumerate(STIMULI_CONSONANTS)
                    },
                    "session_logit_sd": session_logit_sd,
                },
                "listeners": listener_records,
            },
            f,
            indent=2,
        )

    mnr_path = out_dir / "mnr_vcv_pta_model.json"
    with mnr_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "description": "Multinomial logistic regression predicting audiometric PTA (dB HL) from 10x11 qVCV CCM",
                "stimuli_order": STIMULI_CONSONANTS,
                "response_order": DISPLAY_CONSONANTS,
                "B": B.tolist(),
                "C": C.tolist(),
                "ca": ca.tolist(),
                "pa": pa.tolist(),
                "ptawt": ptawt.tolist(),
            },
            f,
            indent=2,
        )

    # Quantitative Human-vs-Simulated Calibration Validation Report
    is_nh = np.array([r["is_nh"] for r in listener_records], dtype=bool)
    is_slight = np.array([r["archetype"] == "Slight" for r in listener_records], dtype=bool)
    is_mild = np.array([r["archetype"] == "Mild" for r in listener_records], dtype=bool)
    is_mod = np.array([r["archetype"] == "Moderate" for r in listener_records], dtype=bool)

    hum_s1_pc = np.array([r["human_s1_pc"] for r in listener_records])
    hum_s2_pc = np.array([r["human_s2_pc"] for r in listener_records])
    hum_s1_pp = np.array([r["human_s1_ppta_db"] for r in listener_records])
    hum_s2_pp = np.array([r["human_s2_ppta_db"] for r in listener_records])

    hum_aroc_all = 0.5 * (
        loocv_hellinger_aroc_unbiased(ccms_sess[:, 0], is_nh, ~is_nh)
        + loocv_hellinger_aroc_unbiased(ccms_sess[:, 1], is_nh, ~is_nh)
    )
    hum_aroc_sl = 0.5 * (
        loocv_hellinger_aroc_unbiased(ccms_sess[:, 0], is_nh, is_slight)
        + loocv_hellinger_aroc_unbiased(ccms_sess[:, 1], is_nh, is_slight)
    )
    hum_aroc_mm = 0.5 * (
        loocv_hellinger_aroc_unbiased(ccms_sess[:, 0], is_nh, is_mild | is_mod)
        + loocv_hellinger_aroc_unbiased(ccms_sess[:, 1], is_nh, is_mild | is_mod)
    )

    rng = np.random.default_rng(20261115)
    sim_r_pc, sim_mad_pc, sim_r_pp, sim_mad_pp = [], [], [], []
    sim_aroc_all, sim_aroc_slight, sim_aroc_mm = [], [], []
    for _ in range(50):
        sim_sess = np.zeros((len(ids_98), 2, N_STIMULI, N_RESPONSES), dtype=float)
        for sk in range(2):
            z_sess = rng.normal(0.0, session_logit_sd, size=len(ids_98))
            log_p = np.log(np.clip(p_ref_cal, 1e-9, 1.0))
            for ci in range(N_STIMULI):
                log_p[:, ci, ci] += z_sess
            p_sk = np.exp(log_p - log_p.max(axis=2, keepdims=True))
            p_sk /= p_sk.sum(axis=2, keepdims=True)
            for i in range(len(ids_98)):
                for ci in range(N_STIMULI):
                    sim_sess[i, sk, ci] = rng.multinomial(10, p_sk[i, ci])
        sc1 = np.trace(sim_sess[:, 0, :, :N_STIMULI], axis1=1, axis2=2)
        sc2 = np.trace(sim_sess[:, 1, :, :N_STIMULI], axis1=1, axis2=2)
        pp1 = np.asarray(predict_ppta_mnr(sim_sess[:, 0], B, C, ca, pa), dtype=float)
        pp2 = np.asarray(predict_ppta_mnr(sim_sess[:, 1], B, C, ca, pa), dtype=float)
        sim_r_pc.append(float(np.corrcoef(sc1, sc2)[0, 1]))
        sim_mad_pc.append(float(np.mean(np.abs(sc1 - sc2))))
        sim_r_pp.append(float(np.corrcoef(pp1, pp2)[0, 1]))
        sim_mad_pp.append(float(np.mean(np.abs(pp1 - pp2))))

        sim_aroc_all.append(loocv_hellinger_aroc_unbiased(sim_sess[:, 0], is_nh, ~is_nh))
        sim_aroc_slight.append(loocv_hellinger_aroc_unbiased(sim_sess[:, 0], is_nh, is_slight))
        sim_aroc_mm.append(loocv_hellinger_aroc_unbiased(sim_sess[:, 0], is_nh, is_mild | is_mod))

    seff_60_all = effective_audible_snr(PRACTICE_SNR_DB, snr_quiet_mat)
    u60_all = 1.0 / (1.0 + np.exp(-np.clip(alpha_map + slope_db * seff_60_all, -35.0, 35.0)))
    pred_prac_arr = 10.0 * np.sum(
        (1.0 - prac_lapse) * (gamma + (ceil_test - gamma) * u60_all) + prac_lapse * gamma,
        axis=1,
    )

    group_validation: dict[str, Any] = {}
    for g in ("Normal", "Slight", "Mild", "Moderate"):
        m = np.array([r["archetype"] == g for r in listener_records], dtype=bool)
        hum_test = 0.5 * (hum_s1_pc[m] + hum_s2_pc[m])
        sim_test = 10.0 * np.trace(p_ref_cal[m, :, :N_STIMULI], axis1=1, axis2=2)
        group_validation[g] = {
            "n": int(m.sum()),
            "human_test_pc_mean": round(float(hum_test.mean()), 2),
            "sim_test_pc_mean": round(float(sim_test.mean()), 2),
            "human_test_pc_sd": round(float(hum_test.std()), 2),
            "sim_test_pc_sd": round(float(sim_test.std()), 2),
            "human_practice_pc_mean": round(float(prac_pc[m & valid_prac].mean()), 2),
            "sim_practice_pc_mean": round(float(pred_prac_arr[m & valid_prac].mean()), 2),
        }

    val_report = {
        "n_listeners": len(listener_records),
        "repeatability_100_trials": {
            "human_pc_r": round(float(np.corrcoef(hum_s1_pc, hum_s2_pc)[0, 1]), 4),
            "sim_pc_r": round(float(np.mean(sim_r_pc)), 4),
            "human_pc_mad_pct": round(float(np.mean(np.abs(hum_s1_pc - hum_s2_pc))), 2),
            "sim_pc_mad_pct": round(float(np.mean(sim_mad_pc)), 2),
            "human_ppta_r": round(float(np.corrcoef(hum_s1_pp, hum_s2_pp)[0, 1]), 4),
            "sim_ppta_r": round(float(np.mean(sim_r_pp)), 4),
            "human_ppta_mad_db": round(float(np.mean(np.abs(hum_s1_pp - hum_s2_pp))), 2),
            "sim_ppta_mad_db": round(float(np.mean(sim_mad_pp)), 2),
        },
        "loocv_hellinger_aroc_100_trials": {
            "human_nh_vs_all_hl": round(hum_aroc_all, 4),
            "sim_nh_vs_all_hl": round(float(np.mean(sim_aroc_all)), 4),
            "human_nh_vs_slight_hl": round(hum_aroc_sl, 4),
            "sim_nh_vs_slight_hl": round(float(np.mean(sim_aroc_slight)), 4),
            "human_nh_vs_mild_mod_hl": round(hum_aroc_mm, 4),
            "sim_nh_vs_mild_mod_hl": round(float(np.mean(sim_aroc_mm)), 4),
        },
        "group_means_test_and_practice": group_validation,
    }
    val_path = out_dir / "calibration_validation.json"
    with val_path.open("w", encoding="utf-8") as f:
        json.dump(val_report, f, indent=2)

    return val_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and calibrate 98 human qVCV listeners.")
    parser.add_argument("--dataset_dir", type=str, default=str(DEFAULT_DATASET_DIR))
    parser.add_argument("--out_dir", type=str, default=str(DEFAULT_OUT_DIR))
    args = parser.parse_args()

    report = export_calibrated_cohort(Path(args.dataset_dir), Path(args.out_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
