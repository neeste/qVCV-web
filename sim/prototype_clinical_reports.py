#!/usr/bin/env python3
"""
prototype_clinical_reports.py

Automated Suprathreshold Hearing-Deficit Clinical Interpretation Engine &
1-Page Clinical Report Prototype combining:
  1. Pure-Tone Audiogram (0.25 - 8 kHz, dB HL + BTNRH RETSPL -> dB SPL)
  2. qCLS Categorical Loudness Scaling (CU 5..50 Equal-Loudness Contours, 9 or 10 bands)
  3. qVCV Closed-Set Consonant-in-Noise Test (100/200 unaided trials + optional Aided verification)

Generates:
  - Individual 1-page clinical reports (PNG + printable PDF) for 6 archetypal
    real human listeners from ``qtest_dataset``:
      1. NH013 : Intact Normal Reference (PTA = 8.3 dB HL, concordant Audiogram/qCLS/qVCV)
      2. HL045 : "Hidden" Suprathreshold Deficit with Normal Audiogram (all thresholds <= 20 dB HL)
      3. HL011 : Mild HF Notch (25-35 dB HL at 2-8 kHz) + Suprathreshold Coding Deficit
      4. HL068 : Peripheral OHC Recruitment + Audibility Loss, Intact Neural Coding
                 (Strong Aided Responder: 64% -> 88% Lab Aid / 85% Personal Aid)
      5. HL030 : Ski-Slope OHC Recruitment + Severe Neural Coding Deficit
                 (Limited Aided Benefit: 42% -> 46% Lab Aid; Remote-Mic Candidate)
      6. HL048 : Mild-Moderate HF Sloping Loss with Aided SIN Rollover
                 (78.5% Unaided -> 68.0% Lab Aid; WDRC / Upward-Spread Over-Masking)
  - 6-Case Clinical Atlas Overview figure: ``clinical_reports/fig_clinical_prototype_overview.png``
  - Full 98-listener normative clinical table: ``clinical_reports/cohort_98_clinical_interpretation.csv``
  - Normative cutoffs & case summaries JSON: ``clinical_reports/clinical_prototype_summary.json``

Also supports standalone clinical CSV files exported by the qCLS and qVCV web apps:
  python3 sim/prototype_clinical_reports.py \
      --qvcv_csv path/to/qVCV_Subj_L_20261115.csv \
      --qcls_elc_csv path/to/qCLS_ELC_Subj_L_20261115.csv \
      --audiogram 15,15,20,25,30,35,45,55,60,65
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

os.environ.setdefault("MPLCONFIGDIR", "/tmp/qvcv_mplconfig")
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from characterize_human_listeners import (
    HF_CONS_IDX,
    LF_CONS_IDX,
    fit_single_listener_biophysical,
)
from export_human_listeners import predict_ppta_mnr
from qvcv_biophysical_model import (
    ANSI_RETSPL_DB,
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
    normalize_rows,
    relative_information_transmitted,
)
from qvcv_tracking_sim import write_dict_rows_csv
from qvcv_trackers import estimate_eb_concentration

FREQ_LABELS = ["250", "500", "750", "1k", "1.5k", "2k", "3k", "4k", "6k", "8k"]
CU_LEVELS = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

# Exact BTNRH RETSPL tables from qCLS/qcls_core.c (250..8000 Hz) + ANSI fallback:
RETSPL_TABLES: dict[str, np.ndarray] = {
    "er3a": np.array([17.5, 9.5, 6.0, 5.5, 9.5, 11.5, 13.0, 15.0, 16.0, 15.5], dtype=float),
    "legacy": np.array([30.0, 19.0, 12.0, 10.0, 9.0, 15.0, 15.5, 13.0, 13.0, 14.0], dtype=float),
    "ansi": ANSI_RETSPL_DB.copy(),
}

# Clinical Palette (matching UW AHS Lab + Boys Town Navy/Cream/Gold theme)
NAVY = "#14284B"
GOLD = "#D4A017"
CREAM = "#FAF8F4"
PANEL_BG = "#FFFFFF"
SLATE = "#334155"
MUTED = "#64748B"
GREEN_OK = "#15803D"
AMBER_WARN = "#D97706"
CORAL_OHC = "#EA580C"
BLUE_AUD = "#1D4ED8"
RED_NEURAL = "#B91C1C"
TEAL_AID = "#0F766E"


@dataclass
class ClinicalNorms:
    """Empirical normative reference anchored to the 32 Normal-Hearing human listeners."""
    nh_ref_listener: BiophysicalVirtualListener
    ds_mono: np.ndarray
    pc_mono: np.ndarray
    pc_nh0: float
    y_nh_elc_10x10: np.ndarray          # (10_freq, 10_cu) NH mean ELC in dB SPL
    y_nh_cu5_sd: np.ndarray             # (10_freq,) NH SD on CU 5
    y_nh_cu50_sd: np.ndarray            # (10_freq,) NH SD on CU 50
    nh_rtr_voicing: float               # NH mean Voicing information transmitted (0..1)
    nh_rtr_manner: float                # NH mean Manner information transmitted (0..1)
    nh_rtr_place: float                 # NH mean Place information transmitted (0..1)
    # 90th & 95th percentile cutoffs from 64 single-session (100-trial) NH runs:
    p90_total_db: float
    p95_total_db: float
    p90_aud_db: float
    p90_ohc_db: float
    p90_neural_db: float
    p95_neural_db: float
    nh_pc_p05: float
    nh_pc_median: float

    def eq_snr_deficit_db(self, pc_val: float) -> float:
        """Map recognition accuracy onto NH-equivalent SNR loss (dB SNR relative to NH median)."""
        if pc_val >= self.pc_nh0:
            return -2.2 * (pc_val - self.pc_nh0) / max(0.990 - self.pc_nh0, 1e-4)
        pc_c = float(np.clip(pc_val, self.pc_mono[0] + 1e-4, self.pc_nh0))
        ds_eq = float(np.interp(pc_c, self.pc_mono, self.ds_mono))
        return -ds_eq


def load_raw_qcls_elc_matrix_map(qcls_mat_path: Path = DEFAULT_QCLS_MAT) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Return per-listener (10_freq, 10_cu) ELC matrix in dB SPL + NH mean & SDs."""
    qc = loadmat(str(qcls_mat_path), squeeze_me=True)
    q_ids = [str(x["name"]).strip() for x in qc["id"]]
    elcb = qc["elcb"]
    tcnd = qc["tcnd"]
    nnh = int(qc["nnh"])

    nh_mats = []
    for i in range(nnh):
        kk = tcnd[i] == 1
        if np.sum(kk) > 0:
            m = np.nanmean(elcb[i, kk, :, :], axis=0)
            if not np.all(np.isnan(m)) and np.nanmean(m[:, 9] - m[:, 0]) > 20.0:
                nh_mats.append(m)
    nh_stack = np.stack(nh_mats, axis=0)
    y_nh = np.nanmean(nh_stack, axis=0)
    cu5_sd = np.nanstd(nh_stack[:, :, 0], axis=0)
    cu50_sd = np.nanstd(nh_stack[:, :, 9], axis=0)

    elc_map: dict[str, np.ndarray] = {}
    for idx, qid in enumerate(q_ids):
        kk = tcnd[idx] == 1
        if np.sum(kk) > 0:
            yi = np.nanmean(elcb[idx, kk, :, :], axis=0)
            if not np.all(np.isnan(yi)):
                elc_map[qid] = yi
    return elc_map, y_nh, cu5_sd, cu50_sd


def parse_qcls_elc_csv(
    elc_csv_path: Path,
    audiogram_hl_10: np.ndarray,
    y_nh_10x10: np.ndarray,
) -> tuple[np.ndarray, QCLSCochlearProfile]:
    """
    Parse a web-app exported ``qCLS_ELC_*.csv`` (9 bands 250..6000 Hz or 10 bands 250..8000 Hz)
    into a (10, 10) ELC SPL array (with NaN at 8 kHz if 9-band) and QCLSCochlearProfile.
    """
    yi = np.full((10, 10), np.nan, dtype=float)
    with elc_csv_path.open("r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            freq = float(row.get("Freq_Hz") or row.get("Freq") or 0.0)
            cu = int(round(float(row.get("CU") or 0)))
            lvl = float(row.get("Level_dB_SPL") or row.get("Level") or np.nan)
            f_idx_arr = np.where(np.isclose(AUD_FREQS_HZ, freq, rtol=0.03))[0]
            if len(f_idx_arr) > 0 and cu in CU_LEVELS:
                yi[int(f_idx_arr[0]), CU_LEVELS.index(cu)] = lvl

    tot = np.maximum(audiogram_hl_10, 0.0)
    loss_low = np.nanmean(yi[:, 0:2] - y_nh_10x10[:, 0:2], axis=1)
    loss_high = np.nanmean(yi[:, 7:9] - y_nh_10x10[:, 7:9], axis=1)
    rec = np.clip(loss_low - np.maximum(loss_high, 0.0), 0.0, 55.0)
    # If 8 kHz was not measured by 9-band qCLS, carry 6 kHz OHC recruitment fraction onto 8 kHz audiogram:
    if np.isnan(rec[9]) and not np.isnan(rec[8]) and tot[8] > 5.0:
        frac_6k = float(np.clip(rec[8] / tot[8], 0.25, 0.85))
        rec[9] = np.minimum(frac_6k * tot[9], 55.0)
    rec = np.where(np.isnan(rec), np.minimum(0.70 * tot, 55.0), rec)
    ohc = np.where(tot >= 10.0, np.clip(rec, 0.25 * tot, np.minimum(tot, 55.0)), 0.70 * tot)
    ohc = np.clip(ohc, 0.0, 55.0)
    ihc = np.maximum(tot - ohc, 0.0)

    dy_nh = y_nh_10x10[:, 5] - y_nh_10x10[:, 2]
    dy_i = np.maximum(yi[:, 5] - yi[:, 2], 8.0)
    sl = np.where(np.isnan(dy_i), 1.0 + 0.025 * ohc, dy_nh / dy_i)
    prof = QCLSCochlearProfile(
        listener_id=elc_csv_path.stem,
        has_measured_qcls=True,
        audiogram_hl_db=audiogram_hl_10,
        hl_ohc_db=ohc,
        hl_ihc_db=ihc,
        cls_speech_slope=np.clip(sl, 0.7, 3.5),
    )
    return yi, prof


def build_clinical_norms(
    bank: SpectroTemporalVCVBank,
    base_listeners: list[dict[str, Any]],
    bio_data: dict[str, Any],
    qcls_map: dict[str, QCLSCochlearProfile],
    y_nh: np.ndarray,
    cu5_sd: np.ndarray,
    cu50_sd: np.ndarray,
) -> tuple[ClinicalNorms, BiophysicalHyperparams, np.ndarray]:
    hp_d = bio_data["biophysical_hyperparams"]
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
    self_off = np.array(bio_data["self_clarity_offset_nh"], dtype=float)
    bio_by_id = {r["listener_id"]: r for r in bio_data["listeners"]}

    # Anchor NH reference listener to empirical median of the 32 human NH listeners:
    nh_auds = np.array([qcls_map[r["listener_id"]].audiogram_hl_db for r in base_listeners if r["is_nh"]])
    nh_ohcs = np.array([qcls_map[r["listener_id"]].hl_ohc_db for r in base_listeners if r["is_nh"]])
    nh_unks = np.array([bio_by_id[r["listener_id"]]["unk_criterion"] for r in base_listeners if r["is_nh"]])

    med_aud = np.median(nh_auds, axis=0)
    med_ohc = np.median(nh_ohcs, axis=0)
    med_unk = float(np.median(nh_unks))
    nh_prof = QCLSCochlearProfile(
        listener_id="NH_MEDIAN_REF",
        has_measured_qcls=True,
        audiogram_hl_db=med_aud,
        hl_ohc_db=med_ohc,
        hl_ihc_db=np.maximum(med_aud - med_ohc, 0.0),
        cls_speech_slope=np.ones(10, dtype=float),
    )
    nh_ref = BiophysicalVirtualListener(
        "NH_MEDIAN_REF", "Normal", "Empirical NH Median Reference", nh_prof, bank, hp, np.ones(10), self_off, med_unk
    )

    ds_grid = np.linspace(-35.0, 4.0, 391)
    pc_raw = np.array([float(np.mean(np.diag(nh_ref.matrix_at_snrs(REF_SNR_ARR + ds)[:, :10]))) for ds in ds_grid])
    idx_pk = int(np.argmax(pc_raw))
    ds_mono = ds_grid[: idx_pk + 1]
    pc_mono = np.maximum.accumulate(pc_raw[: idx_pk + 1])
    pc_nh0 = float(np.interp(0.0, ds_mono, pc_mono))

    nh_rtrs = []
    for r in base_listeners:
        if r["is_nh"]:
            cp = np.array(r["empirical_s1_counts"], dtype=float) + np.array(r["empirical_s2_counts"], dtype=float)
            rt = relative_information_transmitted(normalize_rows(cp))
            nh_rtrs.append([rt["voicing"], rt["manner"], rt["place"]])
    nh_rtr_mean = np.mean(nh_rtrs, axis=0)

    # Provisional norms object to evaluate 64 single-session NH runs:
    tmp_norms = ClinicalNorms(
        nh_ref_listener=nh_ref,
        ds_mono=ds_mono,
        pc_mono=pc_mono,
        pc_nh0=pc_nh0,
        y_nh_elc_10x10=y_nh,
        y_nh_cu5_sd=cu5_sd,
        y_nh_cu50_sd=cu50_sd,
        nh_rtr_voicing=float(nh_rtr_mean[0]),
        nh_rtr_manner=float(nh_rtr_mean[1]),
        nh_rtr_place=float(nh_rtr_mean[2]),
        p90_total_db=2.5,
        p95_total_db=3.5,
        p90_aud_db=1.0,
        p90_ohc_db=0.8,
        p90_neural_db=2.0,
        p95_neural_db=3.0,
        nh_pc_p05=90.0,
        nh_pc_median=98.0,
    )

    nh_tot, nh_aud, nh_ohc, nh_neu, nh_pcs = [], [], [], [], []
    for r in base_listeners:
        if not r["is_nh"]:
            continue
        sid = r["listener_id"]
        prof = qcls_map[sid]
        nd, ac, _ = compute_peripheral_template_features(
            bank, REF_SNR_ARR, prof.audiogram_hl_db[None, :], prof.hl_ohc_db[None, :], hp
        )
        narrow = QCLSCochlearProfile(
            sid, prof.has_measured_qcls, prof.audiogram_hl_db, np.zeros(10), np.maximum(prof.audiogram_hl_db, 0), np.ones(10)
        )
        l_aud = BiophysicalVirtualListener(sid, "Normal", "", narrow, bank, hp, np.ones(10), self_off, med_unk)
        l_per = BiophysicalVirtualListener(sid, "Normal", "", prof, bank, hp, np.ones(10), self_off, med_unk)
        d_aud = max(0.0, tmp_norms.eq_snr_deficit_db(float(np.mean(np.diag(l_aud.matrix_at_snrs(REF_SNR_ARR)[:, :10])))))
        d_per = max(d_aud, tmp_norms.eq_snr_deficit_db(float(np.mean(np.diag(l_per.matrix_at_snrs(REF_SNR_ARR)[:, :10])))))
        for s_key in ("empirical_s1_counts", "empirical_s2_counts"):
            c_mat = np.array(r[s_key], dtype=float)
            eff_s, unk_s, _ = fit_single_listener_biophysical(
                c_mat, nd, ac, self_off, hp.directional_choice_bias, hp.lapse_rate, False
            )
            l_f = BiophysicalVirtualListener(sid, "Normal", "", prof, bank, hp, eff_s, self_off, unk_s)
            d_t = tmp_norms.eq_snr_deficit_db(float(np.mean(np.diag(l_f.matrix_at_snrs(REF_SNR_ARR)[:, :10]))))
            nh_tot.append(d_t)
            nh_aud.append(d_aud)
            nh_ohc.append(d_per - d_aud)
            nh_neu.append(d_t - d_per)
            nh_pcs.append(100.0 * np.trace(c_mat[:, :10]) / c_mat.sum())

    norms = ClinicalNorms(
        nh_ref_listener=nh_ref,
        ds_mono=ds_mono,
        pc_mono=pc_mono,
        pc_nh0=pc_nh0,
        y_nh_elc_10x10=y_nh,
        y_nh_cu5_sd=cu5_sd,
        y_nh_cu50_sd=cu50_sd,
        nh_rtr_voicing=float(nh_rtr_mean[0]),
        nh_rtr_manner=float(nh_rtr_mean[1]),
        nh_rtr_place=float(nh_rtr_mean[2]),
        p90_total_db=float(np.percentile(nh_tot, 90)),
        p95_total_db=float(np.percentile(nh_tot, 95)),
        p90_aud_db=max(float(np.percentile(nh_aud, 90)), 1.0),
        p90_ohc_db=max(float(np.percentile(nh_ohc, 90)), 0.75),
        p90_neural_db=float(np.percentile(nh_neu, 90)),
        p95_neural_db=float(np.percentile(nh_neu, 95)),
        nh_pc_p05=float(np.percentile(nh_pcs, 5)),
        nh_pc_median=float(np.median(nh_pcs)),
    )
    return norms, hp, self_off


def evaluate_listener_clinical_profile(
    listener_id: str,
    group_label: str,
    prof: QCLSCochlearProfile,
    elc_10x10: Optional[np.ndarray],
    counts_s1: np.ndarray,
    counts_s2: Optional[np.ndarray],
    practice_pc: Optional[float],
    aided_sessions: dict[str, np.ndarray],
    bank: SpectroTemporalVCVBank,
    hp: BiophysicalHyperparams,
    self_off: np.ndarray,
    norms: ClinicalNorms,
    mnr_params: Optional[dict[str, np.ndarray]] = None,
    n_boot_se: int = 24,
    retspl_10: Optional[np.ndarray] = None,
    ear: str = "Unspecified",
    measured_srt_loss_db: Optional[float] = None,
    contra_summary: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Run complete automated qPRO clinical interpretation on one patient ear."""
    use_retspl = RETSPL_TABLES["er3a"] if retspl_10 is None else np.asarray(retspl_10, dtype=float)
    ear_norm = ear.strip().capitalize() if ear else "Unspecified"
    if ear_norm.startswith("R"):
        ear_norm = "Right"
    elif ear_norm.startswith("L"):
        ear_norm = "Left"
    elif ear_norm.startswith("B"):
        ear_norm = "Both"
    else:
        ear_norm = "Unspecified"

    counts_tot = counts_s1.copy() if counts_s2 is None else (counts_s1 + counts_s2)
    n_trials = int(round(float(counts_tot.sum())))
    emp_pc_s1 = 100.0 * float(np.trace(counts_s1[:, :10]) / max(counts_s1.sum(), 1.0))
    emp_pc_s2 = (
        100.0 * float(np.trace(counts_s2[:, :10]) / max(counts_s2.sum(), 1.0))
        if counts_s2 is not None and counts_s2.sum() > 0
        else None
    )
    emp_pc = 100.0 * float(np.trace(counts_tot[:, :10]) / max(counts_tot.sum(), 1.0))
    unk_pct = 100.0 * float(counts_tot[:, 10].sum() / max(counts_tot.sum(), 1.0))

    med_unk_nh = norms.nh_ref_listener.unk_criterion
    nd, ac, _ = compute_peripheral_template_features(
        bank, REF_SNR_ARR, prof.audiogram_hl_db[None, :], prof.hl_ohc_db[None, :], hp
    )
    nd_prac, ac_prac, _ = compute_peripheral_template_features(
        bank, np.full(N_STIMULI, 60.0), prof.audiogram_hl_db[None, :], prof.hl_ohc_db[None, :], hp
    )

    # Fit biophysical neural coding parameters (use 3-param MAP for 100 trials, + regularized nuisance if >=180):
    use_nuis = n_trials >= 180
    eff_hat, unk_hat, mu_bio = fit_single_listener_biophysical(
        counts_tot,
        nd,
        ac,
        self_off,
        hp.directional_choice_bias,
        lapse_rate=hp.lapse_rate,
        fit_per_consonant_nuisance=use_nuis,
        practice_score_pct=practice_pc,
        nd_prac_1_10_2_10=nd_prac if practice_pc is not None else None,
        ac_prac_1_10_2=ac_prac if practice_pc is not None else None,
    )

    kappa = estimate_eb_concentration(counts_tot, mu_bio)
    smoothed_ccm = normalize_rows(counts_tot + kappa * mu_bio)
    rtr = relative_information_transmitted(smoothed_ccm)

    # 3-stage biophysical waterfall & NH-equivalent SNR deficit (dB SNR):
    narrow_prof = QCLSCochlearProfile(
        listener_id,
        prof.has_measured_qcls,
        prof.audiogram_hl_db,
        np.zeros(10, dtype=float),
        np.maximum(prof.audiogram_hl_db, 0.0),
        np.ones(10, dtype=float),
    )
    l_aud = BiophysicalVirtualListener(
        listener_id, group_label, "aud_only", narrow_prof, bank, hp, np.ones(10), self_off, med_unk_nh
    )
    l_per = BiophysicalVirtualListener(
        listener_id, group_label, "periph", prof, bank, hp, np.ones(10), self_off, med_unk_nh
    )
    l_ful = BiophysicalVirtualListener(
        listener_id, group_label, "full", prof, bank, hp, eff_hat, self_off, unk_hat
    )

    pc_aud = float(np.mean(np.diag(l_aud.matrix_at_snrs(REF_SNR_ARR)[:, :10])))
    pc_per = float(np.mean(np.diag(l_per.matrix_at_snrs(REF_SNR_ARR)[:, :10])))
    pc_ful = float(np.mean(np.diag(l_ful.matrix_at_snrs(REF_SNR_ARR)[:, :10])))
    pc_quiet_pred = float(np.mean(np.diag(l_ful.matrix_at_snrs(np.full(10, 60.0))[:, :10])))

    d_aud = max(0.0, norms.eq_snr_deficit_db(pc_aud))
    d_per = max(d_aud, norms.eq_snr_deficit_db(pc_per))
    d_tot = float(measured_srt_loss_db) if measured_srt_loss_db is not None else norms.eq_snr_deficit_db(pc_ful)
    d_ohc = d_per - d_aud
    d_neu = d_tot - d_per

    # Fast parametric bootstrap SE for behavioral & neural deficit at the observed trial count:
    rng_b = np.random.default_rng(abs(hash(listener_id)) % (2**31 - 1))
    n_per_row = np.maximum(np.round(counts_tot.sum(axis=1)).astype(int), 1)
    boot_tot, boot_neu = [], []
    p_boot_src = l_ful.matrix_at_snrs(REF_SNR_ARR)
    for _ in range(n_boot_se):
        c_sim = np.zeros_like(counts_tot)
        for c_i in range(N_STIMULI):
            c_sim[c_i] = rng_b.multinomial(n_per_row[c_i], p_boot_src[c_i])
        eff_b, unk_b, _ = fit_single_listener_biophysical(
            c_sim, nd, ac, self_off, hp.directional_choice_bias, lapse_rate=hp.lapse_rate, fit_per_consonant_nuisance=False
        )
        l_b = BiophysicalVirtualListener("B", "B", "", prof, bank, hp, eff_b, self_off, unk_b)
        dt_b = norms.eq_snr_deficit_db(float(np.mean(np.diag(l_b.matrix_at_snrs(REF_SNR_ARR)[:, :10]))))
        boot_tot.append(dt_b)
        boot_neu.append(dt_b - d_per)
    se_tot = float(np.std(boot_tot, ddof=1)) if n_boot_se > 1 else 0.7
    se_neu = float(np.std(boot_neu, ddof=1)) if n_boot_se > 1 else 0.7

    # Pure-tone summary metrics:
    aud_hl = prof.audiogram_hl_db
    aud_spl = aud_hl + use_retspl
    pta_124 = float(np.mean(aud_hl[[3, 5, 7]]))      # 1, 2, 4 kHz
    hf_pta = float(np.mean(aud_hl[[6, 7, 8, 9]]))    # 3, 4, 6, 8 kHz
    lf_pta = float(np.mean(aud_hl[[0, 1, 2, 3]]))    # 0.25, 0.5, 0.75, 1 kHz

    ppta_db = None
    if mnr_params is not None:
        ppta_db = float(
            predict_ppta_mnr(
                counts_tot[None, :, :],
                mnr_params["B"],
                mnr_params["C"],
                mnr_params["ca"],
                mnr_params["pa"],
            )[0]
        )

    # qCLS Loudness Scaling, REM Targets & Cross-Test Concordance Metrics:
    qcls_valid_monotonic = True
    cu5_aud_diff_db = None
    dyn_range_lf_db = None
    dyn_range_hf_db = None
    loudness_growth_cu_per_db = None
    cu50_hf_spl = None
    rem_targets: Optional[dict[str, list[float]]] = None
    if elc_10x10 is not None and not np.all(np.isnan(elc_10x10)):
        spans = elc_10x10[:, 9] - elc_10x10[:, 0]
        if np.nanmin(spans) < 8.0 or np.nanmax(elc_10x10[:, 0]) > 130.0:
            qcls_valid_monotonic = False
        else:
            cu5_aud_diff_db = float(np.nanmean(elc_10x10[:, 0] - aud_spl))
            dyn_range_lf_db = float(np.nanmean(spans[:4]))
            dyn_range_hf_db = float(np.nanmean(spans[6:]))
            loudness_growth_cu_per_db = float(np.nanmean(45.0 / np.maximum(spans, 10.0)))
            cu50_hf_spl = float(np.nanmean(elc_10x10[6:, 9]))
            # 0.5, 1, 2, 4 kHz REM / Verifit targets (indices 1, 3, 5, 7):
            rem_idx = [1, 3, 5, 7]
            rem_targets = {
                "freqs_khz": [0.5, 1.0, 2.0, 4.0],
                "soft_cu15_spl": [round(float(elc_10x10[idx_f, 2]), 0) for idx_f in rem_idx],
                "comfort_cu25_spl": [round(float(elc_10x10[idx_f, 4]), 0) for idx_f in rem_idx],
                "mpo_cu50_spl": [round(float(elc_10x10[idx_f, 9]), 0) for idx_f in rem_idx],
            }

    # Aided session evaluation (if provided):
    aided_eval: list[dict[str, Any]] = []
    for aid_label, c_aid in aided_sessions.items():
        pc_a = 100.0 * float(np.trace(c_aid[:, :10]) / max(c_aid.sum(), 1.0))
        d_eq_aid = norms.eq_snr_deficit_db(pc_a / 100.0)
        aided_eval.append({
            "aid_id": aid_label,
            "aid_type": "Lab Aid" if "HA" in aid_label else ("Personal Aid" if "HP" in aid_label else aid_label),
            "aided_pc": round(pc_a, 1),
            "delta_pc_vs_unaided": round(pc_a - emp_pc, 1),
            "aided_eq_deficit_db": round(d_eq_aid, 2),
            "snr_benefit_db": round(d_tot - d_eq_aid, 2),
        })

    # Top directional confusions:
    conf_pairs = []
    for i in range(N_STIMULI):
        row_n = max(float(counts_tot[i].sum()), 1.0)
        for j in range(N_RESPONSES):
            if i != j and counts_tot[i, j] >= 2:
                conf_pairs.append((
                    float(counts_tot[i, j] / row_n * 100.0),
                    int(round(counts_tot[i, j])),
                    int(round(row_n)),
                    STIMULI_CONSONANTS[i],
                    DISPLAY_CONSONANTS[j],
                ))
    conf_pairs.sort(reverse=True)

    # Helper clinical severity labels:
    def snr_loss_severity(db_val: float) -> str:
        if db_val <= norms.p90_total_db:
            return "Normal"
        if db_val <= 8.0:
            return "Mild"
        if db_val <= 15.0:
            return "Moderate"
        return "Severe"

    def smearing_label(erb_mult: float) -> str:
        if erb_mult < 1.25:
            return "Minimal"
        if erb_mult < 1.60:
            return "Mild"
        if erb_mult < 2.10:
            return "Moderate"
        return "Severe"

    snr_sev = snr_loss_severity(d_tot)
    hf_erb_mult = float(np.mean(prof.erb_broadening_factor(hp.gamma_ohc)[5:]))
    hf_smear_lbl = smearing_label(hf_erb_mult)

    # Automated Clinical Summary Classification & Rules:
    aud_elev = d_aud > norms.p90_aud_db
    ohc_elev = d_ohc > norms.p90_ohc_db or float(np.mean(prof.hl_ohc_db[5:])) >= 15.0
    neu_elev = (d_neu > norms.p90_neural_db) and (d_neu - 1.28 * se_neu > 0.3)
    neu_severe = d_neu > max(norms.p95_neural_db, 3.2)

    if not aud_elev and not ohc_elev and not neu_elev and emp_pc >= norms.nh_pc_p05:
        phenotype = "NORMAL HEARING & SPEECH-IN-NOISE CLARITY"
        pheno_color = GREEN_OK
    elif pta_124 <= 20.0 and np.max(aud_hl) <= 25.0 and (neu_elev or emp_pc < norms.nh_pc_p05):
        phenotype = "SUPRATHRESHOLD CLARITY LOSS WITH NORMAL AUDIOGRAM"
        pheno_color = RED_NEURAL
    elif pta_124 <= 20.0 and np.max(aud_hl) > 25.0 and neu_elev:
        phenotype = "HIGH-FREQUENCY LOSS + SUPRATHRESHOLD CLARITY LOSS"
        pheno_color = RED_NEURAL
    elif (aud_elev or ohc_elev) and not neu_elev:
        phenotype = "REDUCED AUDIBILITY & COCHLEAR RECRUITMENT (PRESERVED CLARITY)"
        pheno_color = CORAL_OHC
    elif (aud_elev or ohc_elev) and neu_severe:
        phenotype = "COCHLEAR RECRUITMENT + MARKED SUPRATHRESHOLD CLARITY LOSS"
        pheno_color = RED_NEURAL
    else:
        phenotype = "AUDIBILITY & RECRUITMENT LOSS + MILD CLARITY LOSS"
        pheno_color = AMBER_WARN

    # Cross-Test Quality Checks (Plain Audiologist Booth Language):
    qc_checks: list[tuple[str, str, str]] = []
    if not prof.has_measured_qcls:
        qc_checks.append(("WARN", "qCLS Loudness Test", "Not provided — recruitment estimated from audiogram average."))
    elif not qcls_valid_monotonic:
        qc_checks.append(("ALERT", "qCLS Loudness Test", "Irregular loudness contour (CU 5 vs. CU 50) — check calibration & repeat qCLS."))
    elif cu5_aud_diff_db is not None:
        if abs(cu5_aud_diff_db) <= 10.0:
            qc_checks.append(("PASS", "Audiogram vs. qCLS Softest (CU 5)", f"Agrees within {cu5_aud_diff_db:+.1f} dB across frequencies."))
        elif cu5_aud_diff_db > 10.0:
            qc_checks.append(("WARN", "Audiogram vs. qCLS Softest (CU 5)", f"CU 5 is {cu5_aud_diff_db:+.1f} dB above audiogram — cautious soft-sound criterion or check transducer setting."))
        else:
            qc_checks.append(("WARN", "Audiogram vs. qCLS Softest (CU 5)", f"CU 5 is {abs(cu5_aud_diff_db):.1f} dB softer than audiogram — check insert foam seal, booth noise, or transducer setting."))

    if practice_pc is not None:
        if practice_pc >= 85.0:
            qc_checks.append(("PASS", "Clear-Speech Practice (Quiet)", f"{practice_pc:.0f}% correct in quiet — task understood."))
        elif pc_quiet_pred * 100.0 < 82.0:
            qc_checks.append(("WARN", "Clear-Speech Practice (Quiet)", f"{practice_pc:.0f}% at 60 dB SPL (consistent with high-frequency inaudibility before amplification)."))
        else:
            qc_checks.append(("ALERT", "Clear-Speech Practice (Quiet)", f"{practice_pc:.0f}% despite audible speech — reduced quiet clarity; verify standard word recognition."))

    if emp_pc_s2 is not None:
        diff_s12 = abs(emp_pc_s1 - emp_pc_s2)
        status_s = "PASS" if diff_s12 <= 10.0 else "WARN"
        qc_checks.append((status_s, "Repeatability & '???' Unsure Rate", f"Test {emp_pc_s1:.0f}% / Retest {emp_pc_s2:.0f}% ({diff_s12:.0f}% diff); '???' unsure button used {unk_pct:.1f}%."))
    else:
        status_u = "PASS" if unk_pct <= 8.0 else "WARN"
        qc_checks.append((status_u, "Response Caution ('???' Unsure)", f"{n_trials}-trial run; '???' unsure button used on {unk_pct:.1f}% of trials."))

    # Optional Binaural Interaural Asymmetry Check (when contralateral summary is available):
    asymmetry_flagged = False
    if contra_summary is not None:
        d_pta_lr = abs(pta_124 - float(contra_summary.get("pta_124_db_hl", pta_124)))
        d_pc_lr = abs(emp_pc - float(contra_summary.get("emp_pc_pooled", emp_pc)))
        d_neu_lr = abs(d_neu - float(contra_summary.get("neural_coding_deficit_db", d_neu)))
        if d_pta_lr >= 15.0 or d_pc_lr >= 15.0 or d_neu_lr >= 4.0:
            asymmetry_flagged = True
            qc_checks.append((
                "ALERT",
                "Interaural Asymmetry",
                f"|ΔPTA|={d_pta_lr:.0f} dB, |ΔqVCV|={d_pc_lr:.0f}%, |ΔClarity|={d_neu_lr:.1f} dB — consider medical/otologic evaluation.",
            ))
        else:
            qc_checks.append((
                "PASS",
                "Binaural Symmetry",
                f"Symmetric with opposite ear (|ΔPTA|={d_pta_lr:.0f} dB, |ΔqVCV|={d_pc_lr:.0f}%).",
            ))

    # Targeted Clinical Fitting & Counseling Guidance:
    recs: list[str] = []
    if phenotype.startswith("NORMAL"):
        recs.append("Amplification: Not indicated. Threshold sensitivity, loudness growth, and speech-in-noise clarity are all within Normal-Hearing limits.")
        recs.append("Counseling & Follow-Up: Reassure patient; retain report as baseline for future comparison.")
    else:
        # 1. Gain & Bandwidth:
        if pta_124 <= 20.0 and np.max(aud_hl) <= 25.0:
            recs.append("Amplification Gain: Conventional hearing-aid gain is NOT indicated (all thresholds <= 20 dB HL; inaudibility accounts for < 1 dB of SNR loss).")
        elif pta_124 <= 20.0 and hf_pta >= 25.0:
            recs.append(f"Amplification Gain: Standard 1-4 kHz PTA ({pta_124:.0f} dB HL) misses 3-8 kHz loss ({hf_pta:.0f} dB HL); consider mild open-fit high-frequency gain if communication needs warrant.")
        else:
            recs.append(f"Amplification Gain: Prescribe frequency-specific gain (+{d_aud:.1f} dB SNR loss recoverable from audibility), prioritizing 2-8 kHz consonant cues (HF-PTA = {hf_pta:.0f} dB HL).")

        # 2. Compression & MPO (from qCLS):
        if ohc_elev and dyn_range_hf_db is not None:
            mpo_str = f" <= {cu50_hf_spl:.0f} dB SPL" if cu50_hf_spl is not None else ""
            recs.append(
                f"Compression & MPO (UCL): Steep HF loudness growth (HF dynamic range = {dyn_range_hf_db:.0f} dB; {hf_smear_lbl.lower()} spectral smearing). Use moderate WDRC ratios, cap HF MPO{mpo_str} (CU 50), and avoid excess low-frequency gain (<1 kHz) to limit upward spread of masking."
            )
        else:
            recs.append("Compression & MPO: Normal dynamic range supports low-ratio / near-linear WDRC to preserve consonant contrast without loudness discomfort.")

        # 3. Suprathreshold SNR Support & Aided Verification:
        if aided_eval:
            best_aid = max(aided_eval, key=lambda x: x["aided_pc"])
            worst_aid = min(aided_eval, key=lambda x: x["aided_pc"])
            if worst_aid["delta_pc_vs_unaided"] <= -4.0:
                recs.append(
                    f"AIDED ROLLOVER ALERT: Score dropped from {emp_pc:.0f}% unaided to {worst_aid['aided_pc']:.0f}% ({worst_aid['aid_type']}). Reduce low/mid-band gain, slow WDRC release times, and verify real-ear output."
                )
            elif best_aid["delta_pc_vs_unaided"] >= 15.0 and best_aid["aided_pc"] >= 80.0:
                recs.append(
                    f"AIDED VERIFICATION PASS: Amplification improved qVCV from {emp_pc:.0f}% to {best_aid['aided_pc']:.0f}% ({best_aid['aid_type']}), confirming preserved speech clarity once audibility is restored."
                )
            elif best_aid["aided_pc"] < 65.0:
                recs.append(
                    f"AIDED CLARITY CEILING: Aided score plateaued at {best_aid['aided_pc']:.0f}% (vs. {emp_pc:.0f}% unaided) due to +{d_neu:.1f} dB suprathreshold clarity loss and {dyn_range_hf_db:.0f} dB HF dynamic range — recommend directional mics or a Remote Microphone / DM system."
                )
        elif neu_elev:
            recs.append(
                f"Speech-in-Noise Technology & Counseling: +{d_neu:.1f} dB suprathreshold clarity loss will not be resolved by gain alone. Recommend directional or remote-microphone technology in noise, realistic counseling, and standard audiologic follow-up if asymmetric."
            )
    if asymmetry_flagged:
        recs.append("Otologic / Audiologic Referral: Marked interaural asymmetry detected — complete standard diagnostic immittance, reflex, and medical evaluation.")

    eff_lf = float(np.mean(eff_hat[LF_CONS_IDX]))
    eff_hf = float(np.mean(eff_hat[HF_CONS_IDX]))

    return {
        "listener_id": listener_id,
        "ear": ear_norm,
        "group": group_label,
        "n_trials": n_trials,
        "has_measured_qcls": prof.has_measured_qcls,
        "qcls_valid_monotonic": qcls_valid_monotonic,
        "pta_124_db_hl": round(pta_124, 1),
        "lf_pta_db_hl": round(lf_pta, 1),
        "hf_pta_db_hl": round(hf_pta, 1),
        "ppta_db_hl": round(ppta_db, 1) if ppta_db is not None else None,
        "practice_pc": round(practice_pc, 1) if practice_pc is not None else None,
        "emp_pc_s1": round(emp_pc_s1, 1),
        "emp_pc_s2": round(emp_pc_s2, 1) if emp_pc_s2 is not None else None,
        "emp_pc_pooled": round(emp_pc, 1),
        "unk_pct": round(unk_pct, 1),
        "pc_nh_ref": round(norms.pc_nh0 * 100.0, 1),
        "pc_stage1_aud": round(pc_aud * 100.0, 1),
        "pc_stage2_periph": round(pc_per * 100.0, 1),
        "pc_stage3_full": round(pc_ful * 100.0, 1),
        "total_deficit_db": round(d_tot, 2),
        "total_deficit_se_db": round(se_tot, 2),
        "audibility_deficit_db": round(d_aud, 2),
        "ohc_recruitment_deficit_db": round(d_ohc, 2),
        "neural_coding_deficit_db": round(d_neu, 2),
        "neural_coding_se_db": round(se_neu, 2),
        "snr_loss_severity": snr_sev,
        "measured_srt_loss_db": round(float(measured_srt_loss_db), 2) if measured_srt_loss_db is not None else None,
        "neural_eff_lf_pct": round(100.0 * min(eff_lf, 1.0), 0),
        "neural_eff_hf_pct": round(100.0 * min(eff_hf, 1.0), 0),
        "qcls_hf_ohc_db": round(float(np.mean(prof.hl_ohc_db[5:])), 1),
        "qcls_hf_ihc_db": round(float(np.mean(prof.hl_ihc_db[5:])), 1),
        "hf_erb_broadening_x": round(hf_erb_mult, 2),
        "hf_smearing_label": hf_smear_lbl,
        "cu5_aud_diff_db": round(cu5_aud_diff_db, 1) if cu5_aud_diff_db is not None else None,
        "dyn_range_lf_db": round(dyn_range_lf_db, 1) if dyn_range_lf_db is not None else None,
        "dyn_range_hf_db": round(dyn_range_hf_db, 1) if dyn_range_hf_db is not None else None,
        "loudness_growth_cu_per_db": round(loudness_growth_cu_per_db, 2) if loudness_growth_cu_per_db is not None else None,
        "rem_targets": rem_targets,
        "rtr_voicing_pct_nh": round(min(100.0 * rtr["voicing"] / max(norms.nh_rtr_voicing, 1e-4), 100.0), 0),
        "rtr_manner_pct_nh": round(min(100.0 * rtr["manner"] / max(norms.nh_rtr_manner, 1e-4), 100.0), 0),
        "rtr_place_pct_nh": round(min(100.0 * rtr["place"] / max(norms.nh_rtr_place, 1e-4), 100.0), 0),
        "phenotype": phenotype,
        "phenotype_color": pheno_color,
        "qc_checks": qc_checks,
        "recommendations": recs,
        "top_confusions": [
            {"stim": s, "resp": r_c, "pct": round(p, 1), "count": cnt, "out_of": tot_r}
            for (p, cnt, tot_r, s, r_c) in conf_pairs[:6]
        ],
        "aided_sessions": aided_eval,
        "audiogram_hl_db": prof.audiogram_hl_db.tolist(),
        "audiogram_spl_db": aud_spl.tolist(),
        "hl_ohc_db": prof.hl_ohc_db.tolist(),
        "hl_ihc_db": prof.hl_ihc_db.tolist(),
        "erb_factor": prof.erb_broadening_factor(hp.gamma_ohc).tolist(),
        "elc_10x10": elc_10x10.tolist() if elc_10x10 is not None else None,
        "smoothed_ccm": smoothed_ccm.tolist(),
        "raw_counts": counts_tot.tolist(),
    }


def render_clinical_report_page(
    rep: dict[str, Any],
    norms: ClinicalNorms,
    out_png: Path,
    out_pdf: Optional[Path] = None,
    pdf_booklet: Optional[PdfPages] = None,
    audiogram_topdown: bool = True,
) -> None:
    """Render a crisp, audiologist-friendly 1-page LANDSCAPE qPRO clinical report (US Letter 11:8.5 ratio)."""
    fig = plt.figure(figsize=(16.5, 12.75), dpi=170, facecolor=CREAM)
    ear_val = rep.get("ear", "Unspecified")
    if ear_val == "Right":
        ear_badge_txt, ear_badge_col, aud_mk, aud_col = "RIGHT EAR (O)", "#DC2626", "o", "#DC2626"
    elif ear_val == "Left":
        ear_badge_txt, ear_badge_col, aud_mk, aud_col = "LEFT EAR (X)", "#2563EB", "x", "#2563EB"
    elif ear_val == "Both":
        ear_badge_txt, ear_badge_col, aud_mk, aud_col = "BINAURAL", "#0F766E", "o", "#0F766E"
    else:
        ear_badge_txt, ear_badge_col, aud_mk, aud_col = "MONAURAL", "#475569", "o", "#0F172A"

    # ------------------------------------------------------------------
    # HEADER BANNER (Full Landscape Width)
    # ------------------------------------------------------------------
    ax_hdr = fig.add_axes([0.036, 0.872, 0.946, 0.106])
    ax_hdr.set_axis_off()
    hdr_box = mpatches.FancyBboxPatch(
        (0.0, 0.02),
        1.0,
        0.96,
        boxstyle="round,pad=0.010,rounding_size=0.018",
        facecolor=NAVY,
        edgecolor=GOLD,
        linewidth=2.0,
        transform=ax_hdr.transAxes,
    )
    ax_hdr.add_patch(hdr_box)

    ax_hdr.text(
        0.018,
        0.75,
        "qPRO — QUICK SUPRATHRESHOLD CLINICAL PROFILE",
        color="#FFFFFF",
        fontsize=14.0,
        fontweight="bold",
        transform=ax_hdr.transAxes,
        va="center",
    )
    # Ear Badge Pill in top-right next to subtitle:
    ear_pill = mpatches.FancyBboxPatch(
        (0.665, 0.62),
        0.108,
        0.24,
        boxstyle="round,pad=0.004,rounding_size=0.010",
        facecolor=ear_badge_col,
        edgecolor="#FFFFFF",
        linewidth=0.8,
        transform=ax_hdr.transAxes,
    )
    ax_hdr.add_patch(ear_pill)
    ax_hdr.text(
        0.719,
        0.74,
        ear_badge_txt,
        color="#FFFFFF",
        fontsize=8.2,
        fontweight="bold",
        ha="center",
        va="center",
        transform=ax_hdr.transAxes,
    )
    ax_hdr.text(
        0.982,
        0.75,
        "Audiogram + qCLS + qVCV Interpreter",
        color=GOLD,
        fontsize=10.0,
        fontweight="bold",
        ha="right",
        va="center",
        transform=ax_hdr.transAxes,
    )

    s12_txt = (
        f"{rep['emp_pc_pooled']:.1f}% (Test {rep['emp_pc_s1']:.0f}% / Retest {rep['emp_pc_s2']:.0f}%)"
        if rep["emp_pc_s2"] is not None
        else f"{rep['emp_pc_pooled']:.1f}% ({rep['n_trials']} trials)"
    )
    meta_line = (
        f"Patient ID: {rep['listener_id']}   |   Audiometric Group: {rep['group']}   |   "
        f"PTA (1,2,4 kHz): {rep['pta_124_db_hl']:.1f} dB HL   |   HF-PTA (3-8 kHz): {rep['hf_pta_db_hl']:.1f} dB HL   |   "
        f"qVCV: {s12_txt}   |   Est. SNR Loss: {rep['total_deficit_db']:+.1f} dB ({rep['snr_loss_severity']})"
    )
    ax_hdr.text(
        0.018,
        0.44,
        meta_line,
        color="#E2E8F0",
        fontsize=8.9,
        transform=ax_hdr.transAxes,
        va="center",
    )

    # Clinical Summary Pill inside Header:
    pill = mpatches.FancyBboxPatch(
        (0.018, 0.09),
        0.964,
        0.22,
        boxstyle="round,pad=0.005,rounding_size=0.010",
        facecolor=rep["phenotype_color"],
        edgecolor="#FFFFFF",
        linewidth=0.9,
        transform=ax_hdr.transAxes,
    )
    ax_hdr.add_patch(pill)
    ax_hdr.text(
        0.026,
        0.20,
        f"CLINICAL SUMMARY:  {rep['phenotype']}",
        color="#FFFFFF",
        fontsize=8.9,
        fontweight="bold",
        va="center",
        transform=ax_hdr.transAxes,
    )
    if rep["aided_sessions"]:
        aid_parts = [f"{a['aid_type']}: {a['aided_pc']:.0f}% ({a['delta_pc_vs_unaided']:+.0f}%)" for a in rep["aided_sessions"]]
        ax_hdr.text(
            0.974,
            0.20,
            "Aided:  " + "  •  ".join(aid_parts),
            color="#FFFFFF",
            fontsize=8.6,
            fontweight="bold",
            ha="right",
            va="center",
            transform=ax_hdr.transAxes,
        )

    x_idx = np.arange(10)
    aud_hl = np.array(rep["audiogram_hl_db"], dtype=float)
    aud_spl = np.array(rep.get("audiogram_spl_db", (aud_hl + RETSPL_TABLES["er3a"]).tolist()), dtype=float)

    # ------------------------------------------------------------------
    # ROW 1, COL 1 — PANEL A: Loudness Scaling & Threshold SPL-o-gram
    # ------------------------------------------------------------------
    ax_a = fig.add_axes([0.048, 0.488, 0.265, 0.340], facecolor=PANEL_BG)
    y_nh = norms.y_nh_elc_10x10
    ax_a.fill_between(
        x_idx,
        y_nh[:, 0] - norms.y_nh_cu5_sd,
        y_nh[:, 0] + norms.y_nh_cu5_sd,
        color="#BFDBFE",
        alpha=0.45,
        label="NH Norm Soft / UCL (±1 SD)",
    )
    ax_a.fill_between(
        x_idx,
        y_nh[:, 9] - norms.y_nh_cu50_sd,
        y_nh[:, 9] + norms.y_nh_cu50_sd,
        color="#FECACA",
        alpha=0.38,
    )

    # Consonant Cue Frequency Zones at top of Panel A (connects SPL-o-gram to confusions):
    cue_zones = [
        (0.0, 2.1, "Voicing / Nasals", "#DBEAFE", "#1E40AF"),
        (3.8, 6.2, "Place Bursts", "#FEF3C7", "#92400E"),
        (6.8, 9.0, "Fricatives (S/SH/Z)", "#FCE7F3", "#9D174D"),
    ]
    for z0, z1, z_lbl, z_bg, z_fg in cue_zones:
        ax_a.fill_between([z0, z1], 125.5, 136.5, color=z_bg, alpha=0.85, edgecolor=z_fg, linewidth=0.5)
        ax_a.text(0.5 * (z0 + z1), 131.0, z_lbl, ha="center", va="center", fontsize=6.4, fontweight="bold", color=z_fg)

    if rep["elc_10x10"] is not None and rep["qcls_valid_monotonic"]:
        elc = np.array(rep["elc_10x10"], dtype=float)
        cu_sel = [(0, "CU 5 (Very Soft)", "#2563EB", 2.0, "-"),
                  (2, "CU 15 (Soft)", "#0284C7", 1.3, "-"),
                  (4, "CU 25 (Comfortable)", "#7C3AED", 1.6, "-"),
                  (6, "CU 35 (Loud)", "#D97706", 1.3, "-"),
                  (9, "CU 50 (Too Loud / UCL)", "#DC2626", 2.1, "-")]
        for c_i, c_lbl, col, lw, ls in cu_sel:
            ax_a.plot(x_idx, elc[:, c_i], ls, color=col, lw=lw, marker="o", ms=3.6, label=c_lbl)
    else:
        ax_a.text(
            0.5, 0.48, "Valid qCLS Contours Not Available\n(Showing Audiogram in dB SPL Only)",
            ha="center", va="center", transform=ax_a.transAxes, fontsize=9.0, color=MUTED, fontweight="bold"
        )

    ax_a.plot(
        x_idx, aud_spl, "s--", color=NAVY, lw=1.9, ms=4.8, label="Audiogram (dB SPL)"
    )
    ax_a.axhline(60.0, color="#0F766E", ls=":", lw=1.4, label="Conversational Speech (60 dB SPL)")
    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(FREQ_LABELS, fontsize=7.8)
    ax_a.set_ylim(-15, 140)
    ax_a.set_xlabel("Frequency (Hz)", fontsize=8.3, fontweight="bold", color=SLATE)
    ax_a.set_ylabel("Presentation Level (dB SPL)", fontsize=8.3, fontweight="bold", color=SLATE)
    dr_txt = (
        f"  [HF Range: {rep['dyn_range_hf_db']:.0f} dB | {rep['loudness_growth_cu_per_db']:.2f} CU/dB]"
        if rep["dyn_range_hf_db"] is not None
        else ""
    )
    ax_a.set_title(f"A. Loudness Scaling & SPL-o-gram{dr_txt}", fontsize=8.5, fontweight="bold", color=NAVY, pad=7)
    ax_a.legend(loc="lower right", fontsize=6.4, ncol=2, framealpha=0.92)
    ax_a.grid(alpha=0.25)

    # ------------------------------------------------------------------
    # ROW 1, COL 2 — PANEL B: Cochlear Loss & Spectral Smearing (dB HL)
    # ------------------------------------------------------------------
    ax_b = fig.add_axes([0.362, 0.488, 0.265, 0.340], facecolor=PANEL_BG)
    ohc_db = np.array(rep["hl_ohc_db"], dtype=float)
    ihc_db = np.array(rep["hl_ihc_db"], dtype=float)
    erb_x = np.array(rep["erb_factor"], dtype=float)

    ax_b.axhspan(-12, 20, color="#DCFCE7", alpha=0.45, label="Normal Threshold Zone (<= 20 dB HL)")
    ax_b.bar(x_idx, ohc_db, width=0.62, color=CORAL_OHC, alpha=0.86, label="Outer Hair Cell Loss (Loudness Recruitment)")
    ax_b.bar(x_idx, ihc_db, width=0.62, bottom=ohc_db, color=BLUE_AUD, alpha=0.82, label="Inner Hair Cell / Sensitivity Loss")
    if aud_mk == "o" and ear_val == "Right":
        ax_b.plot(x_idx, aud_hl, "-", color=aud_col, lw=1.7, marker="o", mfc="none", mec=aud_col, mew=2.0, ms=6.2, label="Pure-Tone Audiogram (dB HL)")
    elif aud_mk == "x":
        ax_b.plot(x_idx, aud_hl, "-", color=aud_col, lw=1.7, marker="x", mec=aud_col, mew=2.0, ms=6.5, label="Pure-Tone Audiogram (dB HL)")
    else:
        ax_b.plot(x_idx, aud_hl, "o-", color=aud_col, lw=1.6, ms=4.8, label="Pure-Tone Audiogram (dB HL)")

    def _short_smear_tag(mult: float) -> str:
        if mult < 1.25:
            return "Min"
        if mult < 1.60:
            return "Mild"
        if mult < 2.10:
            return "Mod"
        return "Sev"

    y_max_b = max(92.0, float(np.max(aud_hl)) + 18.0)
    for k in [1, 3, 5, 7, 9]:
        top_y = max(ohc_db[k] + ihc_db[k], aud_hl[k], 0.0)
        tag_s = _short_smear_tag(float(erb_x[k]))
        if audiogram_topdown:
            ax_b.text(
                x_idx[k], top_y + 2.6, tag_s, ha="center", va="top",
                fontsize=7.2, fontweight="bold", color=CORAL_OHC if erb_x[k] >= 1.25 else SLATE
            )
        else:
            ax_b.text(
                x_idx[k], top_y + 2.2, tag_s, ha="center", va="bottom",
                fontsize=7.2, fontweight="bold", color=CORAL_OHC if erb_x[k] >= 1.25 else SLATE
            )

    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(FREQ_LABELS, fontsize=7.8)
    if audiogram_topdown:
        ax_b.set_ylim(y_max_b, -12)
        tag_pos_note = "Tip tags"
        leg_loc_b = "lower left"
    else:
        ax_b.set_ylim(-8, y_max_b)
        tag_pos_note = "Top tags"
        leg_loc_b = "upper left"
    ax_b.set_xlabel(f"Frequency (Hz)  [{tag_pos_note} = Smearing: Min / Mild / Mod / Sev]", fontsize=7.85, fontweight="bold", color=SLATE)
    ax_b.set_ylabel("Hearing Level (dB HL)", fontsize=8.3, fontweight="bold", color=SLATE)
    ax_b.set_title(
        f"B. Audiogram & Cochlear Loss  [HF Recr.: {rep['qcls_hf_ohc_db']:.0f} dB | {rep['hf_smearing_label']}]",
        fontsize=8.5, fontweight="bold", color=NAVY, pad=7
    )
    ax_b.legend(loc=leg_loc_b, fontsize=6.65, framealpha=0.92)
    ax_b.grid(axis="y", alpha=0.25)

    # ------------------------------------------------------------------
    # ROW 1, COL 3 — PANEL C: Estimated SNR Loss Breakdown (dB)
    # ------------------------------------------------------------------
    ax_c = fig.add_axes([0.724, 0.488, 0.258, 0.340], facecolor=PANEL_BG)
    labels_c = [
        "Total Estimated\nSNR Loss",
        "1. Inaudibility\n(Restorable w/ Gain)",
        "2. Recruitment &\nUpward Masking",
        "3. Suprathreshold\nClarity Loss",
    ]
    vals_c = [
        rep["total_deficit_db"],
        rep["audibility_deficit_db"],
        rep["ohc_recruitment_deficit_db"],
        rep["neural_coding_deficit_db"],
    ]
    ses_c = [rep["total_deficit_se_db"], 0.0, 0.0, rep["neural_coding_se_db"]]
    cols_c = [NAVY, BLUE_AUD, CORAL_OHC, RED_NEURAL if rep["neural_coding_deficit_db"] > norms.p90_neural_db else GREEN_OK]
    yc = np.arange(len(labels_c))[::-1]

    ax_c.axvspan(-2.5, norms.p90_total_db, color="#DCFCE7", alpha=0.55, label=f"Normal Range (<= +{norms.p90_total_db:.1f} dB)")
    ax_c.axvline(0.0, color=SLATE, lw=1.0, ls="-")
    ax_c.axvline(norms.p90_neural_db, color=RED_NEURAL, lw=1.1, ls="--", alpha=0.7, label=f"Clarity Loss Cutoff (+{norms.p90_neural_db:.1f} dB)")

    x_hi_c = max(18.0, max(vals_c) + 6.5)
    for idx_bar, (yv, val, se, col) in enumerate(zip(yc, vals_c, ses_c, cols_c)):
        ax_c.barh(yv, val, height=0.52, color=col, alpha=0.88, edgecolor="#1E293B", linewidth=0.7)
        if se > 0.05:
            ax_c.errorbar(val, yv, xerr=se, fmt="none", ecolor="#0F172A", elinewidth=1.5, capsize=3.5)
        txt_x = max(val + se + 0.35, 0.4) if val >= 0 else 0.4
        se_str = f" ± {se:.1f} dB" if se > 0.05 else " dB"
        ax_c.text(txt_x, yv, f"{val:+.1f}{se_str}", va="center", ha="left", fontsize=8.0, fontweight="bold", color=col)

    ax_c.set_yticks(yc)
    ax_c.set_yticklabels(labels_c, fontsize=7.8, fontweight="bold", color=SLATE)
    ax_c.set_xlim(-2.5, x_hi_c)
    ax_c.set_xlabel("Estimated SNR Loss vs. Normal Hearing (dB) ↑", fontsize=8.1, fontweight="bold", color=SLATE)
    ax_c.set_title(
        f"C. Estimated SNR Loss Breakdown  [Overall: {rep['snr_loss_severity']}]",
        fontsize=8.55, fontweight="bold", color=NAVY, pad=7
    )
    ax_c.legend(loc="center right", bbox_to_anchor=(0.99, 0.39), fontsize=6.7, framealpha=0.94)
    ax_c.grid(axis="x", alpha=0.25)

    # ------------------------------------------------------------------
    # ROW 2, COL 1 — PANEL D: Expected vs. Actual qVCV Score
    # ------------------------------------------------------------------
    ax_d = fig.add_axes([0.048, 0.074, 0.265, 0.338], facecolor=PANEL_BG)
    wf_labels = [
        "Normal\nHearing\nAvg",
        "Expected:\nAudiogram\nOnly",
        "Expected:\nAudiogram\n+ qCLS",
        "Actual\nUnaided\nqVCV",
    ]
    wf_vals = [
        rep["pc_nh_ref"],
        rep["pc_stage1_aud"],
        rep["pc_stage2_periph"],
        rep["emp_pc_pooled"],
    ]
    wf_cols = ["#15803D", BLUE_AUD, CORAL_OHC, NAVY]

    for a_ev in rep["aided_sessions"]:
        lbl_short = a_ev["aid_type"].replace(" ", "\n")
        wf_labels.append(f"Actual\n{lbl_short}")
        wf_vals.append(a_ev["aided_pc"])
        wf_cols.append(TEAL_AID if a_ev["delta_pc_vs_unaided"] >= 0 else RED_NEURAL)

    xd = np.arange(len(wf_vals))
    bars_d = ax_d.bar(xd, wf_vals, width=0.54, color=wf_cols, alpha=0.88, edgecolor="#1E293B", linewidth=0.8)
    ax_d.axhspan(norms.nh_pc_p05, 103.0, color="#DCFCE7", alpha=0.45, label=f"Normal Hearing Range (>={norms.nh_pc_p05:.0f}%)")
    ax_d.axhline(10.0, color=MUTED, ls=":", lw=1.1, label="Guessing Floor (10%)")

    if rep["emp_pc_s2"] is not None:
        ax_d.scatter([3 - 0.11, 3 + 0.11], [rep["emp_pc_s1"], rep["emp_pc_s2"]], color=GOLD, edgecolor="#000000", s=34, zorder=5, label="Test / Retest Runs")

    for idx_d, (b_rect, v_pct) in enumerate(zip(bars_d, wf_vals)):
        top_ref = v_pct
        if idx_d == 3 and rep["emp_pc_s2"] is not None:
            top_ref = max(v_pct, rep["emp_pc_s1"], rep["emp_pc_s2"])
        ax_d.text(
            b_rect.get_x() + b_rect.get_width() / 2.0,
            min(top_ref + 2.0, 99.6),
            f"{v_pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8.0,
            fontweight="bold",
            color=NAVY,
        )

    ax_d.set_xticks(xd)
    ax_d.set_xticklabels(wf_labels, fontsize=7.25, fontweight="bold", color=SLATE)
    ax_d.set_ylim(0, 108)
    ax_d.set_ylabel("qVCV Consonant Recognition (%)", fontsize=8.3, fontweight="bold", color=SLATE)
    gap_pp = rep["emp_pc_pooled"] - rep["pc_stage2_periph"]
    ax_d.set_title(
        f"D. Expected vs. Actual qVCV Score  [Actual - Exp: {gap_pp:+.1f}%]",
        fontsize=8.55, fontweight="bold", color=NAVY, pad=7
    )
    ax_d.legend(loc="lower left", fontsize=6.7, framealpha=0.92)
    ax_d.grid(axis="y", alpha=0.25)

    # ------------------------------------------------------------------
    # ROW 2, COL 2 — PANEL E: Consonant Confusion Pattern (%)
    # ------------------------------------------------------------------
    ax_e = fig.add_axes([0.362, 0.074, 0.265, 0.338], facecolor=PANEL_BG)
    ccm = np.array(rep["smoothed_ccm"], dtype=float) * 100.0
    raw_c = np.array(rep["raw_counts"], dtype=float)

    disp_rgb = np.ones((N_STIMULI, N_RESPONSES, 3), dtype=float)
    for i in range(N_STIMULI):
        for j in range(N_RESPONSES):
            p_ij = ccm[i, j] / 100.0
            if i == j:
                disp_rgb[i, j] = np.array([1.0 - 0.72 * p_ij, 1.0 - 0.52 * p_ij, 1.0 - 0.18 * p_ij])
            else:
                strength = min(p_ij / 0.55, 1.0)
                disp_rgb[i, j] = np.array([1.0, 1.0 - 0.68 * strength, 1.0 - 0.80 * strength])

    ax_e.imshow(disp_rgb, aspect="auto")
    for i in range(N_STIMULI):
        for j in range(N_RESPONSES):
            val_p = ccm[i, j]
            cnt_ij = int(round(raw_c[i, j]))
            if i == j:
                txt_col = "#FFFFFF" if val_p >= 62.0 else NAVY
                ax_e.text(j, i, f"{val_p:.0f}", ha="center", va="center", fontsize=7.4, fontweight="bold", color=txt_col)
            elif cnt_ij >= 1 or val_p >= 6.0:
                txt_col = "#7F1D1D" if val_p >= 15.0 else SLATE
                ax_e.text(j, i, f"{val_p:.0f}", ha="center", va="center", fontsize=6.9, fontweight="bold" if val_p >= 15.0 else "normal", color=txt_col)

    ax_e.set_xticks(np.arange(N_RESPONSES))
    ax_e.set_xticklabels(DISPLAY_CONSONANTS, fontsize=7.6, fontweight="bold", color=NAVY)
    ax_e.set_yticks(np.arange(N_STIMULI))
    ax_e.set_yticklabels(STIMULI_CONSONANTS, fontsize=7.6, fontweight="bold", color=NAVY)
    ax_e.set_xlabel(
        f"Patient Response (%)  |  Cue Clarity (% Normal): Voicing {rep['rtr_voicing_pct_nh']:.0f}% • Manner {rep['rtr_manner_pct_nh']:.0f}% • Place {rep['rtr_place_pct_nh']:.0f}%",
        fontsize=7.25, fontweight="bold", color=SLATE
    )
    ax_e.set_ylabel("Presented Consonant", fontsize=8.3, fontweight="bold", color=SLATE)
    ax_e.set_title("E. Consonant Confusion Pattern (%) — De-Noised Estimate", fontsize=8.55, fontweight="bold", color=NAVY, pad=7)

    # ------------------------------------------------------------------
    # ROW 2, COL 3 — PANEL F: Clinical Interpretation & Fitting Guidance
    # ------------------------------------------------------------------
    ax_f = fig.add_axes([0.654, 0.048, 0.328, 0.376], facecolor=PANEL_BG)
    ax_f.set_axis_off()
    card = mpatches.FancyBboxPatch(
        (0.0, 0.0),
        1.0,
        1.0,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        facecolor="#FFFFFF",
        edgecolor="#CBD5E1",
        linewidth=1.2,
        transform=ax_f.transAxes,
    )
    ax_f.add_patch(card)

    y_cur = 0.948
    ax_f.text(0.030, y_cur, "F. Clinical Interpretation, Test Quality & Fitting Guidance", fontsize=9.2, fontweight="bold", color=NAVY, transform=ax_f.transAxes)
    y_cur -= 0.048

    # Section 1: Test Quality & Booth Cross-Checks
    ax_f.text(0.030, y_cur, "1. TEST QUALITY & BOOTH CROSS-CHECKS", fontsize=7.75, fontweight="bold", color=SLATE, transform=ax_f.transAxes)
    y_cur -= 0.036
    badge_colors = {"PASS": GREEN_OK, "WARN": AMBER_WARN, "ALERT": RED_NEURAL}
    for st, chk_name, chk_msg in rep["qc_checks"]:
        b_col = badge_colors.get(st, SLATE)
        ax_f.text(
            0.032, y_cur, f"[{st}]", fontsize=7.2, fontweight="bold", color=b_col, transform=ax_f.transAxes, va="top"
        )
        wrapped_qc = textwrap.fill(f"{chk_name}: {chk_msg}", width=67)
        n_lines = wrapped_qc.count("\n") + 1
        ax_f.text(0.145, y_cur, wrapped_qc, fontsize=7.2, color="#1E293B", transform=ax_f.transAxes, va="top")
        y_cur -= 0.0315 * n_lines + 0.008

    # Section 2: Main Consonant Confusions
    y_cur -= 0.010
    ax_f.text(0.030, y_cur, "2. MAIN CONSONANT CONFUSIONS", fontsize=7.75, fontweight="bold", color=SLATE, transform=ax_f.transAxes)
    y_cur -= 0.035
    if rep["top_confusions"]:
        conf_strs = [f"Heard '{c['stim']}' as '{c['resp']}' ({c['count']}/{c['out_of']}, {c['pct']:.0f}%)" for c in rep["top_confusions"][:4]]
        conf_line = "  •  ".join(conf_strs)
    else:
        conf_line = "Minimal consonant confusions (<10% per pair); Voicing, Manner, and Place speech cues preserved."
    wrapped_conf = textwrap.fill(conf_line, width=75)
    ax_f.text(0.034, y_cur, wrapped_conf, fontsize=7.25, color="#1E293B", fontweight="bold" if rep["top_confusions"] else "normal", transform=ax_f.transAxes, va="top")
    y_cur -= 0.0315 * (wrapped_conf.count("\n") + 1) + 0.018

    # Section 3: Clinical Fitting & Counseling Guidance + Direct REM / Verifit Targets
    ax_f.text(0.030, y_cur, "3. CLINICAL FITTING & COUNSELING GUIDANCE", fontsize=7.75, fontweight="bold", color=SLATE, transform=ax_f.transAxes)
    y_cur -= 0.034

    rem_t = rep.get("rem_targets")
    if rem_t is not None and all(np.isfinite(rem_t["mpo_cu50_spl"])):
        c25_s = "/".join(f"{int(v)}" for v in rem_t["comfort_cu25_spl"])
        mpo_s = "/".join(f"{int(v)}" for v in rem_t["mpo_cu50_spl"])
        rem_box = mpatches.FancyBboxPatch(
            (0.030, y_cur - 0.045),
            0.940,
            0.048,
            boxstyle="round,pad=0.004,rounding_size=0.010",
            facecolor="#EFF6FF",
            edgecolor="#93C5FD",
            linewidth=0.8,
            transform=ax_f.transAxes,
        )
        ax_f.add_patch(rem_box)
        ax_f.text(
            0.042,
            y_cur - 0.021,
            f"REM (0.5/1/2/4 kHz, dB SPL):  Comfort CU25 = {c25_s}  •  MPO Cap CU50 = {mpo_s}",
            fontsize=6.8,
            fontweight="bold",
            color="#1E3A8A",
            va="center",
            transform=ax_f.transAxes,
        )
        y_cur -= 0.058

    for rec_item in rep["recommendations"]:
        wrapped_r = textwrap.fill(f"• {rec_item}", width=76, subsequent_indent="  ")
        n_lr = wrapped_r.count("\n") + 1
        if y_cur - 0.030 * n_lr < 0.008:
            break
        ax_f.text(0.032, y_cur, wrapped_r, fontsize=7.15, color="#0F172A", transform=ax_f.transAxes, va="top")
        y_cur -= 0.031 * n_lr + 0.009

    out_png.parent.mkdir(parents=True, exist_ok=True)
    buf_png = io.BytesIO()
    fig.savefig(buf_png, format="png", dpi=170, facecolor=CREAM)
    out_png.write_bytes(buf_png.getvalue())
    if out_pdf is not None:
        buf_pdf = io.BytesIO()
        fig.savefig(buf_pdf, format="pdf", facecolor=CREAM)
        out_pdf.write_bytes(buf_pdf.getvalue())
    if pdf_booklet is not None:
        pdf_booklet.savefig(fig, facecolor=CREAM)
    plt.close(fig)


def render_overview_atlas_figure(
    case_reports: list[dict[str, Any]],
    norms: ClinicalNorms,
    all_unaided_reports: list[dict[str, Any]],
    out_path: Path,
    pdf_booklet: Optional[PdfPages] = None,
) -> None:
    """Render a 1-page multi-case summary comparing the 6 prototype cases & 98-listener cohort."""
    fig, axes = plt.subplots(2, 2, figsize=(16.5, 12.75), dpi=170, facecolor=CREAM)
    fig.suptitle(
        "qPRO — Quick Suprathreshold Profile: 6 Clinical Case Examples & 98-Listener Cohort\n"
        "(Automated SNR Loss & REM Target Synthesis from Pure-Tone Audiogram + qCLS + qVCV)",
        fontsize=12.4,
        fontweight="bold",
        color=NAVY,
        y=0.982,
    )

    # Panel A: Stacked 3-Factor SNR Loss Breakdown Across the 6 Prototype Patients
    ax = axes[0, 0]
    ax.set_facecolor(PANEL_BG)
    ids_6 = [
        f"{r['listener_id']}\n(PTA {r['pta_124_db_hl']:.0f} dB)"
        for r in case_reports
    ]
    x6 = np.arange(len(case_reports))
    b_aud = np.array([r["audibility_deficit_db"] for r in case_reports])
    b_ohc = np.array([r["ohc_recruitment_deficit_db"] for r in case_reports])
    b_neu = np.array([max(r["neural_coding_deficit_db"], 0.0) for r in case_reports])
    tot_v = np.array([r["total_deficit_db"] for r in case_reports])

    ax.bar(x6, b_aud, 0.54, color=BLUE_AUD, label="1. Inaudibility (Restorable with Gain)")
    ax.bar(x6, b_ohc, 0.54, bottom=b_aud, color=CORAL_OHC, label="2. Loudness Recruitment & Upward Masking (qCLS)")
    ax.bar(x6, b_neu, 0.54, bottom=b_aud + b_ohc, color=RED_NEURAL, label="3. Suprathreshold Clarity Loss (Beyond Audibility/qCLS)")
    ax.axhline(norms.p90_total_db, color=GREEN_OK, ls="--", lw=1.3, label=f"Normal Range Cutoff (+{norms.p90_total_db:.1f} dB)")

    for k, r in enumerate(case_reports):
        top_k = float(b_aud[k] + b_ohc[k] + b_neu[k])
        ax.text(k, top_k + 0.35, f"{tot_v[k]:+.1f} dB", ha="center", va="bottom", fontsize=8.3, fontweight="bold", color=NAVY)

    ax.set_xticks(x6)
    ax.set_xticklabels(ids_6, fontsize=8.4, fontweight="bold")
    ax.set_ylim(0, max(float(np.max(b_aud + b_ohc + b_neu)) + 3.2, 18.0))
    ax.set_ylabel("Estimated SNR Loss (dB) ↑", fontsize=9.2, fontweight="bold", color=SLATE)
    ax.set_title("A. Estimated SNR Loss Breakdown Across 6 Clinical Case Examples", fontsize=10.4, fontweight="bold", color=NAVY)
    ax.legend(loc="upper left", fontsize=8.0)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Expected Score vs. Actual Unaided & Aided qVCV Scores
    ax = axes[0, 1]
    ax.set_facecolor(PANEL_BG)
    periph_p = [r["pc_stage2_periph"] for r in case_reports]
    unaid_p = [r["emp_pc_pooled"] for r in case_reports]

    ax.bar(x6 - 0.22, periph_p, 0.21, color="#93C5FD", edgecolor=BLUE_AUD, lw=1.0, label="Expected from Audiogram + qCLS (Intact Clarity)")
    ax.bar(x6, unaid_p, 0.21, color=NAVY, label="Actual Unaided qVCV (%)")

    aid_vals = []
    for k, r in enumerate(case_reports):
        if r["aided_sessions"]:
            ha_match = [a for a in r["aided_sessions"] if "Lab" in a["aid_type"]]
            a_use = ha_match[0] if ha_match else r["aided_sessions"][0]
            aid_vals.append((k, a_use["aided_pc"], a_use["delta_pc_vs_unaided"]))
    for idx_a, (k, apc, dlt) in enumerate(aid_vals):
        col_a = TEAL_AID if dlt >= 0 else RED_NEURAL
        ax.bar(k + 0.22, apc, 0.21, color=col_a, label="Actual Aided qVCV (Lab Aid)" if idx_a == 0 else None)
        ax.text(k + 0.22, apc + 1.8, f"{dlt:+.0f}%", ha="center", va="bottom", fontsize=8.0, fontweight="bold", color=col_a)

    ax.axhspan(norms.nh_pc_p05, 102, color="#DCFCE7", alpha=0.45, label=f"Normal Hearing Range (>={norms.nh_pc_p05:.0f}%)")
    ax.set_xticks(x6)
    ax.set_xticklabels(ids_6, fontsize=8.4, fontweight="bold")
    ax.set_ylim(0, 108)
    ax.set_ylabel("qVCV Recognition Score (%)", fontsize=9.2, fontweight="bold", color=SLATE)
    ax.set_title("B. Expected vs. Actual Unaided & Aided qVCV Scores", fontsize=10.4, fontweight="bold", color=NAVY)
    ax.legend(loc="lower left", fontsize=7.8)
    ax.grid(axis="y", alpha=0.25)

    # Panel C: Full 98-Listener Cohort Map — Audibility/Recruitment vs. Suprathreshold Clarity Loss
    ax = axes[1, 0]
    ax.set_facecolor(PANEL_BG)
    grp_styles = {
        "Normal": ("#15803D", "o"),
        "Slight": ("#0284C7", "^"),
        "Mild": ("#D97706", "s"),
        "Moderate": ("#B91C1C", "D"),
    }
    for g_name, (g_col, g_mk) in grp_styles.items():
        sub = [r for r in all_unaided_reports if r["group"] == g_name]
        px = [r["audibility_deficit_db"] + r["ohc_recruitment_deficit_db"] for r in sub]
        py = [r["neural_coding_deficit_db"] for r in sub]
        ax.scatter(px, py, c=g_col, marker=g_mk, s=38, alpha=0.72, label=f"{g_name} (n={len(sub)})")

    ax.axhline(norms.p90_neural_db, color=RED_NEURAL, ls="--", lw=1.2, label=f"Clarity Loss Cutoff (+{norms.p90_neural_db:.1f} dB)")
    ax.axvline(norms.p90_aud_db + norms.p90_ohc_db, color=BLUE_AUD, ls=":", lw=1.2, label="Audibility/Recruitment Cutoff")

    for r in case_reports:
        px_i = r["audibility_deficit_db"] + r["ohc_recruitment_deficit_db"]
        py_i = r["neural_coding_deficit_db"]
        ax.scatter([px_i], [py_i], s=115, facecolors="none", edgecolors="#000000", linewidths=2.0, zorder=6)
        ax.annotate(
            r["listener_id"],
            (px_i, py_i),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8.3,
            fontweight="bold",
            color=NAVY,
        )

    ax.set_xlabel("Inaudibility + Loudness Recruitment SNR Loss (dB) →", fontsize=9.0, fontweight="bold", color=SLATE)
    ax.set_ylabel("Suprathreshold Clarity SNR Loss (dB) →", fontsize=9.0, fontweight="bold", color=SLATE)
    ax.set_title("C. Clinical Profile Map Across All 98 Listeners", fontsize=10.4, fontweight="bold", color=NAVY)
    ax.legend(loc="upper right", fontsize=7.6, ncol=2)
    ax.grid(alpha=0.25)

    # Panel D: Distribution of Automated Clinical Summaries Across the 98-Listener Cohort
    ax = axes[1, 1]
    ax.set_facecolor(PANEL_BG)
    short_map = {
        "NORMAL HEARING & SPEECH-IN-NOISE CLARITY": "Normal Hearing &\nSIN Clarity",
        "SUPRATHRESHOLD CLARITY LOSS WITH NORMAL AUDIOGRAM": "Clarity Loss w/\nNormal Audiogram",
        "HIGH-FREQUENCY LOSS + SUPRATHRESHOLD CLARITY LOSS": "HF Loss +\nClarity Loss",
        "REDUCED AUDIBILITY & COCHLEAR RECRUITMENT (PRESERVED CLARITY)": "Audibility & Recr.\n(Preserved Clarity)",
        "AUDIBILITY & RECRUITMENT LOSS + MILD CLARITY LOSS": "Audibility/Recr. +\nMild Clarity Loss",
        "COCHLEAR RECRUITMENT + MARKED SUPRATHRESHOLD CLARITY LOSS": "Recruitment +\nMarked Clarity Loss",
    }
    ph_order = list(short_map.keys())
    ph_cols = [GREEN_OK, "#DC2626", "#E11D48", CORAL_OHC, AMBER_WARN, "#7F1D1D"]
    counts_ph = [sum(1 for r in all_unaided_reports if r["phenotype"] == k) for k in ph_order]
    yp = np.arange(len(ph_order))[::-1]

    bars_p = ax.barh(yp, counts_ph, height=0.56, color=ph_cols, alpha=0.88, edgecolor="#1E293B")
    for b_r, cnt in zip(bars_p, counts_ph):
        ax.text(cnt + 0.6, b_r.get_y() + b_r.get_height() / 2.0, f"{cnt} ({100.0*cnt/len(all_unaided_reports):.0f}%)", va="center", fontsize=8.5, fontweight="bold", color=NAVY)

    ax.set_yticks(yp)
    ax.set_yticklabels([short_map[k] for k in ph_order], fontsize=8.2, fontweight="bold", color=SLATE)
    ax.set_xlim(0, max(counts_ph) + 9)
    ax.set_xlabel("Number of Listeners in Cohort (n = 98)", fontsize=9.0, fontweight="bold", color=SLATE)
    ax.set_title("D. Clinical Summary Distribution in qtest_dataset", fontsize=10.4, fontweight="bold", color=NAVY)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout(rect=[0, 0, 1, 0.942])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf_atlas = io.BytesIO()
    fig.savefig(buf_atlas, format="png", dpi=165, facecolor=CREAM)
    out_path.write_bytes(buf_atlas.getvalue())
    if pdf_booklet is not None:
        pdf_booklet.savefig(fig, facecolor=CREAM)
    plt.close(fig)


def load_qvcv_csv_counts(csv_path: Path) -> np.ndarray:
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
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="qPRO — Generate automated suprathreshold clinical profiles.")
    parser.add_argument("--listener_id", type=str, default=None, help="Generate a 1-page report for a specific qtest_dataset ID (e.g. HL054)")
    parser.add_argument("--qvcv_csv", type=str, default=None, help="Optional single qVCV_*.csv file to interpret")
    parser.add_argument("--qcls_elc_csv", type=str, default=None, help="Optional paired qCLS_ELC_*.csv file")
    parser.add_argument("--audiogram", type=str, default=None, help="Optional 10 comma-separated thresholds (250..8000 Hz)")
    parser.add_argument("--audiogram_units", choices=["hl", "spl"], default="hl", help="Units of --audiogram (default: hl)")
    parser.add_argument("--ear", choices=["Right", "Left", "Both", "Unspecified"], default="Unspecified", help="Test ear badge (Right O, Left X)")
    parser.add_argument("--audiogram_orientation", choices=["topdown", "bottomup"], default="topdown", help="Panel B dB HL orientation (default: topdown)")
    parser.add_argument("--measured_srt_snr", type=float, default=None, help="Optional direct adaptive-SNR loss (dB) override")
    parser.add_argument("--transducer", choices=["er3a", "legacy", "ansi"], default="er3a", help="BTNRH RETSPL table matching qCLS (default: er3a)")
    parser.add_argument("--aided_qvcv_csv", type=str, default=None, help="Optional aided qVCV_*.csv file")
    parser.add_argument("--out", type=str, default=None, help="Output PNG path for single-patient mode (PDF saved alongside)")
    args = parser.parse_args()

    sim_dir = Path(__file__).resolve().parent
    human_dir = sim_dir / "human_data"
    out_dir = sim_dir / "clinical_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    sel_retspl = RETSPL_TABLES[args.transducer]

    print("[1/4] Loading waveform spectro-temporal cues, qCLS contours, and 98-listener normative reference...", flush=True)
    bank = extract_vcv_acoustic_bank(DEFAULT_STIMULI_DIR)
    base_data = json.loads((human_dir / "human_listeners_98.json").read_text())
    bio_data = json.loads((human_dir / "biophysical_listeners_98.json").read_text())
    mnr_raw = json.loads((human_dir / "mnr_vcv_pta_model.json").read_text())
    mnr_params = {k: np.array(mnr_raw[k], dtype=float) for k in ("B", "C", "ca", "pa")}

    base_listeners = base_data["listeners"]
    qcls_map = load_qcls_cochlear_profiles(base_listeners, DEFAULT_QCLS_MAT)
    elc_map, y_nh, cu5_sd, cu50_sd = load_raw_qcls_elc_matrix_map(DEFAULT_QCLS_MAT)
    norms, hp, self_off = build_clinical_norms(bank, base_listeners, bio_data, qcls_map, y_nh, cu5_sd, cu50_sd)

    # Single qtest_dataset Listener ID Mode:
    if args.listener_id is not None:
        sid = args.listener_id.strip().upper()
        base_by_id = {r["listener_id"]: r for r in base_listeners}
        if sid not in base_by_id:
            raise SystemExit(f"Unknown listener_id '{sid}'. Valid IDs: {', '.join(sorted(base_by_id.keys()))}")
        aid_dict_ds: dict[str, np.ndarray] = {}
        perf_mat_path = Path("/Users/neely/STN/2026_Quick_Tests/qtest_dataset/qVCV_analysis/QVC_perf.mat")
        if perf_mat_path.exists() and sid.startswith("HL"):
            mp = loadmat(str(perf_mat_path))
            ids_p = ["".join(ch).strip() for ch in mp["id"]]
            ccms_p = mp["ccms"]
            for idx_p, aid_id in enumerate(ids_p):
                if aid_id in ("HA" + sid[2:], "HP" + sid[2:]):
                    aid_dict_ds[aid_id] = np.asarray(ccms_p[idx_p], dtype=float)
        r_ds = base_by_id[sid]
        rep_ds = evaluate_listener_clinical_profile(
            listener_id=sid,
            group_label=r_ds["archetype"],
            prof=qcls_map[sid],
            elc_10x10=elc_map.get(sid, None),
            counts_s1=np.array(r_ds["empirical_s1_counts"], dtype=float),
            counts_s2=np.array(r_ds["empirical_s2_counts"], dtype=float),
            practice_pc=float(r_ds["practice_pc"]),
            aided_sessions=aid_dict_ds,
            bank=bank,
            hp=hp,
            self_off=self_off,
            norms=norms,
            mnr_params=mnr_params,
            n_boot_se=36,
            retspl_10=sel_retspl,
            ear=args.ear,
            measured_srt_loss_db=args.measured_srt_snr,
        )
        out_png = Path(args.out) if args.out else (out_dir / f"report_{sid}.png")
        out_pdf = out_png.with_suffix(".pdf")
        render_clinical_report_page(rep_ds, norms, out_png, out_pdf, audiogram_topdown=(args.audiogram_orientation == "topdown"))
        print(f"Saved clinical report for {sid} ({rep_ds['phenotype']}) -> {out_png} and {out_pdf}")
        return

    # Standalone Web-App CSV Mode:
    if args.qvcv_csv is not None:
        qvcv_p = Path(args.qvcv_csv)
        c_s1 = load_qvcv_csv_counts(qvcv_p)
        det_ear = args.ear
        if det_ear == "Unspecified":
            parts_u = qvcv_p.stem.upper().split("_")
            if "R" in parts_u or "RIGHT" in parts_u:
                det_ear = "Right"
            elif "L" in parts_u or "LEFT" in parts_u:
                det_ear = "Left"
        raw_aud = (
            np.array([float(x) for x in args.audiogram.split(",")], dtype=float)
            if args.audiogram
            else np.zeros(10, dtype=float)
        )
        if len(raw_aud) == 9:
            raw_aud = np.append(raw_aud, raw_aud[-1])
        aud_10 = (raw_aud - sel_retspl) if (args.audiogram and args.audiogram_units == "spl") else raw_aud
        if args.qcls_elc_csv is not None:
            elc_mat, prof = parse_qcls_elc_csv(Path(args.qcls_elc_csv), aud_10, y_nh)
        else:
            elc_mat = None
            prof = profile_from_audiogram_only(qvcv_p.stem, aud_10)

        aid_dict = {}
        if args.aided_qvcv_csv is not None:
            ap = Path(args.aided_qvcv_csv)
            aid_dict[ap.stem] = load_qvcv_csv_counts(ap)

        rep_single = evaluate_listener_clinical_profile(
            listener_id=qvcv_p.stem,
            group_label="Clinical",
            prof=prof,
            elc_10x10=elc_mat,
            counts_s1=c_s1,
            counts_s2=None,
            practice_pc=None,
            aided_sessions=aid_dict,
            bank=bank,
            hp=hp,
            self_off=self_off,
            norms=norms,
            mnr_params=mnr_params,
            n_boot_se=36,
            retspl_10=sel_retspl,
            ear=det_ear,
            measured_srt_loss_db=args.measured_srt_snr,
        )
        out_png = Path(args.out) if args.out else (out_dir / f"report_{qvcv_p.stem}.png")
        out_pdf = out_png.with_suffix(".pdf")
        render_clinical_report_page(rep_single, norms, out_png, out_pdf, audiogram_topdown=(args.audiogram_orientation == "topdown"))
        print(f"Saved clinical report -> {out_png} and {out_pdf}")
        return

    # Load Aided sessions from QVC_perf.mat (strictly matching HLxxx <-> HAxxx / HPxxx!):
    aided_by_hl: dict[str, dict[str, np.ndarray]] = {}
    perf_mat_path = Path("/Users/neely/STN/2026_Quick_Tests/qtest_dataset/qVCV_analysis/QVC_perf.mat")
    if perf_mat_path.exists():
        mp = loadmat(str(perf_mat_path))
        ids_p = ["".join(ch).strip() for ch in mp["id"]]
        ccms_p = mp["ccms"]
        for idx_p, aid_id in enumerate(ids_p):
            if len(aid_id) >= 3 and aid_id.startswith(("HA", "HP")):
                hl_id = "HL" + aid_id[2:]
                aided_by_hl.setdefault(hl_id, {})[aid_id] = np.asarray(ccms_p[idx_p], dtype=float)

    print("[2/4] Running automated clinical interpretation across all 98 qtest_dataset listeners...", flush=True)
    all_reports: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []
    rep_by_id: dict[str, dict[str, Any]] = {}

    for r in base_listeners:
        sid = r["listener_id"]
        prof = qcls_map[sid]
        elc_i = elc_map.get(sid, None)
        c1 = np.array(r["empirical_s1_counts"], dtype=float)
        c2 = np.array(r["empirical_s2_counts"], dtype=float)
        aid_s = aided_by_hl.get(sid, {})

        # Fast SE (16 bootstrap draws across all 98; 40 draws on the 6 featured cases below):
        rep = evaluate_listener_clinical_profile(
            listener_id=sid,
            group_label=r["archetype"],
            prof=prof,
            elc_10x10=elc_i,
            counts_s1=c1,
            counts_s2=c2,
            practice_pc=float(r["practice_pc"]),
            aided_sessions=aid_s,
            bank=bank,
            hp=hp,
            self_off=self_off,
            norms=norms,
            mnr_params=mnr_params,
            n_boot_se=12,
        )
        all_reports.append(rep)
        rep_by_id[sid] = rep
        csv_rows.append({
            "listener_id": sid,
            "group": rep["group"],
            "phenotype": rep["phenotype"],
            "pta_124_db_hl": rep["pta_124_db_hl"],
            "hf_pta_db_hl": rep["hf_pta_db_hl"],
            "ppta_db_hl": rep["ppta_db_hl"],
            "emp_pc_pooled": rep["emp_pc_pooled"],
            "pc_stage1_aud": rep["pc_stage1_aud"],
            "pc_stage2_periph": rep["pc_stage2_periph"],
            "total_deficit_db": rep["total_deficit_db"],
            "audibility_deficit_db": rep["audibility_deficit_db"],
            "ohc_recruitment_deficit_db": rep["ohc_recruitment_deficit_db"],
            "neural_coding_deficit_db": rep["neural_coding_deficit_db"],
            "neural_coding_se_db": rep["neural_coding_se_db"],
            "qcls_hf_ohc_db": rep["qcls_hf_ohc_db"],
            "hf_erb_broadening_x": rep["hf_erb_broadening_x"],
            "dyn_range_hf_db": rep["dyn_range_hf_db"],
            "cu5_aud_diff_db": rep["cu5_aud_diff_db"],
            "qcls_valid_monotonic": rep["qcls_valid_monotonic"],
            "aided_lab_pc": next((a["aided_pc"] for a in rep["aided_sessions"] if "Lab" in a["aid_type"]), None),
            "aided_personal_pc": next((a["aided_pc"] for a in rep["aided_sessions"] if "Personal" in a["aid_type"]), None),
        })

    write_dict_rows_csv(csv_rows, out_dir / "cohort_98_clinical_interpretation.csv")

    # 6 Featured Real Human Cases from qtest_dataset:
    featured_specs = [
        ("NH013", "case1_NH013_normal_reference"),
        ("HL045", "case2_HL045_hidden_suprathreshold_deficit"),
        ("HL011", "case3_HL011_subclinical_hf_neural_deficit"),
        ("HL068", "case4_HL068_ohc_recruitment_aided_responder"),
        ("HL030", "case5_HL030_skislope_neural_aided_ceiling"),
        ("HL048", "case6_HL048_aided_rollover_overmasking"),
    ]

    print("[3/4] Rendering 1-page clinical reports (PNG + PDF) for 6 archetypal qtest_dataset patients...", flush=True)
    base_by_id = {r["listener_id"]: r for r in base_listeners}
    featured_reports: list[dict[str, Any]] = []
    for sid, slug in featured_specs:
        r = base_by_id[sid]
        rep_hi = evaluate_listener_clinical_profile(
            listener_id=sid,
            group_label=r["archetype"],
            prof=qcls_map[sid],
            elc_10x10=elc_map.get(sid, None),
            counts_s1=np.array(r["empirical_s1_counts"], dtype=float),
            counts_s2=np.array(r["empirical_s2_counts"], dtype=float),
            practice_pc=float(r["practice_pc"]),
            aided_sessions=aided_by_hl.get(sid, {}),
            bank=bank,
            hp=hp,
            self_off=self_off,
            norms=norms,
            mnr_params=mnr_params,
            n_boot_se=36,
            retspl_10=sel_retspl,
        )
        featured_reports.append(rep_hi)

    print("[4/4] Rendering combined 7-page PDF booklet, PNG previews, & normative summary JSON...", flush=True)
    atlas_p = out_dir / "fig_clinical_prototype_overview.png"
    booklet_p = out_dir / "QuickTests_6_Clinical_Case_Examples.pdf"
    buf_booklet = io.BytesIO()
    with PdfPages(buf_booklet) as pdf_bk:
        render_overview_atlas_figure(featured_reports, norms, all_reports, atlas_p, pdf_booklet=pdf_bk)
        for (_, slug), rep_hi in zip(featured_specs, featured_reports):
            png_p = out_dir / f"{slug}.png"
            render_clinical_report_page(
                rep_hi, norms, png_p, out_pdf=None, pdf_booklet=pdf_bk,
                audiogram_topdown=(args.audiogram_orientation == "topdown"),
            )
            print(
                f"  -> {rep_hi['listener_id']} ({rep_hi['phenotype']}): "
                f"Total={rep_hi['total_deficit_db']:+.1f} dB "
                f"[Aud={rep_hi['audibility_deficit_db']:.1f}, OHC={rep_hi['ohc_recruitment_deficit_db']:.1f}, "
                f"Neural={rep_hi['neural_coding_deficit_db']:+.1f} ± {rep_hi['neural_coding_se_db']:.1f} dB] "
                f"Saved {png_p.name}"
            )
    booklet_p.write_bytes(buf_booklet.getvalue())
    print(f"  -> Saved combined 7-page PDF booklet: {booklet_p}")

    summary_json = {
        "normative_nh_cutoffs_100_trials": {
            "n_nh_listeners": 32,
            "n_nh_sessions": 64,
            "nh_median_pc_pct": round(norms.nh_pc_median, 2),
            "nh_5th_pct_pc_pct": round(norms.nh_pc_p05, 2),
            "p90_total_deficit_db_snr": round(norms.p90_total_db, 2),
            "p95_total_deficit_db_snr": round(norms.p95_total_db, 2),
            "p90_audibility_deficit_db_snr": round(norms.p90_aud_db, 2),
            "p90_ohc_deficit_db_snr": round(norms.p90_ohc_db, 2),
            "p90_neural_deficit_db_snr": round(norms.p90_neural_db, 2),
            "p95_neural_deficit_db_snr": round(norms.p95_neural_db, 2),
        },
        "cohort_phenotype_counts_98": {
            ph: sum(1 for r in all_reports if r["phenotype"] == ph)
            for ph in sorted({r["phenotype"] for r in all_reports})
        },
        "featured_cases": [
            {
                k: v
                for k, v in rep.items()
                if k not in ("elc_10x10", "smoothed_ccm", "raw_counts")
            }
            for rep in featured_reports
        ],
    }
    (out_dir / "clinical_prototype_summary.json").write_text(json.dumps(summary_json, indent=2))
    print(f"Done! Reports written to {out_dir}")


if __name__ == "__main__":
    main()
