"""
qvcv_biophysical_model.py

Realistic biophysical & spectro-temporal listener model for the qVCV test,
unifying:
  1. Waveform-driven Spectro-Temporal Cue Extraction directly from the 20 clear
     qVCV stimuli (Stimuli/practice/VCV_a*a_{1,2}_60SNR.wav) and LTASS masker:
       - 3 intervocalic consonant epochs (Core, Release/Burst, CV Transition)
         x 10 ERB cochlear channels (250..8000 Hz)
       - Flanking /a/ vowel ERB spectrum (driving upward spread of masking)
       - Waveform F0 pitch-periodicity autocorrelation (voicing cue) and
         temporal-envelope continuity vs. plosive transience (manner cue)
  2. Paired qCLS Equal-Loudness-Contour Coupling (Neely slct_lstn.m framework):
       - Partitions 10-frequency threshold elevation into Outer-Hair-Cell (OHC)
         active cochlear gain loss (from loudness recruitment) vs. Inner-Hair-Cell
         (IHC) / passive sensitivity loss across 96/98 paired human listeners
       - Drives OHC auditory filter broadening (1x..2.8x ERB) and upward spread
         of masking (USOM) from low-frequency /a/ vowel and noise energy
       - Optional multi-channel WDRC hearing-aid stage (audibility restoration +
         compression-induced envelope contrast reduction + HF roll-off)
  3. Asymmetric Luce Choice + Audible-Template Decision Stage (+ 1.5% Lapse Floor):
       - Captures human directional confusions (V->Z 29:1, K->T 4.6:1, S->T 3.5:1,
         D->G 2.6:1) from band-specific cue loss + coronal markedness priors +
         listener-specific '???' catch-all decision criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy.io import loadmat, wavfile

try:
    from .qvcv_simulator import (
        CONSONANT_SNR_DB,
        DISPLAY_CONSONANTS,
        N_RESPONSES,
        N_STIMULI,
        STIMULI_CONSONANTS,
        VirtualListener,
        normalize_rows,
    )
except ImportError:
    from qvcv_simulator import (
        CONSONANT_SNR_DB,
        DISPLAY_CONSONANTS,
        N_RESPONSES,
        N_STIMULI,
        STIMULI_CONSONANTS,
        VirtualListener,
        normalize_rows,
    )

AUD_FREQS_HZ: np.ndarray = np.array(
    [250.0, 500.0, 750.0, 1000.0, 1500.0, 2000.0, 3000.0, 4000.0, 6000.0, 8000.0],
    dtype=float,
)
ANSI_RETSPL_DB: np.ndarray = np.array(
    [18.0, 11.0, 8.5, 7.5, 7.5, 9.0, 10.0, 9.5, 15.5, 13.0],
    dtype=float,
)
REF_SNR_ARR: np.ndarray = np.array([CONSONANT_SNR_DB[c] for c in STIMULI_CONSONANTS], dtype=float)
DEFAULT_STIMULI_DIR: Path = Path(__file__).resolve().parent.parent / "Stimuli"
DEFAULT_QCLS_MAT: Path = Path("/Users/neely/STN/2026_Quick_Tests/qtest_dataset/qCLS_analysis/qcls_data.mat")


# --------------------------------------------------------------------------
# 1. Waveform Spectro-Temporal Cue Bank
# --------------------------------------------------------------------------

@dataclass
class SpectroTemporalVCVBank:
    """Acoustic cue templates measured directly from the 20 clear qVCV .wav waveforms."""
    cue_spl: np.ndarray          # (10_cons, 2_tok, 3_epoch, 10_erb) dB SPL
    vowel_spl: np.ndarray        # (10_cons, 2_tok, 10_erb) dB SPL
    ltass_0snr_spl: np.ndarray   # (10_erb,) dB SPL at 0 dB nominal VCV SNR
    voicing_acorr: np.ndarray    # (10_cons, 2_tok) in [0, 1] F0 periodicity in core
    continuant_idx: np.ndarray   # (10_cons, 2_tok) in [0, 1] envelope continuity vs plosive burst
    timing_ms: dict[str, tuple[int, int, int]]  # V1_peak, C_core, V2_peak (ms) for token 1

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "aud_freqs_hz": AUD_FREQS_HZ.tolist(),
            "ltass_0snr_spl": np.round(self.ltass_0snr_spl, 3).tolist(),
            "cue_spl": np.round(self.cue_spl, 3).tolist(),
            "vowel_spl": np.round(self.vowel_spl, 3).tolist(),
            "voicing_acorr": np.round(self.voicing_acorr, 4).tolist(),
            "continuant_idx": np.round(self.continuant_idx, 4).tolist(),
            "timing_ms_token1": {k: list(v) for k, v in self.timing_ms.items()},
        }


def extract_vcv_acoustic_bank(stim_dir: Path = DEFAULT_STIMULI_DIR) -> SpectroTemporalVCVBank:
    """
    Extract 3-epoch x 10-ERB spectro-temporal consonant cues, vowel masking spectra,
    F0 voicing periodicity, and envelope continuity directly from qVCV .wav files.
    """
    ltass_band_pwr: list[list[float]] = []
    for c in ("k", "s", "sh", "t", "z"):
        for tok in (1, 2):
            p_path = stim_dir / "practice" / f"VCV_a{c}a_{tok}_60SNR.wav"
            t_path = stim_dir / "test" / f"VCV_a{c}a_{tok}_0SNR.wav"
            sr, p_dat = wavfile.read(str(p_path))
            _, t_dat = wavfile.read(str(t_path))
            scale = float(np.iinfo(p_dat.dtype).max)
            n_wav = (t_dat.astype(float) - p_dat.astype(float)) / scale
            freqs = np.fft.rfftfreq(len(n_wav), 1.0 / float(sr))
            psd = (np.abs(np.fft.rfft(n_wav)) ** 2) / (len(n_wav) ** 2) * 2.0
            bp = []
            for f0 in AUD_FREQS_HZ:
                erb = 24.7 * (4.37 * (f0 / 1000.0) + 1.0)
                m = (freqs >= f0 - 0.5 * erb) & (freqs <= f0 + 0.5 * erb)
                bp.append(60.0 + 10.0 * np.log10(max(float(np.sum(psd[m])), 1e-20) / (0.01 ** 2)))
            ltass_band_pwr.append(bp)
    ltass_0snr_spl = np.mean(ltass_band_pwr, axis=0)

    cue_spl = np.zeros((N_STIMULI, 2, 3, 10), dtype=float)
    vowel_spl = np.zeros((N_STIMULI, 2, 10), dtype=float)
    voicing_acorr = np.zeros((N_STIMULI, 2), dtype=float)
    continuant_idx = np.zeros((N_STIMULI, 2), dtype=float)
    timing_ms: dict[str, tuple[int, int, int]] = {}

    for ci, c_up in enumerate(STIMULI_CONSONANTS):
        c = c_up.lower()
        for ti, tok in enumerate((1, 2)):
            sr, p_dat = wavfile.read(str(stim_dir / "practice" / f"VCV_a{c}a_{tok}_60SNR.wav"))
            x = p_dat.astype(float) / float(np.iinfo(p_dat.dtype).max)
            n_win = int(0.020 * sr)
            n_hop = int(0.010 * sr)
            hann = np.hanning(n_win)
            h_corr = float(np.mean(hann ** 2))
            n_fr = (len(x) - n_win) // n_hop + 1
            freqs = np.fft.rfftfreq(n_win, 1.0 / float(sr))
            b_masks = [
                (freqs >= f0 - 0.5 * 24.7 * (4.37 * (f0 / 1000.0) + 1.0))
                & (freqs <= f0 + 0.5 * 24.7 * (4.37 * (f0 / 1000.0) + 1.0))
                for f0 in AUD_FREQS_HZ
            ]
            spec = np.zeros((n_fr, 10), dtype=float)
            for fr in range(n_fr):
                seg = x[fr * n_hop : fr * n_hop + n_win] * hann
                psd = (np.abs(np.fft.rfft(seg)) ** 2) / (n_win ** 2 * h_corr) * 2.0
                for bi, bm in enumerate(b_masks):
                    spec[fr, bi] = 60.0 + 10.0 * np.log10(max(float(np.sum(psd[bm])), 1e-20) / (0.01 ** 2))

            # Active speech bounds & intervocalic F1/F2 valley between V1 and V2:
            v_db = 10.0 * np.log10(np.sum(10.0 ** (spec[:, 1:5] / 10.0), axis=1))
            active = np.where(v_db >= np.max(v_db) - 18.0)[0]
            f_on, f_off = int(active[0]), int(active[-1])
            f_mid = (f_on + f_off) // 2
            pk1 = (f_on + 2) + int(np.argmax(v_db[f_on + 2 : f_mid - 1]))
            pk2 = (f_mid + 2) + int(np.argmax(v_db[f_mid + 2 : f_off - 2]))
            c_cen = pk1 + int(np.argmin(v_db[pk1 : pk2 + 1]))

            def mpwr(lo: int, hi: int) -> np.ndarray:
                lo_c = max(0, min(n_fr - 1, lo))
                hi_c = max(lo_c + 1, min(n_fr, hi))
                return 10.0 * np.log10(np.mean(10.0 ** (spec[lo_c:hi_c] / 10.0), axis=0))

            cue_spl[ci, ti, 0] = mpwr(c_cen - 3, c_cen + 3)    # Core closure/murmur/frication
            cue_spl[ci, ti, 1] = mpwr(c_cen + 1, c_cen + 6)    # Release burst / late frication
            cue_spl[ci, ti, 2] = mpwr(c_cen + 4, c_cen + 10)   # CV formant transition
            vowel_spl[ci, ti] = 10.0 * np.log10(
                0.5 * (10.0 ** (spec[pk1] / 10.0) + 10.0 ** (spec[pk2] / 10.0))
            )

            # Waveform F0 periodicity (normalized autocorrelation in 85..300 Hz) in 60-ms core:
            s0 = max(0, (c_cen - 2) * n_hop)
            s1 = min(len(x), (c_cen + 4) * n_hop)
            seg_core = x[s0:s1] - np.mean(x[s0:s1])
            denom = float(np.sum(seg_core ** 2))
            if denom > 1e-16:
                lag_min, lag_max = int(sr / 300.0), int(sr / 85.0)
                ac = [
                    float(np.sum(seg_core[:-lg] * seg_core[lg:])) / denom
                    for lg in range(lag_min, min(lag_max, len(seg_core) // 2))
                ]
                # Weight periodicity by presence of low-frequency voice-bar energy (>24 dB SPL):
                lf_gate = 1.0 / (1.0 + np.exp(-(cue_spl[ci, ti, 0, :2].mean() - 25.5) / 2.5))
                voicing_acorr[ci, ti] = float(np.clip(max(ac, default=0.0) * lf_gate, 0.0, 1.0))

            # Temporal continuity vs plosive closure-to-burst step:
            # Sustained fricatives/nasals (S, SH, Z, V, N) maintain smooth energy across core->release,
            # whereas plosives (B, D, G, K, T) have a silent/damped closure followed by abrupt release!
            broad_env = 10.0 * np.log10(np.sum(10.0 ** (spec[max(0, c_cen - 4) : min(n_fr, c_cen + 7)] / 10.0), axis=1))
            dip_depth = float(np.max(broad_env) - np.min(broad_env))
            # Lower dip_depth / sustained energy -> higher continuant index in [0, 1]:
            continuant_idx[ci, ti] = float(1.0 / (1.0 + np.exp((dip_depth - 14.5) / 3.5)))

            if ti == 0:
                timing_ms[c_up] = (pk1 * 10, c_cen * 10, pk2 * 10)

    return SpectroTemporalVCVBank(
        cue_spl=cue_spl,
        vowel_spl=vowel_spl,
        ltass_0snr_spl=ltass_0snr_spl,
        voicing_acorr=voicing_acorr,
        continuant_idx=continuant_idx,
        timing_ms=timing_ms,
    )


# --------------------------------------------------------------------------
# 2. qCLS Cochlear Gain & Filter-Broadening Profile
# --------------------------------------------------------------------------

@dataclass
class QCLSCochlearProfile:
    """10-frequency peripheral cochlear profile derived from qCLS + audiogram."""
    listener_id: str
    has_measured_qcls: bool
    audiogram_hl_db: np.ndarray     # (10,) pure-tone thresholds (dB HL)
    hl_ohc_db: np.ndarray           # (10,) active OHC gain loss (0..55 dB)
    hl_ihc_db: np.ndarray           # (10,) passive / IHC sensitivity loss (>=0 dB)
    cls_speech_slope: np.ndarray    # (10,) loudness-growth compression ratio around 50-85 phon

    def erb_broadening_factor(self, gamma_ohc: float = 1.8) -> np.ndarray:
        """Cochlear filter bandwidth multiplier w_B(f) = 1 + gamma_ohc * (HL_OHC / 50)."""
        return 1.0 + gamma_ohc * (np.clip(self.hl_ohc_db, 0.0, 55.0) / 50.0)


def profile_from_audiogram_only(listener_id: str, audiogram_hl_db: np.ndarray | list[float]) -> QCLSCochlearProfile:
    """Moore & Glasberg (1997) ~70% OHC / 30% IHC fallback when qCLS contours are unavailable."""
    aud = np.asarray(audiogram_hl_db, dtype=float)
    tot = np.maximum(aud, 0.0)
    ohc = np.minimum(0.70 * tot, 55.0)
    ihc = np.maximum(tot - ohc, 0.0)
    slope = 1.0 + 0.025 * ohc
    return QCLSCochlearProfile(
        listener_id=listener_id,
        has_measured_qcls=False,
        audiogram_hl_db=aud,
        hl_ohc_db=ohc,
        hl_ihc_db=ihc,
        cls_speech_slope=slope,
    )


def load_qcls_cochlear_profiles(
    listeners_meta: list[dict[str, Any]],
    qcls_mat_path: Path = DEFAULT_QCLS_MAT,
) -> dict[str, QCLSCochlearProfile]:
    """
    Extract paired 10-frequency OHC vs. IHC loss and loudness-growth slopes
    directly from ``qcls_data.mat`` following Neely ``slct_lstn.m``.
    """
    out: dict[str, QCLSCochlearProfile] = {}
    if not qcls_mat_path.exists():
        for r in listeners_meta:
            out[r["listener_id"]] = profile_from_audiogram_only(r["listener_id"], r["audiogram_db"])
        return out

    qc = loadmat(str(qcls_mat_path), squeeze_me=True)
    q_ids = [str(x["name"]).strip() for x in qc["id"]]
    elcb = qc["elcb"]
    tcnd = qc["tcnd"]
    nnh = int(qc["nnh"])

    nh_c1 = [np.nanmean(elcb[i, tcnd[i] == 1, :, :], axis=0) for i in range(nnh) if np.sum(tcnd[i] == 1) > 0]
    y_nh = np.nanmean(nh_c1, axis=0)  # (10_freq, 10_lc) in dB SPL
    q_by_id = {qid: idx for idx, qid in enumerate(q_ids)}

    for r in listeners_meta:
        sid = r["listener_id"]
        aud = np.asarray(r["audiogram_db"], dtype=float)
        tot = np.maximum(aud, 0.0)
        if sid in q_by_id:
            qi = q_by_id[sid]
            kk = tcnd[qi] == 1
            if np.sum(kk) > 0:
                yi = np.nanmean(elcb[qi, kk, :, :], axis=0)
                if not np.all(np.isnan(yi)):
                    loss_low = np.nanmean(yi[:, 0:2] - y_nh[:, 0:2], axis=1)
                    loss_high = np.nanmean(yi[:, 7:9] - y_nh[:, 7:9], axis=1)
                    rec = np.clip(loss_low - np.maximum(loss_high, 0.0), 0.0, 55.0)
                    rec = np.where(np.isnan(rec), np.minimum(0.70 * tot, 55.0), rec)
                    ohc = np.where(tot >= 10.0, np.clip(rec, 0.25 * tot, np.minimum(tot, 55.0)), 0.70 * tot)
                    ohc = np.clip(ohc, 0.0, 55.0)
                    ihc = np.maximum(tot - ohc, 0.0)

                    dy_nh = y_nh[:, 5] - y_nh[:, 2]
                    dy_i = np.maximum(yi[:, 5] - yi[:, 2], 8.0)
                    sl = np.where(np.isnan(dy_i), 1.0, dy_nh / dy_i)
                    out[sid] = QCLSCochlearProfile(
                        listener_id=sid,
                        has_measured_qcls=True,
                        audiogram_hl_db=aud,
                        hl_ohc_db=ohc,
                        hl_ihc_db=ihc,
                        cls_speech_slope=np.clip(sl, 0.7, 3.5),
                    )
                    continue
        out[sid] = profile_from_audiogram_only(sid, aud)

    return out


# --------------------------------------------------------------------------
# 3. Peripheral Masking & Asymmetric Audible-Template Choice Engine
# --------------------------------------------------------------------------

@dataclass
class BiophysicalHyperparams:
    """Global biophysical & directional choice parameters calibrated to the human cohort."""
    gamma_ohc: float = 1.8             # Filter broadening at 50 dB OHC loss -> 2.8x ERB
    usom_vowel_att_db: float = 16.0    # Flanking /a/ forward/simultaneous masking offset (dB)
    thr_offset_db: float = -4.0        # Internal ERB detection threshold offset (dB)
    spec_scale: float = 13.5           # Spectro-temporal template sensitivity
    voice_scale: float = 3.8           # Waveform F0 periodicity weight
    cont_scale: float = 2.4            # Waveform continuant-vs-plosive envelope weight
    lapse_rate: float = 0.015          # 1.5% SNR-independent human attention lapse floor
    # Directional cue-loss & unmarked coronal choice matrix A[stim_c, resp_j] (10x10, zero diag):
    directional_choice_bias: Optional[np.ndarray] = None

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "gamma_ohc": self.gamma_ohc,
            "usom_vowel_att_db": self.usom_vowel_att_db,
            "thr_offset_db": self.thr_offset_db,
            "spec_scale": self.spec_scale,
            "voice_scale": self.voice_scale,
            "cont_scale": self.cont_scale,
            "lapse_rate": self.lapse_rate,
            "directional_choice_bias": (
                np.round(self.directional_choice_bias, 4).tolist()
                if self.directional_choice_bias is not None
                else None
            ),
        }


def compute_peripheral_template_features(
    bank: SpectroTemporalVCVBank,
    snr_per_consonant: np.ndarray,     # (10,)
    audiogram_hl_n10: np.ndarray,      # (N, 10)
    hl_ohc_n10: np.ndarray,            # (N, 10)
    hp: BiophysicalHyperparams,
    ha_insertion_gain_n10: Optional[np.ndarray] = None,
    ha_wdrc_smear_factor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized biophysical peripheral front-end across N listeners simultaneously.

    Returns:
      neg_dist:    (N, 10_stim, 2_tok, 10_cand) negative perceptual distance <= 0 (0 on self when clean)
      aud_cue_db:  (N, 10_stim, 2_tok) audible consonant sensation above peripheral masker (dB)
      hf_noise_aud:(N, 10_stim, 2_tok) audible external HF noise above threshold (drives V->Z hiss illusion)
    """
    n_subj = audiogram_hl_n10.shape[0]
    hl_tot = np.maximum(audiogram_hl_n10, 0.0)
    w_erb = 1.0 + hp.gamma_ohc * (np.clip(hl_ohc_n10, 0.0, 55.0) / 50.0)
    in_band_pen = 10.0 * np.log10(w_erb)

    gain_n10 = np.zeros_like(hl_tot) if ha_insertion_gain_n10 is None else np.asarray(ha_insertion_gain_n10, dtype=float)
    # Effective threshold in input-referred dB SPL is reduced by insertion gain:
    thr_spl = ANSI_RETSPL_DB[None, :] + hl_tot - gain_n10 + hp.thr_offset_db  # (N, 10)

    log2_f = np.log2(AUD_FREQS_HZ)
    df_oct = log2_f[None, :] - log2_f[:, None]  # (10_src, 10_dst)
    lower_mask = (df_oct > 0.05)[None, :, :]
    s_low = 65.0 / w_erb  # (N, 10_dst)

    proto_spl = 10.0 * np.log10(np.mean(10.0 ** (bank.cue_spl / 10.0), axis=1))  # (10_cand, 3, 10)
    proto_voice = np.mean(bank.voicing_acorr, axis=1)                            # (10_cand,)
    proto_cont = np.mean(bank.continuant_idx, axis=1)                            # (10_cand,)
    ep_wt = np.array([0.40, 0.40, 0.20])[None, :, None]

    neg_dist = np.zeros((n_subj, N_STIMULI, 2, N_STIMULI), dtype=float)
    aud_cue_db = np.zeros((n_subj, N_STIMULI, 2), dtype=float)
    hf_noise_aud = np.zeros((n_subj, N_STIMULI, 2), dtype=float)

    for c in range(N_STIMULI):
        snr_val = float(snr_per_consonant[c])
        # Fast WDRC compression amplifies background noise in dips by ~1.5-3 dB when smear > 0:
        eff_snr = snr_val - 2.5 * ha_wdrc_smear_factor
        n_ext = bank.ltass_0snr_spl[None, :] - eff_snr + in_band_pen  # (N, 10)

        for m in range(2):
            v_src = np.maximum(
                bank.vowel_spl[c, m][None, :] - hp.usom_vowel_att_db,
                bank.ltass_0snr_spl[None, :] - eff_snr,
            )
            att = s_low[:, None, :] * np.maximum(df_oct[None, :, :], 0.0)
            usom_pwr = np.sum(
                np.where(lower_mask, 10.0 ** (np.clip(v_src[:, :, None] - att, -40.0, 120.0) / 10.0), 0.0),
                axis=1,
            )
            tot_mask_pwr = (
                10.0 ** (np.clip(n_ext, -40.0, 120.0) / 10.0)
                + usom_pwr
                + 10.0 ** (np.clip(thr_spl, -40.0, 120.0) / 10.0)
            )
            m_db = 10.0 * np.log10(tot_mask_pwr)  # (N, 10)

            sp_pwr = 10.0 ** (bank.cue_spl[c, m][None, :, :] / 10.0)
            y_obs = 10.0 * np.log10(sp_pwr + tot_mask_pwr[:, None, :])  # (N, 3, 10)
            aud_sens = np.maximum(y_obs - m_db[:, None, :], 0.0)
            aud_cue_db[:, c, m] = np.sum(ep_wt * aud_sens, axis=(1, 2))

            # Audible external HF noise sensation (3-8 kHz) relative to listener threshold:
            hf_noise_aud[:, c, m] = np.mean(
                np.maximum((bank.ltass_0snr_spl[None, 6:] - eff_snr) - thr_spl[:, 6:], 0.0),
                axis=1,
            )

            pr_pwr = 10.0 ** (proto_spl[None, :, :, :] / 10.0)
            t_exp = 10.0 * np.log10(pr_pwr + tot_mask_pwr[:, None, None, :])
            d_spec = np.sum(ep_wt[:, None, :, :] * ((y_obs[:, None, :, :] - t_exp) ** 2), axis=(2, 3)) / 100.0

            # Voicing periodicity & envelope continuity visibility scale smoothly with low/broadband audibility:
            lf_snr_margin = np.mean(bank.cue_spl[c, m, 0, :3][None, :] - m_db[:, :3], axis=1)  # (N,)
            vis_voice = 1.0 / (1.0 + np.exp(-np.clip((lf_snr_margin + 8.0) / 4.0, -20.0, 20.0)))
            vis_cont = (1.0 - 0.35 * ha_wdrc_smear_factor) / (
                1.0 + np.exp(-np.clip((aud_cue_db[:, c, m] - 3.0) / 3.5, -20.0, 20.0))
            )

            d_voice = vis_voice[:, None] * ((bank.voicing_acorr[c, m] - proto_voice[None, :]) ** 2)
            d_cont = vis_cont[:, None] * ((bank.continuant_idx[c, m] - proto_cont[None, :]) ** 2)

            neg_dist[:, c, m, :] = -(
                hp.spec_scale * d_spec + hp.voice_scale * d_voice + hp.cont_scale * d_cont
            )

    return neg_dist, aud_cue_db, hf_noise_aud


class BiophysicalVirtualListener:
    """
    Mechanistic virtual listener combining:
      - Waveform spectro-temporal VCV cue templates (3 epochs x 10 ERB bands + F0 + continuity)
      - qCLS-derived OHC filter broadening & upward spread of masking vs. IHC threshold loss
      - Neural/suprathreshold coding efficiency + directional cue-loss choice biases + 1.5% lapse floor.

    Implements the full MultiSNRVirtualListener & VirtualListener interface so all
    existing qVCV trackers and simulation scripts run unmodified.
    """

    def __init__(
        self,
        listener_id: str,
        archetype: str,
        description: str,
        cochlear_profile: QCLSCochlearProfile,
        bank: SpectroTemporalVCVBank,
        hyperparams: BiophysicalHyperparams,
        neural_efficiency_per_cons: np.ndarray,  # (10,) multiplicative coding fidelity (>0; 1.0 = NH norm)
        self_clarity_offset: np.ndarray,         # (10,) consonant-specific supra-threshold clarity offset
        unk_criterion: float = -2.2,             # '???' catch-all log-odds threshold when audible cues vanish
        ha_insertion_gain_db: Optional[np.ndarray] = None,
        ha_wdrc_smear: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.listener_id = listener_id
        self.archetype = archetype
        self.description = description
        self.cochlear_profile = cochlear_profile
        self.bank = bank
        self.hp = hyperparams
        self.neural_efficiency_per_cons = np.asarray(neural_efficiency_per_cons, dtype=float)
        self.self_clarity_offset = np.asarray(self_clarity_offset, dtype=float)
        self.unk_criterion = float(unk_criterion)
        self.ha_insertion_gain_db = None if ha_insertion_gain_db is None else np.asarray(ha_insertion_gain_db, dtype=float)
        self.ha_wdrc_smear = float(ha_wdrc_smear)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._cache: dict[tuple[int, int, float], np.ndarray] = {}

    def clone_with_rng(self, rng: np.random.Generator) -> "BiophysicalVirtualListener":
        return BiophysicalVirtualListener(
            listener_id=self.listener_id,
            archetype=self.archetype,
            description=self.description,
            cochlear_profile=self.cochlear_profile,
            bank=self.bank,
            hyperparams=self.hp,
            neural_efficiency_per_cons=self.neural_efficiency_per_cons,
            self_clarity_offset=self.self_clarity_offset,
            unk_criterion=self.unk_criterion,
            ha_insertion_gain_db=self.ha_insertion_gain_db,
            ha_wdrc_smear=self.ha_wdrc_smear,
            rng=rng,
        )

    def _compute_all_at_snr_vec(self, snr_vec_10: np.ndarray) -> np.ndarray:
        """Return (10_stim, 2_tok, 11_resp) confusion probability tensor."""
        neg_dist, aud_cue, _ = compute_peripheral_template_features(
            bank=self.bank,
            snr_per_consonant=snr_vec_10,
            audiogram_hl_n10=self.cochlear_profile.audiogram_hl_db[None, :],
            hl_ohc_n10=self.cochlear_profile.hl_ohc_db[None, :],
            hp=self.hp,
            ha_insertion_gain_n10=None if self.ha_insertion_gain_db is None else self.ha_insertion_gain_db[None, :],
            ha_wdrc_smear_factor=self.ha_wdrc_smear,
        )
        nd = neg_dist[0]     # (10, 2, 10)
        ac = aud_cue[0]      # (10, 2)

        logits = np.zeros((N_STIMULI, 2, N_RESPONSES), dtype=float)
        eff = self.neural_efficiency_per_cons[:, None, None]  # (10, 1, 1)
        logits[:, :, :N_STIMULI] = eff * nd

        # Audible target self-recognition clarity grows with audible cue sensation ac[c, m]:
        eye_10_11 = np.eye(N_STIMULI, N_RESPONSES, dtype=float)[:, None, :]
        diag_boost = self.self_clarity_offset[:, None] + 0.14 * self.neural_efficiency_per_cons[:, None] * ac
        logits += diag_boost[:, :, None] * eye_10_11

        # Directional cue-loss & unmarked coronal choice priors (modulated as uncertainty grows):
        if self.hp.directional_choice_bias is not None:
            # Stronger directional bias when audible target sensation is weak/moderate:
            uncert_gate = 1.0 / (1.0 + np.exp((ac - 22.0) / 7.5))  # (10, 2) in (0, 1)
            logits[:, :, :N_STIMULI] += (
                (0.35 + 0.65 * uncert_gate[:, :, None]) * self.hp.directional_choice_bias[:, None, :]
            )

        # '???' catch-all option activates when audible consonant sensation collapses toward 0:
        low_evid = 1.0 / (1.0 + np.exp((ac - 8.0) / 4.0))
        logits[:, :, N_STIMULI] = self.unk_criterion + 2.2 * low_evid

        logits -= np.max(logits, axis=2, keepdims=True)
        probs = np.exp(logits)
        probs /= np.sum(probs, axis=2, keepdims=True)

        # Stationary 1.5% human attention-lapse floor (distributes slips across alternatives):
        lam = self.hp.lapse_rate
        if lam > 0.0:
            lapse_dist = np.full(N_RESPONSES, 1.0 / float(N_RESPONSES), dtype=float)
            probs = (1.0 - lam) * probs + lam * lapse_dist[None, None, :]

        return probs

    def response_probs(
        self,
        stim_idx: int,
        snr_db: float,
        token_idx: Optional[int] = None,
    ) -> np.ndarray:
        snr_key = round(float(snr_db), 3)
        if token_idx is None:
            p0 = self.response_probs(stim_idx, snr_key, token_idx=0)
            p1 = self.response_probs(stim_idx, snr_key, token_idx=1)
            return 0.5 * (p0 + p1)

        ck = (stim_idx, token_idx, snr_key)
        if ck not in self._cache:
            s_vec = REF_SNR_ARR.copy()
            s_vec[stim_idx] = snr_key
            all_p = self._compute_all_at_snr_vec(s_vec)
            self._cache[(stim_idx, 0, snr_key)] = all_p[stim_idx, 0]
            self._cache[(stim_idx, 1, snr_key)] = all_p[stim_idx, 1]
        return self._cache[ck]

    def matrix_at_snrs(self, snr_per_consonant: np.ndarray | list[float]) -> np.ndarray:
        s_vec = np.asarray(snr_per_consonant, dtype=float)
        all_p = self._compute_all_at_snr_vec(s_vec)
        return normalize_rows(0.5 * (all_p[:, 0, :] + all_p[:, 1, :]))

    def pooled_matrix_for_design(self, snr_grid_10xK: np.ndarray) -> np.ndarray:
        grid = np.asarray(snr_grid_10xK, dtype=float)
        acc = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
        for k in range(grid.shape[1]):
            acc += self.matrix_at_snrs(grid[:, k])
        return normalize_rows(acc / float(grid.shape[1]))

    def sample_response(self, stim_idx: int, snr_db: float, token_idx: Optional[int] = None) -> int:
        p = self.response_probs(stim_idx, snr_db, token_idx=token_idx)
        return int(self.rng.choice(N_RESPONSES, p=p))

    def to_fixed_snr_listener(self) -> VirtualListener:
        """Export a drop-in fixed-SNR VirtualListener at standard qVCV Ref SNRs."""
        t0_mat = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
        t1_mat = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
        all_p = self._compute_all_at_snr_vec(REF_SNR_ARR)
        t0_mat[:] = all_p[:, 0, :]
        t1_mat[:] = all_p[:, 1, :]
        mean_mat = normalize_rows(0.5 * (t0_mat + t1_mat))
        return VirtualListener(
            listener_id=self.listener_id,
            gt_matrix=mean_mat,
            archetype=self.archetype,
            description=self.description,
            token_matrices=np.stack([t0_mat, t1_mat], axis=0),
            rng=self.rng,
        )
