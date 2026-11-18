"""
qvcv_multisnr_simulator.py

SNR-dependent virtual-listener model for the qVCV Consonant Confusion Test,
grounded in Miller & Nicely (1955) and Phatak & Allen (2007).

Models each virtual listener's full 3D confusion probability function
P(Response = j | Stimulus = i, SNR_dB) continuously over SNR in [-18, +30] dB:
  - Low-frequency voicing & manner cues emerge at lower SNRs than spectral place cues.
  - Consonant-specific reference thresholds align with qVCV's baked-in reference SNRs
    (B: +18, V: +18, N: +15, D: +4, G: +3, K/S/SH/T/Z: 0 dB), so at Delta_SNR = 0 dB
    normal-hearing recognition sits around ~85-88% while hearing-impaired profiles
    span realistic audibility shifts and supra-threshold distortion plateaus.
  - Supports talker-token (_1_ vs _2_) differences and idiosyncratic VCV asymmetries.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    from .qvcv_simulator import (
        CONSONANT_SNR_DB,
        DISPLAY_CONSONANTS,
        N_RESPONSES,
        N_STIMULI,
        PHONETIC_FEATURES,
        RESP_INDEX,
        STIM_INDEX,
        STIMULI_CONSONANTS,
        normalize_rows,
    )
except ImportError:
    from qvcv_simulator import (
        CONSONANT_SNR_DB,
        DISPLAY_CONSONANTS,
        N_RESPONSES,
        N_STIMULI,
        PHONETIC_FEATURES,
        RESP_INDEX,
        STIM_INDEX,
        STIMULI_CONSONANTS,
        normalize_rows,
    )

# Standard evaluation SNR offset grid (relative to consonant reference SNR) for
# measuring full psychometric-surface recovery across [-9, +9] dB
EVAL_REL_SNRS_DB: np.ndarray = np.array([-9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0], dtype=float)


@dataclass
class MultiSNRArchetypeSpec:
    archetype: str
    description: str
    # Listener SNR threshold shift (dB) relative to NH baseline, for LF and HF consonants
    snr_shift_lf_db: float
    snr_shift_hf_db: float
    # Logistic growth slope (per dB)
    slope_per_db: float
    # Supra-threshold maximum self-identification log-odds ceiling (lower = distortion plateau)
    max_self_lf: float
    max_self_hf: float
    # Maximum phonetic feature weights at moderate/high SNR
    w_voice: float
    w_manner: float
    w_place: float
    w_sibilant: float
    # Catch-all '???' propensity at low SNR and high-SNR floor
    unk_low_snr: float
    unk_high_snr: float
    # Archetype-specific asymmetric confusions
    asymmetry: dict[tuple[str, str], float]


class MultiSNRVirtualListener:
    """
    Virtual listener with a continuous SNR-dependent confusion matrix
    P(R = j | S = i, snr_db) for each consonant i in 0..9 and token in {0, 1}.
    """

    def __init__(
        self,
        listener_id: str,
        archetype: str,
        description: str,
        tau_50_db: np.ndarray,           # (10,) consonant 50%-transition SNR center (dB)
        slope_db: np.ndarray,            # (10,) consonant psychometric growth rate (1/dB)
        max_self_logit: np.ndarray,      # (10,) asymptotic target-ID logit at high SNR
        feat_weights: np.ndarray,        # (4,) [w_voice, w_manner, w_place, w_sibilant]
        unk_params: np.ndarray,          # (2,) [unk_low_snr, unk_high_snr]
        asymmetry_matrix: np.ndarray,    # (10, 11) static off-diagonal/response bias logits
        token_logit_offsets: np.ndarray, # (2, 10, 11) subtle _1_ vs _2_ talker differences
        rng: Optional[np.random.Generator] = None,
    ):
        self.listener_id = listener_id
        self.archetype = archetype
        self.description = description
        self.tau_50_db = np.asarray(tau_50_db, dtype=float)
        self.slope_db = np.asarray(slope_db, dtype=float)
        self.max_self_logit = np.asarray(max_self_logit, dtype=float)
        self.feat_weights = np.asarray(feat_weights, dtype=float)
        self.unk_params = np.asarray(unk_params, dtype=float)
        self.asymmetry_matrix = np.asarray(asymmetry_matrix, dtype=float)
        self.token_logit_offsets = np.asarray(token_logit_offsets, dtype=float)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._cache: dict[tuple[int, int, float], np.ndarray] = {}

    def clone_with_rng(self, rng: np.random.Generator) -> "MultiSNRVirtualListener":
        return MultiSNRVirtualListener(
            listener_id=self.listener_id,
            archetype=self.archetype,
            description=self.description,
            tau_50_db=self.tau_50_db,
            slope_db=self.slope_db,
            max_self_logit=self.max_self_logit,
            feat_weights=self.feat_weights,
            unk_params=self.unk_params,
            asymmetry_matrix=self.asymmetry_matrix,
            token_logit_offsets=self.token_logit_offsets,
            rng=rng,
        )

    def response_probs(
        self,
        stim_idx: int,
        snr_db: float,
        token_idx: Optional[int] = None,
    ) -> np.ndarray:
        """
        Return length-11 probability vector P(R | S = stim_idx, snr_db).
        If ``token_idx`` is None, returns the exact average across _1_ and _2_ tokens.
        """
        snr_key = round(float(snr_db), 3)
        if token_idx is None:
            p0 = self.response_probs(stim_idx, snr_key, token_idx=0)
            p1 = self.response_probs(stim_idx, snr_key, token_idx=1)
            return 0.5 * (p0 + p1)

        cache_key = (stim_idx, token_idx, snr_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        sc = STIMULI_CONSONANTS[stim_idx]
        sf = PHONETIC_FEATURES[sc]
        tau = self.tau_50_db[stim_idx]
        k = self.slope_db[stim_idx]

        # Normalized psychometric activations in [0, 1] at different SNR offsets:
        # Voicing & manner emerge ~5 dB earlier in noise than spectral place & self ID
        act_self = 1.0 / (1.0 + np.exp(-np.clip(k * (snr_key - tau), -40.0, 40.0)))
        act_low_feat = 1.0 / (1.0 + np.exp(-np.clip(0.85 * k * (snr_key - (tau - 5.0)), -40.0, 40.0)))
        act_place = 1.0 / (1.0 + np.exp(-np.clip(k * (snr_key - (tau - 1.0)), -40.0, 40.0)))
        act_unk_decay = 1.0 / (1.0 + np.exp(np.clip(0.95 * k * (snr_key - (tau - 2.5)), -40.0, 40.0)))

        w_voice, w_manner, w_place, w_sib = self.feat_weights
        unk_low, unk_high = self.unk_params

        logits = np.full(N_RESPONSES, -2.5, dtype=float)
        for j, rc in enumerate(DISPLAY_CONSONANTS):
            if rc == "???":
                logits[j] = unk_high + (unk_low - unk_high) * act_unk_decay
            elif j == stim_idx:
                logits[j] = -2.4 + (self.max_self_logit[stim_idx] - (-2.4)) * act_self
            else:
                rf = PHONETIC_FEATURES[rc]
                sim = -2.5
                if sf["voicing"] == rf["voicing"]:
                    sim += w_voice * act_low_feat
                if sf["manner"] == rf["manner"]:
                    sim += w_manner * act_low_feat
                if sf["place"] == rf["place"]:
                    sim += w_place * act_place
                if sf["sibilant"] == 1 and rf["sibilant"] == 1:
                    sim += w_sib * act_low_feat
                logits[j] = sim

        logits = logits + self.asymmetry_matrix[stim_idx] + self.token_logit_offsets[token_idx, stim_idx]
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        self._cache[cache_key] = probs
        return probs

    def matrix_at_snrs(self, snr_per_consonant: np.ndarray | list[float]) -> np.ndarray:
        """Return (10, 11) confusion matrix when each consonant i is tested at snr_per_consonant[i]."""
        out = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
        for i in range(N_STIMULI):
            out[i] = self.response_probs(i, float(snr_per_consonant[i]), token_idx=None)
        return out

    def pooled_matrix_for_design(self, snr_grid_10xK: np.ndarray) -> np.ndarray:
        """
        Return the exact expected (10, 11) confusion matrix pooled evenly across
        a stimulus set where consonant i is presented at snr_grid_10xK[i, 0..K-1].
        """
        grid = np.asarray(snr_grid_10xK, dtype=float)
        out = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
        for i in range(N_STIMULI):
            row_acc = np.zeros(N_RESPONSES, dtype=float)
            for snr_val in grid[i]:
                row_acc += self.response_probs(i, float(snr_val), token_idx=None)
            out[i] = row_acc / grid.shape[1]
        return normalize_rows(out)

    def true_srt50_db(self, stim_idx: int) -> float:
        """
        Compute the listener's true 50%-correct SNR threshold (dB) for consonant
        ``stim_idx`` over [-24, +36] dB (or NaN if asymptotic plateau < 50%,
        though clamped to +36 dB for finite error reporting).
        """
        snrs = np.linspace(-24.0, 36.0, 241)
        pcs = np.array([self.response_probs(stim_idx, float(s))[stim_idx] for s in snrs])
        if pcs[-1] < 0.50:
            return 36.0
        if pcs[0] >= 0.50:
            return -24.0
        idx = int(np.searchsorted(pcs, 0.50))
        s0, s1 = snrs[idx - 1], snrs[idx]
        p0, p1 = pcs[idx - 1], pcs[idx]
        frac = (0.50 - p0) / max(p1 - p0, 1e-12)
        return float(s0 + frac * (s1 - s0))

    def respond(self, stim_idx: int, snr_db: float, token_idx: int) -> int:
        probs = self.response_probs(stim_idx, snr_db, token_idx=token_idx)
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        u = float(self.rng.random())
        return min(int(np.searchsorted(cdf, u, side="right")), N_RESPONSES - 1)


def _build_multisnr_listener(
    listener_id: str,
    spec: MultiSNRArchetypeSpec,
    rng: np.random.Generator,
) -> MultiSNRVirtualListener:
    """Instantiate an individual MultiSNRVirtualListener from an archetype spec."""
    # Baseline NH 50% transition SNR sits ~5.5 dB below qVCV's reference SNRs
    # so that NH listeners achieve ~85-88% at Delta_SNR = 0 dB
    nh_tau_offset = {
        "B": -6.0, "D": -4.8, "G": -4.2, "K": -5.2, "N": -7.0,
        "S": -5.5, "SH": -6.0, "T": -5.3, "V": -5.5, "Z": -4.8,
    }

    listener_lf_shift = spec.snr_shift_lf_db + rng.normal(0.0, 1.3)
    listener_hf_shift = spec.snr_shift_hf_db + rng.normal(0.0, 1.6)

    tau_50 = np.zeros(N_STIMULI, dtype=float)
    slope_db = np.zeros(N_STIMULI, dtype=float)
    max_self = np.zeros(N_STIMULI, dtype=float)

    for i, sc in enumerate(STIMULI_CONSONANTS):
        is_hf = bool(PHONETIC_FEATURES[sc]["hf_cue"])
        ref_snr = CONSONANT_SNR_DB[sc]
        shift = listener_hf_shift if is_hf else listener_lf_shift
        tau_50[i] = ref_snr + nh_tau_offset[sc] + shift + rng.normal(0.0, 0.8)
        slope_db[i] = np.clip(spec.slope_per_db * rng.uniform(0.85, 1.18), 0.16, 0.65)
        base_ceil = spec.max_self_hf if is_hf else spec.max_self_lf
        max_self[i] = base_ceil + rng.normal(0.0, 0.25)

    feat_weights = np.maximum(
        0.0,
        np.array([spec.w_voice, spec.w_manner, spec.w_place, spec.w_sibilant]) + rng.normal(0.0, 0.10, size=4),
    )
    unk_params = np.array([
        spec.unk_low_snr + rng.normal(0.0, 0.20),
        spec.unk_high_snr + rng.normal(0.0, 0.20),
    ])

    asym = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
    # Empirical VCV confusion asymmetries (Phatak & Allen 2007)
    base_asym = {
        ("G", "D"): 0.65, ("D", "G"): 0.30,
        ("V", "B"): 0.70, ("B", "V"): 0.35,
        ("T", "K"): 0.55, ("K", "T"): 0.40,
        ("Z", "S"): 0.35, ("Z", "D"): 0.40,
        ("S", "SH"): 0.40, ("SH", "S"): 0.30,
    }
    for (sc, rc), val in base_asym.items():
        asym[STIM_INDEX[sc], RESP_INDEX[rc]] += val
    for (sc, rc), val in spec.asymmetry.items():
        asym[STIM_INDEX[sc], RESP_INDEX[rc]] += val
    # Add individual idiosyncratic off-diagonal perturbations
    asym[:, :N_STIMULI] += rng.normal(0.0, 0.18, size=(N_STIMULI, N_STIMULI))
    np.fill_diagonal(asym[:, :N_STIMULI], 0.0)

    # Mild talker-token (_1_ vs _2_) acoustic differences (zero-sum so mean is preserved)
    tok_delta = rng.normal(0.0, 0.14, size=(N_STIMULI, N_RESPONSES))
    token_offsets = np.stack([tok_delta, -tok_delta], axis=0)

    return MultiSNRVirtualListener(
        listener_id=listener_id,
        archetype=spec.archetype,
        description=spec.description,
        tau_50_db=tau_50,
        slope_db=slope_db,
        max_self_logit=max_self,
        feat_weights=feat_weights,
        unk_params=unk_params,
        asymmetry_matrix=asym,
        token_logit_offsets=token_offsets,
    )


def generate_multisnr_cohort(seed: int = 20261115, per_archetype: int = 4) -> list[MultiSNRVirtualListener]:
    """Generate 24 virtual listeners (4 per clinical archetype) with full SNR psychometric functions."""
    rng = np.random.default_rng(seed)

    specs: list[tuple[str, MultiSNRArchetypeSpec]] = [
        (
            "NH",
            MultiSNRArchetypeSpec(
                archetype="normal_hearing",
                description="Normal hearing: low SRTs, steep slopes, ~97% quiet ceiling",
                snr_shift_lf_db=0.0, snr_shift_hf_db=0.0,
                slope_per_db=0.38,
                max_self_lf=4.2, max_self_hf=4.1,
                w_voice=1.05, w_manner=1.15, w_place=0.65, w_sibilant=0.85,
                unk_low_snr=0.45, unk_high_snr=-3.6,
                asymmetry={},
            ),
        ),
        (
            "HF_Mild",
            MultiSNRArchetypeSpec(
                archetype="sloping_hf_mild",
                description="Mild-moderate sloping HF loss: +4.5 dB HF SRT shift, mild HF place confusions",
                snr_shift_lf_db=1.2, snr_shift_hf_db=4.5,
                slope_per_db=0.31,
                max_self_lf=3.8, max_self_hf=2.7,
                w_voice=1.05, w_manner=1.05, w_place=0.30, w_sibilant=0.95,
                unk_low_snr=0.65, unk_high_snr=-2.4,
                asymmetry={("S", "T"): 0.35, ("Z", "V"): 0.45},
            ),
        ),
        (
            "HF_Steep",
            MultiSNRArchetypeSpec(
                archetype="sloping_hf_steep",
                description="Steep ski-slope HF loss: near-normal LF, +9 dB HF shift & HF ceiling plateau",
                snr_shift_lf_db=2.0, snr_shift_hf_db=9.0,
                slope_per_db=0.26,
                max_self_lf=3.4, max_self_hf=1.55,
                w_voice=1.00, w_manner=0.85, w_place=0.05, w_sibilant=0.75,
                unk_low_snr=0.90, unk_high_snr=-1.3,
                asymmetry={("S", "SH"): 0.45, ("T", "K"): 0.50, ("Z", "V"): 0.70, ("G", "B"): 0.55},
            ),
        ),
        (
            "Broad_Place",
            MultiSNRArchetypeSpec(
                archetype="spectral_smearing",
                description="Broad spectral smearing: +5 dB shift, strong voicing/manner, persistent place ceiling",
                snr_shift_lf_db=4.5, snr_shift_hf_db=5.5,
                slope_per_db=0.25,
                max_self_lf=2.2, max_self_hf=1.9,
                w_voice=1.20, w_manner=1.30, w_place=0.0, w_sibilant=0.80,
                unk_low_snr=0.55, unk_high_snr=-2.2,
                asymmetry={("B", "D"): 0.55, ("D", "B"): 0.45, ("S", "SH"): 0.65, ("SH", "S"): 0.60},
            ),
        ),
        (
            "Voicing_Deficit",
            MultiSNRArchetypeSpec(
                archetype="temporal_voicing",
                description="Temporal / voicing deficit: +3.5 dB shift, heavy cognate voicing confusions (D/T, G/K, Z/S)",
                snr_shift_lf_db=3.5, snr_shift_hf_db=3.5,
                slope_per_db=0.29,
                max_self_lf=2.6, max_self_hf=2.5,
                w_voice=0.05, w_manner=1.00, w_place=0.85, w_sibilant=0.85,
                unk_low_snr=0.55, unk_high_snr=-2.3,
                asymmetry={
                    ("D", "T"): 1.15, ("T", "D"): 0.95,
                    ("G", "K"): 1.10, ("K", "G"): 0.95,
                    ("Z", "S"): 1.15, ("S", "Z"): 0.85,
                    ("D", "N"): 0.65,
                },
            ),
        ),
        (
            "Elevated_SRT",
            MultiSNRArchetypeSpec(
                archetype="poor_sin_elevated_srt",
                description="Broad speech-in-noise loss: +7.5 dB uniform SNR shift, good quiet recovery at high SNR",
                snr_shift_lf_db=7.5, snr_shift_hf_db=8.0,
                slope_per_db=0.30,
                max_self_lf=3.5, max_self_hf=3.1,
                w_voice=0.90, w_manner=0.95, w_place=0.45, w_sibilant=0.75,
                unk_low_snr=1.05, unk_high_snr=-2.5,
                asymmetry={},
            ),
        ),
    ]

    cohort: list[MultiSNRVirtualListener] = []
    for prefix, spec in specs:
        for k in range(1, per_archetype + 1):
            cohort.append(_build_multisnr_listener(f"{prefix}_{k:02d}", spec, rng))
    return cohort


def save_multisnr_cohort_json(listeners: list[MultiSNRVirtualListener], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "listener_id": l.listener_id,
            "archetype": l.archetype,
            "description": l.description,
            "tau_50_db": np.round(l.tau_50_db, 4).tolist(),
            "slope_db": np.round(l.slope_db, 4).tolist(),
            "max_self_logit": np.round(l.max_self_logit, 4).tolist(),
            "feat_weights": np.round(l.feat_weights, 4).tolist(),
            "unk_params": np.round(l.unk_params, 4).tolist(),
            "asymmetry_matrix": np.round(l.asymmetry_matrix, 4).tolist(),
            "token_logit_offsets": np.round(l.token_logit_offsets, 4).tolist(),
        }
        for l in listeners
    ]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# --------------------------------------------------------------------------
# Human-Calibrated Multi-SNR Virtual Listener (98-Listener Dataset)
# --------------------------------------------------------------------------

class HumanCalibratedMultiSNRListener:
    """
    Continuous SNR-dependent virtual listener calibrated directly to one of the
    98 empirical human participants in ``qtest_dataset``:
      - Exact match to the participant's calibrated confusion matrix at each
        consonant's reference SNR (B:18, V:18, N:15, D:4, G:3, K/S/SH/T/Z:0 dB).
      - Plomp (1978) / SII power-sum internal + external noise masking for fixed
        60 dB SPL speech presentation, calibrated against all 98 participants'
        60-dB-SNR Practice (quiet) vs. Ref-SNR Test scores.
      - Empirical off-diagonal phonetic confusion trajectory with low-SNR
        relaxation toward marginal guessing and empirical _1_ vs _2_ token offsets.
    """

    def __init__(
        self,
        listener_id: str,
        archetype: str,
        is_nh: bool,
        pta_124_db: float,
        rhl_db: float,
        description: str,
        ref_matrix: np.ndarray,                 # (10, 11) calibrated P(R | S, Ref_SNR)
        slope_per_db: float,
        snr_quiet_db: np.ndarray,               # (10,) quiet equivalent SNR ceiling at 60 dB SPL
        ceil_prob: np.ndarray,                  # (10,) high-SNR asymptote per consonant
        eta0_logit: np.ndarray,                 # (10,) anchor logit at Delta_SNR_eff = 0
        error_profile: np.ndarray,              # (10, 11) empirical off-diagonal profile
        token_diag_logit_halfdiff: np.ndarray,  # (10,) _1_ vs _2_ difficulty offset
        marginal_low_snr_profile: np.ndarray,   # (11,) deep-noise response prior
        session_logit_sd: float = 0.33,
        session_logit_shift: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ):
        self.listener_id = listener_id
        self.archetype = archetype
        self.is_nh = bool(is_nh)
        self.pta_124_db = float(pta_124_db)
        self.rhl_db = float(rhl_db)
        self.description = description
        self.ref_matrix = normalize_rows(np.asarray(ref_matrix, dtype=float))
        self.slope_per_db = float(slope_per_db)
        self.snr_quiet_db = np.asarray(snr_quiet_db, dtype=float)
        self.ceil_prob = np.asarray(ceil_prob, dtype=float)
        self.eta0_logit = np.asarray(eta0_logit, dtype=float)
        self.error_profile = np.asarray(error_profile, dtype=float)
        self.token_diag_logit_halfdiff = np.asarray(token_diag_logit_halfdiff, dtype=float)
        self.marginal_low_snr_profile = np.asarray(marginal_low_snr_profile, dtype=float)
        self.session_logit_sd = float(session_logit_sd)
        self.session_logit_shift = float(session_logit_shift)
        self.rng = rng if rng is not None else np.random.default_rng()
        self._ref_snrs = np.array([CONSONANT_SNR_DB[c] for c in STIMULI_CONSONANTS], dtype=float)
        self._seff_ref = self._calc_seff_vec(self._ref_snrs, self.snr_quiet_db)
        self._cache: dict[tuple[int, Optional[int], float], np.ndarray] = {}

        self._ref_offdiag = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
        for ci in range(N_STIMULI):
            off = self.ref_matrix[ci].copy()
            off[ci] = 0.0
            s = float(off.sum())
            if s > 1e-12:
                self._ref_offdiag[ci] = off / s
            else:
                self._ref_offdiag[ci] = self.error_profile[ci]

    @staticmethod
    def _calc_seff_vec(snr_db: np.ndarray | float, snr_quiet_db: np.ndarray | float) -> np.ndarray:
        ln10_10 = np.log(10.0) / 10.0
        a = -np.asarray(snr_db, dtype=float) * ln10_10
        b = -np.asarray(snr_quiet_db, dtype=float) * ln10_10
        m = np.maximum(a, b)
        return -10.0 * (m + np.log(np.exp(a - m) + np.exp(b - m))) / np.log(10.0)

    def clone_with_rng(
        self,
        rng: np.random.Generator,
        draw_session_shift: bool = False,
    ) -> "HumanCalibratedMultiSNRListener":
        sess_z = float(rng.normal(0.0, self.session_logit_sd)) if draw_session_shift else 0.0
        return HumanCalibratedMultiSNRListener(
            listener_id=self.listener_id,
            archetype=self.archetype,
            is_nh=self.is_nh,
            pta_124_db=self.pta_124_db,
            rhl_db=self.rhl_db,
            description=self.description,
            ref_matrix=self.ref_matrix,
            slope_per_db=self.slope_per_db,
            snr_quiet_db=self.snr_quiet_db,
            ceil_prob=self.ceil_prob,
            eta0_logit=self.eta0_logit,
            error_profile=self.error_profile,
            token_diag_logit_halfdiff=self.token_diag_logit_halfdiff,
            marginal_low_snr_profile=self.marginal_low_snr_profile,
            session_logit_sd=self.session_logit_sd,
            session_logit_shift=sess_z,
            rng=rng,
        )

    def response_probs(
        self,
        stim_idx: int,
        snr_db: float,
        token_idx: Optional[int] = None,
    ) -> np.ndarray:
        snr_key = round(float(snr_db), 3)
        cache_key = (stim_idx, token_idx, snr_key)
        if cache_key in self._cache:
            return self._cache[cache_key]

        if token_idx is None:
            p0 = self.response_probs(stim_idx, snr_key, token_idx=0)
            p1 = self.response_probs(stim_idx, snr_key, token_idx=1)
            out = 0.5 * (p0 + p1)
            self._cache[cache_key] = out
            return out

        gamma = 1.0 / float(N_RESPONSES)
        seff = float(self._calc_seff_vec(snr_key, float(self.snr_quiet_db[stim_idx])))
        d_seff = seff - float(self._seff_ref[stim_idx])
        dsnr_nom = snr_key - float(self._ref_snrs[stim_idx])

        tok_sign = 1.0 if token_idx == 0 else -1.0
        tok_shift = tok_sign * float(self.token_diag_logit_halfdiff[stim_idx])

        eta = float(self.eta0_logit[stim_idx]) + self.slope_per_db * d_seff + tok_shift + self.session_logit_shift
        u = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
        c_max = float(self.ceil_prob[stim_idx])
        p_corr = float(np.clip(gamma + (c_max - gamma) * u, 1e-6, 1.0 - 1e-6))

        w_away = 1.0 - float(np.exp(-0.25 * (d_seff * d_seff)))
        phon_err = (1.0 - w_away) * self._ref_offdiag[stim_idx] + w_away * self.error_profile[stim_idx]

        w_noise = 1.0 / (1.0 + float(np.exp(0.45 * (dsnr_nom + 10.0))))
        guess_off = self.marginal_low_snr_profile.copy()
        guess_off[stim_idx] = 0.0
        guess_off /= max(float(guess_off.sum()), 1e-12)

        off_dist = (1.0 - w_noise) * phon_err + w_noise * guess_off
        off_dist[stim_idx] = 0.0
        off_dist /= max(float(off_dist.sum()), 1e-12)

        probs = (1.0 - p_corr) * off_dist
        probs[stim_idx] = p_corr
        probs /= float(probs.sum())
        self._cache[cache_key] = probs
        return probs

    def matrix_at_snrs(self, snr_per_consonant: np.ndarray | list[float]) -> np.ndarray:
        out = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
        for i in range(N_STIMULI):
            out[i] = self.response_probs(i, float(snr_per_consonant[i]), token_idx=None)
        return out

    def pooled_matrix_for_design(self, snr_grid_10xK: np.ndarray) -> np.ndarray:
        grid = np.asarray(snr_grid_10xK, dtype=float)
        out = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
        for i in range(N_STIMULI):
            row_acc = np.zeros(N_RESPONSES, dtype=float)
            for snr_val in grid[i]:
                row_acc += self.response_probs(i, float(snr_val), token_idx=None)
            out[i] = row_acc / grid.shape[1]
        return normalize_rows(out)

    def true_srt50_db(self, stim_idx: int) -> float:
        snrs = np.linspace(-24.0, 36.0, 241)
        pcs = np.array([self.response_probs(stim_idx, float(s))[stim_idx] for s in snrs])
        if pcs[-1] < 0.50:
            return 36.0
        if pcs[0] >= 0.50:
            return -24.0
        idx = int(np.searchsorted(pcs, 0.50))
        s0, s1 = snrs[idx - 1], snrs[idx]
        p0, p1 = pcs[idx - 1], pcs[idx]
        frac = (0.50 - p0) / max(p1 - p0, 1e-12)
        return float(s0 + frac * (s1 - s0))

    def respond(self, stim_idx: int, snr_db: float, token_idx: int) -> int:
        probs = self.response_probs(stim_idx, snr_db, token_idx=token_idx)
        cdf = np.cumsum(probs)
        cdf[-1] = 1.0
        u = float(self.rng.random())
        return min(int(np.searchsorted(cdf, u, side="right")), N_RESPONSES - 1)


def load_human_multisnr_cohort(
    json_path: Optional[Path] = None,
) -> list[HumanCalibratedMultiSNRListener]:
    """Load all 98 human-calibrated multi-SNR virtual listeners from human_listeners_98.json."""
    if json_path is None:
        json_path = Path(__file__).resolve().parent / "human_data" / "human_listeners_98.json"
    with Path(json_path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    out: list[HumanCalibratedMultiSNRListener] = []
    for rec in data["listeners"]:
        mp = rec["multisnr_params"]
        out.append(
            HumanCalibratedMultiSNRListener(
                listener_id=rec["listener_id"],
                archetype=rec["archetype"],
                is_nh=bool(rec["is_nh"]),
                pta_124_db=float(rec["pta_124_db"]),
                rhl_db=float(rec["rhl_db"]),
                description=rec["description"],
                ref_matrix=np.asarray(rec["confusion_matrix"], dtype=float),
                slope_per_db=float(mp["slope_per_db"]),
                snr_quiet_db=np.asarray(mp["snr_quiet_db"], dtype=float),
                ceil_prob=np.asarray(mp["ceil_prob"], dtype=float),
                eta0_logit=np.asarray(mp["eta0_logit"], dtype=float),
                error_profile=np.asarray(mp["error_profile"], dtype=float),
                token_diag_logit_halfdiff=np.asarray(mp["token_diag_logit_halfdiff"], dtype=float),
                marginal_low_snr_profile=np.asarray(mp["marginal_low_snr_profile"], dtype=float),
                session_logit_sd=float(mp.get("session_logit_sd", 0.33)),
            )
        )
    return out
