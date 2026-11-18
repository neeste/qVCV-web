"""
qvcv_multisnr_designs.py

Multi-SNR stimulus-set designs, trial-selection strategies (random, stratified
Latin-square block, adaptive consonant, and 2D Bayesian Consonant x SNR
tracking), and SNR-psychometric surface estimators for qVCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .qvcv_multisnr_simulator import (
        EVAL_REL_SNRS_DB,
        MultiSNRVirtualListener,
    )
    from .qvcv_simulator import (
        CONSONANT_SNR_DB,
        DISPLAY_CONSONANTS,
        N_RESPONSES,
        N_STIMULI,
        PHONETIC_FEATURES,
        STIMULI_CONSONANTS,
        normalize_rows,
    )
    from .qvcv_trackers import build_weak_phonetic_prior
except ImportError:
    from qvcv_multisnr_simulator import (
        EVAL_REL_SNRS_DB,
        MultiSNRVirtualListener,
    )
    from qvcv_simulator import (
        CONSONANT_SNR_DB,
        DISPLAY_CONSONANTS,
        N_RESPONSES,
        N_STIMULI,
        PHONETIC_FEATURES,
        STIMULI_CONSONANTS,
        normalize_rows,
    )
    from qvcv_trackers import build_weak_phonetic_prior


# --------------------------------------------------------------------------
# 1. Stimulus-Set Designs
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class StimulusSetDesign:
    name: str
    label: str
    snr_grid: np.ndarray  # shape (10 consonants, K snr_slots) in dB

    @property
    def n_slots(self) -> int:
        return int(self.snr_grid.shape[1])


def build_stimulus_designs() -> dict[str, StimulusSetDesign]:
    """Build candidate single-SNR and multi-SNR stimulus sets for qVCV."""
    ref_vec = np.array([CONSONANT_SNR_DB[c] for c in STIMULI_CONSONANTS], dtype=float)[:, None]

    designs = [
        StimulusSetDesign(
            name="single_ref",
            label="Single SNR (5x identical @ Ref SNR)",
            snr_grid=ref_vec + np.array([[0.0, 0.0, 0.0, 0.0, 0.0]]),
        ),
        StimulusSetDesign(
            name="rel_pm3",
            label="Relative 5-SNR Narrow (Ref -3..+3 dB, 1.5 dB step)",
            snr_grid=ref_vec + np.array([[-3.0, -1.5, 0.0, 1.5, 3.0]]),
        ),
        StimulusSetDesign(
            name="rel_pm6",
            label="Relative Symmetric 5-SNR (Ref -6..+6 dB, 3 dB step)",
            snr_grid=ref_vec + np.array([[-6.0, -3.0, 0.0, 3.0, 6.0]]),
        ),
        StimulusSetDesign(
            name="rel_up_m3_p9",
            label="Relative Clinical 5-SNR (Ref -3, 0, +3, +6, +9 dB)",
            snr_grid=ref_vec + np.array([[-3.0, 0.0, 3.0, 6.0, 9.0]]),
        ),
        StimulusSetDesign(
            name="rel_up_m4_p12",
            label="Relative Wide Clinical 5-SNR (Ref -4, 0, +4, +8, +12 dB)",
            snr_grid=ref_vec + np.array([[-4.0, 0.0, 4.0, 8.0, 12.0]]),
        ),
        StimulusSetDesign(
            name="abs_shared",
            label="Shared Absolute 5-SNR (-6, 0, +6, +12, +18 dB)",
            snr_grid=np.repeat(np.array([[-6.0, 0.0, 6.0, 12.0, 18.0]]), N_STIMULI, axis=0),
        ),
        StimulusSetDesign(
            name="rel_dense9",
            label="Dense Relative 9-SNR Bank (Ref -12..+12 dB, 3 dB step)",
            snr_grid=ref_vec + np.array([[-12.0, -9.0, -6.0, -3.0, 0.0, 3.0, 6.0, 9.0, 12.0]]),
        ),
    ]
    return {d.name: d for d in designs}


# --------------------------------------------------------------------------
# 2. Concave SNR-Psychometric Multinomial Choice Fitter (18 parameters)
# --------------------------------------------------------------------------
# Parameter vector theta (18,):
#   0..9  : per-consonant correct-ID intercept alpha_i (at Delta_SNR = 0 dB)
#   10    : LF correct-ID SNR slope beta_lf (per dB)
#   11    : HF correct-ID SNR slope beta_hf (per dB)
#   12    : shared voicing off-diagonal weight
#   13    : shared manner off-diagonal weight
#   14    : shared place off-diagonal weight
#   15    : shared sibilant off-diagonal weight
#   16    : '???' catch-all intercept (at Delta_SNR = 0 dB)
#   17    : '???' catch-all SNR slope (negative = decays as SNR rises)

REF_SNR_ARRAY: np.ndarray = np.array([CONSONANT_SNR_DB[c] for c in STIMULI_CONSONANTS], dtype=float)


def _precompute_static_basis() -> np.ndarray:
    """Static (10, 11, 18) basis for terms that do not depend on SNR (indices 0..9, 12..16)."""
    b = np.zeros((N_STIMULI, N_RESPONSES, 18), dtype=float)
    for i, sc in enumerate(STIMULI_CONSONANTS):
        sf = PHONETIC_FEATURES[sc]
        for j, rc in enumerate(DISPLAY_CONSONANTS):
            if rc == "???":
                b[i, j, 16] = 1.0
            elif j == i:
                b[i, j, i] = 1.0
            else:
                rf = PHONETIC_FEATURES[rc]
                if sf["voicing"] == rf["voicing"]:
                    b[i, j, 12] = 1.0
                if sf["manner"] == rf["manner"]:
                    b[i, j, 13] = 1.0
                if sf["place"] == rf["place"]:
                    b[i, j, 14] = 1.0
                if sf["sibilant"] == 1 and rf["sibilant"] == 1:
                    b[i, j, 15] = 1.0
    return b


STATIC_BASIS_18: np.ndarray = _precompute_static_basis()
IS_HF_VEC: np.ndarray = np.array([float(PHONETIC_FEATURES[c]["hf_cue"]) for c in STIMULI_CONSONANTS])

THETA18_PRIOR_MEAN: np.ndarray = np.array(
    [2.0] * 10 + [0.15, 0.15, 0.70, 0.80, 0.40, 0.60, -0.6, -0.10],
    dtype=float,
)
THETA18_PRIOR_PREC: np.ndarray = np.diag(
    [1.0 / (2.5 ** 2)] * 10 + [1.0 / (0.80 ** 2), 1.0 / (0.80 ** 2)] + [1.0 / (1.8 ** 2)] * 6
)


_SRT_SEARCH_GRID: np.ndarray = np.linspace(-24.0, 36.0, 121)


@dataclass
class FittedPsychometricModel:
    theta: np.ndarray  # (18,)

    def predict_row(self, stim_idx: int, snr_db: float) -> np.ndarray:
        dsnr = float(snr_db) - REF_SNR_ARRAY[stim_idx]
        x = STATIC_BASIS_18[stim_idx].copy()  # (11, 18)
        if IS_HF_VEC[stim_idx] > 0.5:
            x[stim_idx, 11] = dsnr
        else:
            x[stim_idx, 10] = dsnr
        x[N_RESPONSES - 1, 17] = dsnr

        logits = x @ self.theta
        logits -= np.max(logits)
        p = np.exp(logits)
        return p / np.sum(p)

    def predict_grid(self, snr_grid_10xM: np.ndarray) -> np.ndarray:
        """Vectorized prediction returning shape (10, M, 11) for snr_grid_10xM (10, M)."""
        m_pts = snr_grid_10xM.shape[1]
        dsnr = snr_grid_10xM - REF_SNR_ARRAY[:, None]  # (10, M)
        logits_base = np.einsum("ijp,p->ij", STATIC_BASIS_18, self.theta)  # (10, 11)
        logits = np.repeat(logits_base[:, None, :], m_pts, axis=1)         # (10, M, 11)
        for i in range(N_STIMULI):
            slope_val = self.theta[11] if IS_HF_VEC[i] > 0.5 else self.theta[10]
            logits[i, :, i] += slope_val * dsnr[i]
            logits[i, :, N_RESPONSES - 1] += self.theta[17] * dsnr[i]
        logits -= np.max(logits, axis=2, keepdims=True)
        exp_u = np.exp(logits)
        return exp_u / np.sum(exp_u, axis=2, keepdims=True)

    def predict_all_srt50_db(self) -> np.ndarray:
        """Vectorized SRT50 (dB) across all 10 consonants, shape (10,)."""
        grid = np.repeat(_SRT_SEARCH_GRID[None, :], N_STIMULI, axis=0)  # (10, 121)
        probs = self.predict_grid(grid)                                 # (10, 121, 11)
        out = np.zeros(N_STIMULI, dtype=float)
        for i in range(N_STIMULI):
            pcs = probs[i, :, i]
            if pcs[-1] < 0.50:
                out[i] = 36.0
            elif pcs[0] >= 0.50:
                out[i] = -24.0
            else:
                idx = int(np.searchsorted(pcs, 0.50))
                s0, s1 = _SRT_SEARCH_GRID[idx - 1], _SRT_SEARCH_GRID[idx]
                p0, p1 = pcs[idx - 1], pcs[idx]
                out[i] = float(s0 + (0.50 - p0) / max(p1 - p0, 1e-12) * (s1 - s0))
        return out

    def predict_srt50_db(self, stim_idx: int) -> float:
        return float(self.predict_all_srt50_db()[stim_idx])


def fit_multisnr_psychometric_model(
    cell_counts: np.ndarray,  # (10, K, 11) counts at snr_grid (10, K)
    snr_grid: np.ndarray,     # (10, K)
    theta_init: Optional[np.ndarray] = None,
    max_iter: int = 8,
) -> FittedPsychometricModel:
    """
    Fit the strictly concave 18-parameter SNR-dependent multinomial choice model
    via vectorized Newton-Raphson across all (consonant, snr_slot) cells.
    """
    n_slots = snr_grid.shape[1]
    # Build (10, K, 11, 18) design tensor
    x = np.repeat(STATIC_BASIS_18[:, None, :, :], n_slots, axis=1)
    dsnr = snr_grid - REF_SNR_ARRAY[:, None]  # (10, K)
    for i in range(N_STIMULI):
        slope_col = 11 if IS_HF_VEC[i] > 0.5 else 10
        x[i, :, i, slope_col] = dsnr[i, :]
        x[i, :, N_RESPONSES - 1, 17] = dsnr[i, :]

    # Flatten active (i, k) cells where trials > 0 for ultra-fast vectorized updates
    flat_counts = cell_counts.reshape(N_STIMULI * n_slots, N_RESPONSES)
    flat_totals = flat_counts.sum(axis=1)
    active = flat_totals > 0
    if not np.any(active):
        return FittedPsychometricModel(theta=THETA18_PRIOR_MEAN.copy())

    c_act = flat_counts[active]         # (M, 11)
    n_act = flat_totals[active]         # (M,)
    x_act = x.reshape(N_STIMULI * n_slots, N_RESPONSES, 18)[active]  # (M, 11, 18)

    theta = THETA18_PRIOR_MEAN.copy() if theta_init is None else theta_init.copy()

    def _eval(th: np.ndarray) -> tuple[np.ndarray, float]:
        logits = np.einsum("mjp,p->mj", x_act, th)
        logits -= logits.max(axis=1, keepdims=True)
        exp_u = np.exp(logits)
        mu = exp_u / exp_u.sum(axis=1, keepdims=True)
        diff = th - THETA18_PRIOR_MEAN
        obj = float(
            np.sum(c_act * np.log(np.maximum(mu, 1e-15)))
            - 0.5 * float(diff @ THETA18_PRIOR_PREC @ diff)
        )
        return mu, obj

    mu, obj = _eval(theta)
    for _ in range(max_iter):
        resid = c_act - n_act[:, None] * mu
        grad = np.einsum("mj,mjp->p", resid, x_act) - (THETA18_PRIOR_PREC @ (theta - THETA18_PRIOR_MEAN))
        if float(np.max(np.abs(grad))) < 1e-6:
            break
        x_bar = np.einsum("mj,mjp->mp", mu, x_act)
        x_cent = x_act - x_bar[:, None, :]
        weights = n_act[:, None] * mu
        neg_hess = np.einsum("mj,mjp,mjq->pq", weights, x_cent, x_cent) + THETA18_PRIOR_PREC
        step = np.linalg.solve(neg_hess, grad)

        alpha = 1.0
        while alpha >= 1e-3:
            cand = theta + alpha * step
            cand_mu, cand_obj = _eval(cand)
            if cand_obj >= obj - 1e-10:
                theta, mu, obj = cand, cand_mu, cand_obj
                break
            alpha *= 0.5
        else:
            break

    return FittedPsychometricModel(theta=theta)


# --------------------------------------------------------------------------
# 3. Multi-SNR Session Tracker & Simulator
# --------------------------------------------------------------------------

MULTISNR_MODES = ("block", "random", "adaptive_consonant", "adaptive_2d_snr")


@dataclass
class MultiSNRSessionOutput:
    cell_counts: np.ndarray                 # (10, K, 11) raw counts per (consonant, snr_slot, response)
    pooled_empirical: np.ndarray            # (10, 11) raw pooled confusion matrix
    pooled_dirichlet: np.ndarray            # (10, 11) weakly smoothed pooled confusion matrix
    psychometric_fit: FittedPsychometricModel


class MultiSNRTracker:
    """
    Runs a qVCV session on any StimulusSetDesign under one of 4 selection strategies:
      - block              : Stratified Latin-square block schedule (no repeats at N=100 for K=5)
      - random             : Uniform random selection with replacement
      - adaptive_consonant : Adapts target consonant by Dirichlet confusion variance; cycles SNRs evenly
      - adaptive_2d_snr    : 2D Bayesian tracker over (consonant, snr_slot) targeting each
                             consonant's 50%-correct / maximum-entropy transition SNR
    """

    def __init__(
        self,
        design: StimulusSetDesign,
        mode: str = "block",
        prior_conc: float = 1.5,
    ):
        if mode not in MULTISNR_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected {MULTISNR_MODES}")
        self.design = design
        self.mode = mode
        self._weak_prior = build_weak_phonetic_prior(concentration=prior_conc)

        # Prebuild discrete threshold grid per consonant for fast 2D Bayesian SNR tracking
        self._n_grid = 25
        self._tau_grids = np.zeros((N_STIMULI, self._n_grid), dtype=float)
        self._psi_lookup = np.zeros((N_STIMULI, design.n_slots, self._n_grid), dtype=float)
        for i in range(N_STIMULI):
            lo = float(np.min(design.snr_grid[i])) - 8.0
            hi = float(np.max(design.snr_grid[i])) + 8.0
            tgrid = np.linspace(lo, hi, self._n_grid)
            self._tau_grids[i] = tgrid
            for s_idx in range(design.n_slots):
                snr_val = design.snr_grid[i, s_idx]
                # Logistic correctness model from guess floor ~0.09 to ~0.92
                self._psi_lookup[i, s_idx] = 0.09 + 0.83 / (1.0 + np.exp(-0.32 * (snr_val - tgrid)))

    def _build_stratified_schedule(
        self, rng: np.random.Generator, n_trials: int
    ) -> list[tuple[int, int, int]]:
        """
        Build a balanced stratified block schedule of (stim_idx, token_idx, snr_slot).
        For K=5 and n_trials=100, visits every single one of the 100 (consonant, token, snr_slot)
        stimuli literally once, while ensuring every 20-trial block samples all consonants,
        both tokens, and all SNR slots evenly without consecutive consonant repeats.
        """
        k_slots = self.design.n_slots
        # Assign each (consonant, token) its own Latin shift across blocks
        perm_c = rng.permutation(N_STIMULI)
        slot_perm = rng.permutation(k_slots)

        schedule: list[tuple[int, int, int]] = []
        n_blocks = (n_trials + 19) // 20
        for b in range(n_blocks):
            block_items: list[tuple[int, int, int]] = []
            for rank, c_idx in enumerate(perm_c):
                for tok in (0, 1):
                    slot = int(slot_perm[(rank + 2 * b + tok) % k_slots])
                    block_items.append((int(c_idx), tok, slot))
            rng.shuffle(block_items)
            # Avoid boundary consonant repeat
            if schedule and len(block_items) > 1 and block_items[0][0] == schedule[-1][0]:
                swap_k = next((m for m in range(1, len(block_items)) if block_items[m][0] != schedule[-1][0]), 1)
                block_items[0], block_items[swap_k] = block_items[swap_k], block_items[0]
            schedule.extend(block_items)
        return schedule[:n_trials]

    def run_session_at_points(
        self,
        listener: MultiSNRVirtualListener,
        evaluation_points: list[int],
        seed: int,
        fit_psychometric: bool = True,
    ) -> dict[int, MultiSNRSessionOutput]:
        rng = np.random.default_rng(seed)
        max_trials = max(evaluation_points)
        points_set = set(evaluation_points)
        k_slots = self.design.n_slots

        cell_counts = np.zeros((N_STIMULI, k_slots, N_RESPONSES), dtype=float)
        tok_counts = np.zeros((N_STIMULI, 2), dtype=int)
        slot_counts = np.zeros((N_STIMULI, k_slots), dtype=int)

        # Gaussian-shaped prior weights over each consonant's threshold grid
        mid_tau = self._tau_grids[:, self._n_grid // 2][:, None]
        log_w = -0.5 * ((self._tau_grids - mid_tau) / 7.5) ** 2
        tau_weights = np.exp(log_w)
        tau_weights /= tau_weights.sum(axis=1, keepdims=True)

        if self.mode == "block":
            queue = self._build_stratified_schedule(rng, max_trials)
        elif self.mode == "random":
            queue = [
                (int(rng.integers(0, N_STIMULI)), int(rng.integers(0, 2)), int(rng.integers(0, k_slots)))
                for _ in range(max_trials)
            ]
        else:
            # Warm-start 10 trials: 1 presentation of each consonant at middle SNR slot
            mid_slot = k_slots // 2
            c_order = rng.permutation(N_STIMULI)
            queue = [(int(c), int(rng.integers(0, 2)), mid_slot) for c in c_order[: min(N_STIMULI, max_trials)]]

        last_c: Optional[int] = None
        theta_warm: Optional[np.ndarray] = None
        outputs: dict[int, MultiSNRSessionOutput] = {}

        for trial in range(1, max_trials + 1):
            if queue:
                c_idx, tok_idx, s_idx = queue.pop(0)
            elif self.mode == "adaptive_consonant":
                row_counts = cell_counts.sum(axis=1)  # (10, 11)
                alpha = self._weak_prior + row_counts
                a_sum = alpha.sum(axis=1)
                p_hat = alpha / a_sum[:, None]
                scores = (1.0 - np.sum(p_hat * p_hat, axis=1)) / ((a_sum + 1.0) ** 2)
                scores += rng.uniform(0.0, 1e-12, size=N_STIMULI)
                if last_c is not None:
                    scores[last_c] = -1.0
                c_idx = int(np.argmax(scores))
                # Pick least-used SNR slot for this consonant (breaking ties toward middle)
                min_sc = int(np.min(slot_counts[c_idx]))
                cand_slots = [s for s in range(k_slots) if slot_counts[c_idx, s] == min_sc]
                s_idx = int(cand_slots[len(cand_slots) // 2])
                tok_idx = 0 if tok_counts[c_idx, 0] <= tok_counts[c_idx, 1] else 1
            elif self.mode == "adaptive_2d_snr":
                # Evaluate 1-step expected posterior variance reduction in tau_i
                # combined with confusion-row uncertainty for every (c_idx, s_idx)
                row_counts = cell_counts.sum(axis=1)
                alpha = self._weak_prior + row_counts
                a_sum = alpha.sum(axis=1)
                p_hat = alpha / a_sum[:, None]
                row_gini = 1.0 - np.sum(p_hat * p_hat, axis=1)  # (10,)

                best_score = -1.0
                c_idx, s_idx = 0, k_slots // 2
                for ci in range(N_STIMULI):
                    if last_c is not None and ci == last_c:
                        continue
                    w = tau_weights[ci]                # (G,)
                    tg = self._tau_grids[ci]           # (G,)
                    cur_mean = float(w @ tg)
                    cur_var = float(w @ ((tg - cur_mean) ** 2))

                    for si in range(k_slots):
                        psi = self._psi_lookup[ci, si]  # P(correct | tau_g)
                        p_corr = float(w @ psi)
                        p_err = max(1.0 - p_corr, 1e-12)
                        p_corr_safe = max(p_corr, 1e-12)

                        w1 = (w * psi) / p_corr_safe
                        m1 = float(w1 @ tg)
                        v1 = float(w1 @ ((tg - m1) ** 2))

                        w0 = (w * (1.0 - psi)) / p_err
                        m0 = float(w0 @ tg)
                        v0 = float(w0 @ ((tg - m0) ** 2))

                        exp_var = p_corr * v1 + (1.0 - p_corr) * v0
                        var_red = max(0.0, cur_var - exp_var)

                        # Combine threshold information with confusion-zone bonus p_corr*(1-p_corr)
                        transition_weight = 4.0 * p_corr * (1.0 - p_corr)
                        score = var_red + 2.5 * transition_weight * (row_gini[ci] / ((a_sum[ci] + 1.0) ** 2))
                        score += float(rng.uniform(0.0, 1e-12))
                        if score > best_score:
                            best_score = score
                            c_idx, s_idx = ci, si

                tok_idx = 0 if tok_counts[c_idx, 0] <= tok_counts[c_idx, 1] else 1
            else:
                raise RuntimeError(f"Unhandled mode {self.mode}")

            snr_db = float(self.design.snr_grid[c_idx, s_idx])
            resp_idx = listener.respond(c_idx, snr_db, token_idx=tok_idx)

            cell_counts[c_idx, s_idx, resp_idx] += 1.0
            tok_counts[c_idx, tok_idx] += 1
            slot_counts[c_idx, s_idx] += 1
            last_c = c_idx

            # Update discrete tau posterior for consonant c_idx
            psi_vec = self._psi_lookup[c_idx, s_idx]
            like = psi_vec if resp_idx == c_idx else (1.0 - psi_vec)
            tau_weights[c_idx] *= like
            tau_weights[c_idx] /= max(float(np.sum(tau_weights[c_idx])), 1e-30)

            if trial in points_set:
                row_counts = cell_counts.sum(axis=1)
                row_tots = row_counts.sum(axis=1, keepdims=True)
                uniform = np.full((N_STIMULI, N_RESPONSES), 1.0 / N_RESPONSES)
                emp = np.where(row_tots > 0, row_counts / np.maximum(row_tots, 1.0), uniform)
                diri = normalize_rows(self._weak_prior + row_counts)
                if fit_psychometric:
                    psych = fit_multisnr_psychometric_model(cell_counts, self.design.snr_grid, theta_init=theta_warm)
                    theta_warm = psych.theta
                else:
                    psych = FittedPsychometricModel(theta=THETA18_PRIOR_MEAN)
                outputs[trial] = MultiSNRSessionOutput(
                    cell_counts=cell_counts.copy(),
                    pooled_empirical=emp,
                    pooled_dirichlet=diri,
                    psychometric_fit=psych,
                )

        return outputs
