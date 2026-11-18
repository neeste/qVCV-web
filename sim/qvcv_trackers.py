"""
qvcv_trackers.py

Stimulus-selection policies and confusion-matrix estimators for qVCV.
Designed with a clean adapter interface mirroring qCLS's ImplementationAdapter
so every tracker can be evaluated identically in headless simulations and
ported directly to client-side JavaScript if desired.

Supported selection modes (--mode):
  - block            : Exact qVCV/app.js 5x20 shuffled block randomization
  - random           : Uniform random stimulus selection (control arm)
  - adaptive_var     : Bayesian Dirichlet myopic global MSE-reduction tracker
  - adaptive_eig     : Bayesian Dirichlet Expected Information Gain (BALD) tracker
  - adaptive_feature : Coupled Phonetic-Feature Empirical-Bayes + Dirichlet tracker

Supported estimators (--estimator):
  - auto       : Natural estimator for each mode ('empirical' for block/random
                 to match current app.js; 'dirichlet' for adaptive_var/eig;
                 'feature_eb' for adaptive_feature)
  - empirical  : Raw row proportions c_{i,j} / n_i (current qVCV/app.js output)
  - dirichlet  : Weakly-informative phonetic Dirichlet posterior mean
  - feature_eb : Coupled phonetic-feature Empirical-Bayes shrinkage estimator
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    from .qvcv_simulator import (
        DISPLAY_CONSONANTS,
        FILES_BY_CONSONANT,
        N_RESPONSES,
        N_STIMULI,
        PHONETIC_FEATURES,
        STIM_INDEX,
        STIMULI_CONSONANTS,
        TEST_FILES_BASE,
        VirtualListener,
        normalize_rows,
        parse_consonant,
        parse_token_index,
    )
except ImportError:
    from qvcv_simulator import (
        DISPLAY_CONSONANTS,
        FILES_BY_CONSONANT,
        N_RESPONSES,
        N_STIMULI,
        PHONETIC_FEATURES,
        STIM_INDEX,
        STIMULI_CONSONANTS,
        TEST_FILES_BASE,
        VirtualListener,
        normalize_rows,
        parse_consonant,
        parse_token_index,
    )

SELECTION_MODES = (
    "block",
    "random",
    "adaptive_var",
    "adaptive_eig",
    "adaptive_feature",
)
ESTIMATOR_MODES = ("auto", "empirical", "dirichlet", "feature_eb")


# --------------------------------------------------------------------------
# Numerical helpers (pure NumPy; zero external dependencies)
# --------------------------------------------------------------------------

def _digamma(x: np.ndarray) -> np.ndarray:
    """Vectorized digamma psi(x) accurate to <1e-13 for x > 0."""
    x = np.asarray(x, dtype=float).copy()
    result = np.zeros_like(x)
    while np.any(x < 8.0):
        mask = x < 8.0
        result[mask] -= 1.0 / x[mask]
        x[mask] += 1.0
    inv_x = 1.0 / x
    inv_x2 = inv_x * inv_x
    result += (
        np.log(x)
        - 0.5 * inv_x
        - inv_x2 * (1.0 / 12.0 - inv_x2 * (1.0 / 120.0 - inv_x2 * (1.0 / 252.0 - inv_x2 / 240.0)))
    )
    return result


def build_weak_phonetic_prior(concentration: float = 1.5) -> np.ndarray:
    """
    Construct a weakly-informative Dirichlet prior alpha_0 of shape (10, 11)
    using only generic articulatory feature overlap (Miller & Nicely 1955),
    without any knowledge of synthetic listener parameters.

    Row sums equal ``concentration`` (default 1.5 pseudo-trials spread across
    11 categories), gently favoring diagonal + phonetically similar responses
    over impossible cross-feature errors when n_i is very small.
    """
    weights = np.full((N_STIMULI, N_RESPONSES), 0.15, dtype=float)
    for i, sc in enumerate(STIMULI_CONSONANTS):
        sf = PHONETIC_FEATURES[sc]
        for j, rc in enumerate(DISPLAY_CONSONANTS):
            if rc == "???":
                weights[i, j] = 0.35
            elif sc == rc:
                weights[i, j] = 4.50
            else:
                rf = PHONETIC_FEATURES[rc]
                w = 0.12
                if sf["voicing"] == rf["voicing"]:
                    w += 0.25
                if sf["manner"] == rf["manner"]:
                    w += 0.30
                if sf["place"] == rf["place"]:
                    w += 0.20
                if sf["sibilant"] == 1 and rf["sibilant"] == 1:
                    w += 0.25
                weights[i, j] = w
    return concentration * normalize_rows(weights)


def _build_feature_design_tensor() -> np.ndarray:
    """
    Build the fixed (10, 11, 8) phonetic-feature design tensor X for the
    strictly concave multinomial log-linear choice model:
      0: self_lf   (correct ID for voiced/higher-SNR tokens: B, D, G, N, V)
      1: self_hf   (correct ID for 0-dB HF tokens: K, S, SH, T, Z)
      2: voicing   (off-diagonal shared voicing)
      3: manner    (off-diagonal shared manner)
      4: place     (off-diagonal shared place)
      5: sibilant  (off-diagonal shared sibilance)
      6: unk_lf    ('???' catch-all on LF tokens)
      7: unk_hf    ('???' catch-all on HF tokens)
    """
    x = np.zeros((N_STIMULI, N_RESPONSES, 8), dtype=float)
    for i, sc in enumerate(STIMULI_CONSONANTS):
        sf = PHONETIC_FEATURES[sc]
        is_hf = bool(sf["hf_cue"])
        for j, rc in enumerate(DISPLAY_CONSONANTS):
            if rc == "???":
                x[i, j, 7 if is_hf else 6] = 1.0
            elif sc == rc:
                x[i, j, 1 if is_hf else 0] = 1.0
            else:
                rf = PHONETIC_FEATURES[rc]
                if sf["voicing"] == rf["voicing"]:
                    x[i, j, 2] = 1.0
                if sf["manner"] == rf["manner"]:
                    x[i, j, 3] = 1.0
                if sf["place"] == rf["place"]:
                    x[i, j, 4] = 1.0
                if sf["sibilant"] == 1 and rf["sibilant"] == 1:
                    x[i, j, 5] = 1.0
    return x


FEATURE_TENSOR_X: np.ndarray = _build_feature_design_tensor()
# Weakly-regularized neutral prior center for phi
PHI_PRIOR_MEAN: np.ndarray = np.array([3.2, 2.5, 0.7, 0.8, 0.4, 0.6, 0.2, 0.6], dtype=float)
PHI_PRIOR_PREC: float = 1.0 / (2.25 ** 2)  # Gaussian prior variance = 2.25^2


def fit_feature_choice_model(
    counts: np.ndarray,
    phi_init: Optional[np.ndarray] = None,
    max_iter: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the unique global MAP parameter vector phi_hat (8,) and predicted
    row-stochastic confusion matrix mu(phi_hat) (10, 11) via Newton-Raphson.
    """
    phi = PHI_PRIOR_MEAN.copy() if phi_init is None else phi_init.copy()
    row_totals = counts.sum(axis=1)  # (10,)
    reg_eye = PHI_PRIOR_PREC * np.eye(len(phi))

    def _eval_probs_and_obj(p_vec: np.ndarray) -> tuple[np.ndarray, float]:
        logits = np.einsum("ijm,m->ij", FEATURE_TENSOR_X, p_vec)
        logits -= logits.max(axis=1, keepdims=True)
        exp_u = np.exp(logits)
        mu = exp_u / exp_u.sum(axis=1, keepdims=True)
        diff = p_vec - PHI_PRIOR_MEAN
        ll = float(np.sum(counts * np.log(np.maximum(mu, 1e-15))) - 0.5 * PHI_PRIOR_PREC * (diff @ diff))
        return mu, ll

    mu, obj = _eval_probs_and_obj(phi)
    if float(row_totals.sum()) <= 0.0:
        return phi, mu

    for _ in range(max_iter):
        # Gradient (8,)
        resid = counts - row_totals[:, None] * mu
        grad = np.einsum("ij,ijm->m", resid, FEATURE_TENSOR_X) - PHI_PRIOR_PREC * (phi - PHI_PRIOR_MEAN)
        if float(np.max(np.abs(grad))) < 1e-7:
            break

        # Exact SPD negative Hessian (8, 8)
        x_bar = np.einsum("ij,ijm->im", mu, FEATURE_TENSOR_X)  # (10, 8)
        x_centered = FEATURE_TENSOR_X - x_bar[:, None, :]      # (10, 11, 8)
        weights = row_totals[:, None] * mu                      # (10, 11)
        neg_hess = np.einsum("ij,ijm,ijn->mn", weights, x_centered, x_centered) + reg_eye

        step = np.linalg.solve(neg_hess, grad)
        # Backtracking line search
        alpha = 1.0
        while alpha >= 1e-3:
            cand_phi = phi + alpha * step
            cand_mu, cand_obj = _eval_probs_and_obj(cand_phi)
            if cand_obj >= obj - 1e-10:
                phi, mu, obj = cand_phi, cand_mu, cand_obj
                break
            alpha *= 0.5
        else:
            break

    return phi, mu


def estimate_eb_concentration(
    counts: np.ndarray,
    mu: np.ndarray,
    kappa_min: float = 1.25,
    kappa_max: float = 4.50,
    kappa_default: float = 3.25,
) -> float:
    """
    Empirical-Bayes Dirichlet prior concentration kappa via Pearson dispersion.
    When observed counts follow mu closely (or sample size is tiny), keeps kappa
    near ~3.5-4.5 to borrow phonetic strength across consonants; when a listener
    exhibits strong idiosyncratic non-feature confusions, relaxes kappa down
    toward kappa_min so row counts dominate rapidly.
    """
    row_totals = counts.sum(axis=1)
    active = row_totals >= 3.0
    n_active = int(np.sum(active))
    if n_active < 3:
        return kappa_default

    expected = row_totals[active, None] * np.maximum(mu[active], 1e-3)
    pearson_stat = float(np.sum((counts[active] - expected) ** 2 / expected))
    # Effective degrees of freedom accounting for 8 fitted feature parameters
    df = max(1.0, float(n_active * (N_RESPONSES - 1) - 8))
    dispersion = pearson_stat / df
    if dispersion <= 1.05:
        return kappa_max
    mean_n = float(np.mean(row_totals[active]))
    if mean_n <= dispersion:
        return kappa_min
    kappa_hat = (mean_n - dispersion) / (dispersion - 1.0)
    return float(np.clip(kappa_hat, kappa_min, kappa_max))


# --------------------------------------------------------------------------
# Session Result & Tracker Adapter
# --------------------------------------------------------------------------

@dataclass
class SessionResult:
    stimuli: list[str]                  # Stimulus filename per trial
    targets: list[int]                  # Target consonant index (0..9) per trial
    responses: list[int]                # Listener response index (0..10) per trial
    counts: np.ndarray                  # (10, 11) raw response counts
    fitted_matrix: np.ndarray           # (10, 11) estimated confusion probability matrix


class QVCVTrackerAdapter:
    """
    Unified tracker and estimator implementing qVCV stimulus selection and
    confusion-matrix recovery.
    """

    def __init__(
        self,
        mode: str = "block",
        estimator: str = "auto",
        prior_concentration: float = 1.5,
        avoid_consecutive_repeat: bool = True,
    ):
        if mode not in SELECTION_MODES:
            raise ValueError(f"Unknown mode {mode!r}; expected one of {SELECTION_MODES}")
        if estimator not in ESTIMATOR_MODES:
            raise ValueError(f"Unknown estimator {estimator!r}; expected one of {ESTIMATOR_MODES}")

        self.mode = mode
        if estimator == "auto":
            if mode in ("block", "random"):
                self.estimator = "empirical"
            elif mode == "adaptive_feature":
                self.estimator = "feature_eb"
            else:
                self.estimator = "dirichlet"
        else:
            self.estimator = estimator

        self.prior_concentration = prior_concentration
        self.avoid_consecutive_repeat = avoid_consecutive_repeat
        self._weak_prior = build_weak_phonetic_prior(concentration=prior_concentration)

        # Per-session mutable state
        self.rng: np.random.Generator = np.random.default_rng(0)
        self.counts: np.ndarray = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
        self.token_counts: np.ndarray = np.zeros((N_STIMULI, 2), dtype=int)
        self._queue: list[str] = []
        self._pending_stimulus: Optional[str] = None
        self._last_consonant_idx: Optional[int] = None
        self._trial_idx: int = 0
        self._phi_hat: np.ndarray = PHI_PRIOR_MEAN.copy()
        self._mu_hat: np.ndarray = normalize_rows(self._weak_prior)

    def start_session(self, seed: int, n_trials: int) -> None:
        self.rng = np.random.default_rng(seed)
        self.counts.fill(0.0)
        self.token_counts.fill(0)
        self._pending_stimulus = None
        self._last_consonant_idx = None
        self._trial_idx = 0
        self._phi_hat = PHI_PRIOR_MEAN.copy()
        _, self._mu_hat = fit_feature_choice_model(self.counts, self._phi_hat, max_iter=1)

        if self.mode == "block":
            # Exact qVCV/app.js shuffled 20-stimulus blocks
            n_blocks = max(1, (n_trials + len(TEST_FILES_BASE) - 1) // len(TEST_FILES_BASE))
            full_queue: list[str] = []
            for _ in range(n_blocks):
                block = list(TEST_FILES_BASE)
                self.rng.shuffle(block)
                # Prevent accidental boundary repeat between consecutive blocks
                if (
                    full_queue
                    and len(block) > 1
                    and parse_consonant(block[0]) == parse_consonant(full_queue[-1])
                ):
                    swap_idx = next(
                        (
                            k for k in range(1, len(block))
                            if parse_consonant(block[k]) != parse_consonant(full_queue[-1])
                        ),
                        1,
                    )
                    block[0], block[swap_idx] = block[swap_idx], block[0]
                full_queue.extend(block)
            self._queue = full_queue[:n_trials]
        elif self.mode == "random":
            idxs = self.rng.integers(0, len(TEST_FILES_BASE), size=n_trials)
            self._queue = [TEST_FILES_BASE[int(i)] for i in idxs]
        else:
            # Adaptive modes: warm-start with 1 shuffled presentation of each consonant
            order = list(range(N_STIMULI))
            self.rng.shuffle(order)
            warm_files: list[str] = []
            for c_idx in order[: min(N_STIMULI, n_trials)]:
                tok = int(self.rng.integers(0, 2))
                warm_files.append(FILES_BY_CONSONANT[c_idx][tok])
            self._queue = warm_files

    def _choose_token_file(self, consonant_idx: int) -> str:
        """Balance presentations across _1_ and _2_ talker tokens for a consonant."""
        c0, c1 = self.token_counts[consonant_idx]
        if c0 < c1:
            tok = 0
        elif c1 < c0:
            tok = 1
        else:
            tok = int(self.rng.integers(0, 2))
        return FILES_BY_CONSONANT[consonant_idx][tok]

    def _current_dirichlet_alpha(self, for_selection: bool = False) -> np.ndarray:
        """Return (10, 11) Dirichlet posterior parameters alpha."""
        use_feature = (
            (self.mode == "adaptive_feature")
            if for_selection
            else (self.estimator == "feature_eb")
        )
        if use_feature:
            kappa = estimate_eb_concentration(self.counts, self._mu_hat)
            return kappa * self._mu_hat + self.counts
        return self._weak_prior + self.counts

    def _select_adaptive_consonant(self) -> int:
        alpha = self._current_dirichlet_alpha(for_selection=True)
        a_sum = alpha.sum(axis=1)                       # (10,)
        p_hat = alpha / a_sum[:, None]                  # (10, 11)

        if self.mode in ("adaptive_var", "adaptive_feature"):
            # Exact one-step expected reduction in sum of posterior variances:
            # Delta_MSE_i = (1 - sum_j p_hat_{i,j}^2) / (A_i + 1)^2
            row_gini = 1.0 - np.sum(p_hat * p_hat, axis=1)
            scores = row_gini / ((a_sum + 1.0) ** 2)
        elif self.mode == "adaptive_eig":
            # Exact Expected Information Gain (BALD): H(p_hat_i) - E_{theta_i}[H(theta_i)]
            h_mean = -np.sum(p_hat * np.log(np.maximum(p_hat, 1e-15)), axis=1)
            e_h = -np.sum(p_hat * (_digamma(alpha + 1.0) - _digamma(a_sum + 1.0)[:, None]), axis=1)
            scores = np.maximum(0.0, h_mean - e_h)
        else:
            raise RuntimeError(f"Unexpected adaptive mode {self.mode!r}")

        # Break exact ties randomly; optionally avoid immediate consecutive consonant repeat
        scores = scores + self.rng.uniform(0.0, 1e-12, size=N_STIMULI)
        if self.avoid_consecutive_repeat and self._last_consonant_idx is not None:
            scores[self._last_consonant_idx] = -1.0

        return int(np.argmax(scores))

    def next_stimulus(self) -> str:
        if self._queue:
            stim = self._queue.pop(0)
        else:
            c_idx = self._select_adaptive_consonant()
            stim = self._choose_token_file(c_idx)

        self._pending_stimulus = stim
        return stim

    def submit_response(self, response_idx: int) -> None:
        if self._pending_stimulus is None:
            raise RuntimeError("submit_response called before next_stimulus")
        if not (0 <= response_idx < N_RESPONSES):
            raise ValueError(f"response_idx {response_idx} out of range 0..{N_RESPONSES - 1}")

        c_idx = STIM_INDEX[parse_consonant(self._pending_stimulus)]
        tok_idx = parse_token_index(self._pending_stimulus)

        self.counts[c_idx, response_idx] += 1.0
        self.token_counts[c_idx, tok_idx] += 1
        self._last_consonant_idx = c_idx
        self._pending_stimulus = None
        self._trial_idx += 1

        if self.mode == "adaptive_feature":
            self._phi_hat, self._mu_hat = fit_feature_choice_model(
                self.counts, self._phi_hat, max_iter=4
            )

    def evaluate_fit(self) -> np.ndarray:
        """Return (10, 11) estimated confusion probability matrix at current trial count."""
        if self.estimator == "empirical":
            row_totals = self.counts.sum(axis=1, keepdims=True)
            uniform = np.full((N_STIMULI, N_RESPONSES), 1.0 / N_RESPONSES)
            return np.where(row_totals > 0.0, self.counts / np.maximum(row_totals, 1.0), uniform)

        if self.estimator == "dirichlet":
            alpha = self._weak_prior + self.counts
            return normalize_rows(alpha)

        if self.estimator == "feature_eb":
            phi_fit, mu_fit = fit_feature_choice_model(self.counts, self._phi_hat)
            self._phi_hat, self._mu_hat = phi_fit, mu_fit
            kappa = estimate_eb_concentration(self.counts, mu_fit)
            return normalize_rows(kappa * mu_fit + self.counts)

        raise RuntimeError(f"Unknown estimator {self.estimator!r}")

    def get_fit(self) -> np.ndarray:
        return self.evaluate_fit()

    def run_session(
        self,
        listener: VirtualListener,
        n_trials: int,
        seed: int,
    ) -> SessionResult:
        self.start_session(seed=seed, n_trials=n_trials)
        stimuli: list[str] = []
        targets: list[int] = []
        responses: list[int] = []

        for _ in range(n_trials):
            stim = self.next_stimulus()
            resp = listener.respond(stim)
            self.submit_response(resp)
            stimuli.append(stim)
            targets.append(STIM_INDEX[parse_consonant(stim)])
            responses.append(resp)

        return SessionResult(
            stimuli=stimuli,
            targets=targets,
            responses=responses,
            counts=self.counts.copy(),
            fitted_matrix=self.get_fit(),
        )

    def run_session_at_points(
        self,
        listener: VirtualListener,
        evaluation_points: list[int],
        seed: int,
    ) -> dict[int, np.ndarray]:
        """Run one cumulative session and record confusion-matrix estimates at requested trial counts."""
        max_trials = max(evaluation_points)
        self.start_session(seed=seed, n_trials=max_trials)
        points = set(evaluation_points)
        fits: dict[int, np.ndarray] = {}

        for trial in range(1, max_trials + 1):
            stim = self.next_stimulus()
            resp = listener.respond(stim)
            self.submit_response(resp)
            if trial in points:
                fits[trial] = self.evaluate_fit()
        return fits
