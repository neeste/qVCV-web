"""
qvcv_simulator.py

Virtual-listener model, phonetic-feature tables, and ground-truth catalog
generator/loader for the qVCV Consonant Confusion Test.

Supports loading listener ground-truth confusion matrices (10 stimuli x 11
response categories) from:
  1. Profile CSV (``virtual_listeners.csv``; default synthetic clinical cohort)
  2. Profile JSON (``virtual_listeners.json``)
  3. Empirical web-app CSV export(s) (``qVCV_[ID]_[Ear]_[YYYYMMDD].csv`` or a
     directory of exported session CSVs)
  4. MATLAB ``.mat`` files (via ``scipy.io.loadmat``)
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

# Exact stimulus & response ordering from qVCV/app.js
STIMULI_CONSONANTS: list[str] = ["B", "D", "G", "K", "N", "S", "SH", "T", "V", "Z"]
DISPLAY_CONSONANTS: list[str] = ["B", "D", "G", "K", "N", "S", "SH", "T", "V", "Z", "???"]

STIM_INDEX: dict[str, int] = {c: idx for idx, c in enumerate(STIMULI_CONSONANTS)}
RESP_INDEX: dict[str, int] = {c: idx for idx, c in enumerate(DISPLAY_CONSONANTS)}

N_STIMULI: int = len(STIMULI_CONSONANTS)      # 10
N_RESPONSES: int = len(DISPLAY_CONSONANTS)    # 11

# Exact 20 test stimuli from qVCV/app.js (2 talker tokens per consonant, fixed SNR)
TEST_FILES_BASE: list[str] = [
    "VCV_aba_2_18SNR.wav", "VCV_aga_1_3SNR.wav", "VCV_aka_2_0SNR.wav", "VCV_asa_1_0SNR.wav",
    "VCV_asha_2_0SNR.wav", "VCV_ava_1_18SNR.wav", "VCV_aza_2_0SNR.wav", "VCV_ada_1_4SNR.wav",
    "VCV_aga_2_3SNR.wav", "VCV_ana_1_15SNR.wav", "VCV_asa_2_0SNR.wav", "VCV_ata_1_0SNR.wav",
    "VCV_ava_2_18SNR.wav", "VCV_aba_1_18SNR.wav", "VCV_ada_2_4SNR.wav", "VCV_aka_1_0SNR.wav",
    "VCV_ana_2_15SNR.wav", "VCV_asha_1_0SNR.wav", "VCV_ata_2_0SNR.wav", "VCV_aza_1_0SNR.wav",
]

# Embedded SNR (dB) per target consonant in Stimuli/test/
CONSONANT_SNR_DB: dict[str, float] = {
    "B": 18.0,
    "D": 4.0,
    "G": 3.0,
    "K": 0.0,
    "N": 15.0,
    "S": 0.0,
    "SH": 0.0,
    "T": 0.0,
    "V": 18.0,
    "Z": 0.0,
}

# Canonical articulatory / acoustic phonetic features (Miller & Nicely 1955)
PHONETIC_FEATURES: dict[str, dict[str, str | int]] = {
    "B":  {"voicing": "voiced",    "manner": "stop",      "place": "labial",       "sibilant": 0, "hf_cue": 0},
    "D":  {"voicing": "voiced",    "manner": "stop",      "place": "alveolar",     "sibilant": 0, "hf_cue": 0},
    "G":  {"voicing": "voiced",    "manner": "stop",      "place": "velar",        "sibilant": 0, "hf_cue": 0},
    "K":  {"voicing": "voiceless", "manner": "stop",      "place": "velar",        "sibilant": 0, "hf_cue": 1},
    "N":  {"voicing": "voiced",    "manner": "nasal",     "place": "alveolar",     "sibilant": 0, "hf_cue": 0},
    "S":  {"voicing": "voiceless", "manner": "fricative", "place": "alveolar",     "sibilant": 1, "hf_cue": 1},
    "SH": {"voicing": "voiceless", "manner": "fricative", "place": "postalveolar", "sibilant": 1, "hf_cue": 1},
    "T":  {"voicing": "voiceless", "manner": "stop",      "place": "alveolar",     "sibilant": 0, "hf_cue": 1},
    "V":  {"voicing": "voiced",    "manner": "fricative", "place": "labial",       "sibilant": 0, "hf_cue": 0},
    "Z":  {"voicing": "voiced",    "manner": "fricative", "place": "alveolar",     "sibilant": 1, "hf_cue": 1},
}

# Groupings of test filenames by consonant index (sorted token_1, token_2)
FILES_BY_CONSONANT: list[list[str]] = [
    sorted([f for f in TEST_FILES_BASE if f.split("_")[1][1:-1].upper() == c])
    for c in STIMULI_CONSONANTS
]


def parse_consonant(filename: str) -> str:
    """Replicate qVCV/app.js parseConsonant('VCV_asha_1_0SNR.wav') -> 'SH'."""
    part = filename.split("_")[1]
    return part[1:-1].upper()


def parse_token_index(filename: str) -> int:
    """Return 0 for '_1_' token and 1 for '_2_' token."""
    parts = filename.split("_")
    if len(parts) >= 3 and parts[2] in ("1", "2"):
        return int(parts[2]) - 1
    return 0


def normalize_rows(matrix: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Ensure a (10, 11) matrix is strictly non-negative and row-stochastic."""
    clipped = np.maximum(np.asarray(matrix, dtype=float), 0.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    uniform = np.full_like(clipped, 1.0 / clipped.shape[1])
    return np.where(row_sums > eps, clipped / np.maximum(row_sums, eps), uniform)


# --------------------------------------------------------------------------
# Phonetic Information Transmission (Miller & Nicely 1955)
# --------------------------------------------------------------------------

def _mutual_information_bits(joint: np.ndarray) -> tuple[float, float]:
    """Return (I(X; Y), H(X)) in bits from a non-negative joint table."""
    joint = np.maximum(joint, 0.0)
    total = joint.sum()
    if total <= 0.0:
        return 0.0, 0.0
    p_xy = joint / total
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)

    mask_x = p_x[:, 0] > 0.0
    h_x = -float(np.sum(p_x[mask_x, 0] * np.log2(p_x[mask_x, 0])))

    denom = p_x @ p_y
    valid = (p_xy > 0.0) & (denom > 0.0)
    mi = float(np.sum(p_xy[valid] * np.log2(p_xy[valid] / denom[valid])))
    return max(0.0, mi), h_x


def relative_information_transmitted(conf_matrix: np.ndarray) -> dict[str, float]:
    """
    Compute Miller & Nicely (1955) Relative Information Transmitted (%) for:
      - overall (all 10 consonants -> 11 responses)
      - voicing (voiced vs voiceless; '???' treated as a distinct unclassified response)
      - manner  (stop, fricative, nasal; '???' distinct)
      - place   (labial, alveolar, postalveolar, velar; '???' distinct)

    Assumes uniform stimulus prior p(s) = 1/10 so the metric characterizes
    the listener's channel matrix independent of how many trials each stimulus
    received during adaptive testing.
    """
    p_cond = normalize_rows(conf_matrix)
    joint_overall = p_cond / float(N_STIMULI)
    mi_all, h_all = _mutual_information_bits(joint_overall)
    out = {"overall": 100.0 * (mi_all / h_all) if h_all > 0 else 0.0}

    for feat_name in ("voicing", "manner", "place"):
        stim_classes = sorted({str(PHONETIC_FEATURES[c][feat_name]) for c in STIMULI_CONSONANTS})
        resp_classes = stim_classes + ["unknown"]
        s_map = {k: idx for idx, k in enumerate(stim_classes)}
        r_map = {k: idx for idx, k in enumerate(resp_classes)}

        joint_feat = np.zeros((len(stim_classes), len(resp_classes)), dtype=float)
        for i, stim_c in enumerate(STIMULI_CONSONANTS):
            si = s_map[str(PHONETIC_FEATURES[stim_c][feat_name])]
            for j, resp_c in enumerate(DISPLAY_CONSONANTS):
                rk = "unknown" if resp_c == "???" else str(PHONETIC_FEATURES[resp_c][feat_name])
                rj = r_map[rk]
                joint_feat[si, rj] += joint_overall[i, j]

        mi_f, h_f = _mutual_information_bits(joint_feat)
        out[feat_name] = 100.0 * (mi_f / h_f) if h_f > 0 else 0.0

    return out


# --------------------------------------------------------------------------
# Virtual Listener & Ground Truth Profiles
# --------------------------------------------------------------------------

class VirtualListener:
    """Simulate categorical consonant responses from a ground-truth listener profile."""

    def __init__(
        self,
        listener_id: str,
        gt_matrix: np.ndarray,
        archetype: str = "custom",
        description: str = "",
        token_matrices: Optional[np.ndarray] = None,
        rng: Optional[np.random.Generator] = None,
    ):
        self.listener_id = listener_id
        self.archetype = archetype
        self.description = description
        self.gt_matrix = normalize_rows(gt_matrix)
        if self.gt_matrix.shape != (N_STIMULI, N_RESPONSES):
            raise ValueError(
                f"Listener {listener_id} gt_matrix has shape {self.gt_matrix.shape}, "
                f"expected ({N_STIMULI}, {N_RESPONSES})"
            )
        if token_matrices is not None:
            tm = np.asarray(token_matrices, dtype=float)
            if tm.shape != (2, N_STIMULI, N_RESPONSES):
                raise ValueError(
                    f"token_matrices must have shape (2, {N_STIMULI}, {N_RESPONSES})"
                )
            self.token_matrices = np.stack([normalize_rows(tm[0]), normalize_rows(tm[1])], axis=0)
            # Keep gt_matrix exactly equal to the equal-token expectation
            self.gt_matrix = 0.5 * (self.token_matrices[0] + self.token_matrices[1])
        else:
            self.token_matrices = np.stack([self.gt_matrix, self.gt_matrix], axis=0)

        self.rng = rng if rng is not None else np.random.default_rng()
        self._cdfs = np.cumsum(self.token_matrices, axis=2)
        self._cdfs[:, :, -1] = 1.0

    def respond(self, stimulus: str | int) -> int:
        """
        Draw a simulated response index in 0..10 (DISPLAY_CONSONANTS)
        for either a stimulus filename ('VCV_asa_1_0SNR.wav'), consonant label ('S'),
        or consonant index (0..9).
        """
        if isinstance(stimulus, int):
            stim_idx = stimulus
            tok_idx = int(self.rng.integers(0, 2))
        elif stimulus.endswith(".wav") or "_" in stimulus:
            stim_idx = STIM_INDEX[parse_consonant(stimulus)]
            tok_idx = parse_token_index(stimulus)
        else:
            stim_idx = STIM_INDEX[stimulus.upper()]
            tok_idx = int(self.rng.integers(0, 2))

        u = float(self.rng.random())
        resp_idx = int(np.searchsorted(self._cdfs[tok_idx, stim_idx], u, side="right"))
        return min(resp_idx, N_RESPONSES - 1)

    def respond_label(self, stimulus: str | int) -> str:
        return DISPLAY_CONSONANTS[self.respond(stimulus)]

    def full_profile(self) -> np.ndarray:
        """Return a copy of the (10, 11) ground-truth confusion probability matrix."""
        return self.gt_matrix.copy()


@dataclass
class _ArchetypeSpec:
    archetype: str
    description: str
    # Perceptual sensitivity weights
    self_lf: float       # Correct ID boost on voiced/higher-SNR stimuli (B, D, G, N, V)
    self_hf: float       # Correct ID boost on 0-dB HF stimuli (K, S, SH, T, Z)
    w_voice: float       # Shared voicing similarity
    w_manner: float      # Shared manner similarity
    w_place: float       # Shared place similarity
    w_sibilant: float    # Sibilant-to-sibilant similarity
    unk_lf: float        # '???' log-odds on LF tokens
    unk_hf: float        # '???' log-odds on HF tokens
    asymmetry: dict[tuple[str, str], float]  # Asymmetric confusion quirks (stim, resp) -> delta logit
    dirichlet_conc: float  # Individual row perturbation concentration (smaller = more idiosyncratic)


def _build_synthetic_listener(
    listener_id: str,
    spec: _ArchetypeSpec,
    rng: np.random.Generator,
) -> VirtualListener:
    """
    Construct a realistic virtual listener with:
      1. Articulatory/acoustic similarity structure
      2. Asymmetric psychoacoustic confusions (e.g. G->D > D->G, T->K > K->T, V->B > B->V)
      3. Individual listener-level and talker-token (_1_ vs _2_) perturbations
         so ground truth is NOT restricted to any low-dimensional parametric family.
    """
    logits = np.full((N_STIMULI, N_RESPONSES), -3.2, dtype=float)

    # Individual jitter on perceptual sensitivities
    self_lf = spec.self_lf + rng.normal(0.0, 0.22)
    self_hf = spec.self_hf + rng.normal(0.0, 0.28)
    w_voice = max(0.0, spec.w_voice + rng.normal(0.0, 0.12))
    w_manner = max(0.0, spec.w_manner + rng.normal(0.0, 0.12))
    w_place = max(0.0, spec.w_place + rng.normal(0.0, 0.12))
    w_sib = max(0.0, spec.w_sibilant + rng.normal(0.0, 0.12))
    unk_lf = spec.unk_lf + rng.normal(0.0, 0.20)
    unk_hf = spec.unk_hf + rng.normal(0.0, 0.25)

    # Consonant-specific SNR difficulty adjustments (matching test file SNRs)
    snr_bonus = {
        "N": 0.85, "B": 0.45, "V": 0.15, "D": -0.25, "G": -0.45,
        "S": 0.15, "SH": 0.30, "T": 0.05, "K": -0.15, "Z": -0.20,
    }
    # Listener response bias across buttons (subtle natural preference)
    response_bias = rng.normal(0.0, 0.14, size=N_RESPONSES)

    for i, sc in enumerate(STIMULI_CONSONANTS):
        sf = PHONETIC_FEATURES[sc]
        is_hf = bool(sf["hf_cue"])
        for j, rc in enumerate(DISPLAY_CONSONANTS):
            if rc == "???":
                logits[i, j] = unk_hf if is_hf else unk_lf
                continue
            rf = PHONETIC_FEATURES[rc]
            if sc == rc:
                base_self = self_hf if is_hf else self_lf
                logits[i, j] = base_self + snr_bonus[sc] + rng.normal(0.0, 0.18)
            else:
                sim = -2.6
                if sf["voicing"] == rf["voicing"]:
                    sim += w_voice
                if sf["manner"] == rf["manner"]:
                    sim += w_manner
                if sf["place"] == rf["place"]:
                    sim += w_place
                if sf["sibilant"] == 1 and rf["sibilant"] == 1:
                    sim += w_sib
                logits[i, j] = sim
            logits[i, j] += response_bias[j]

    # Classic empirical VCV asymmetric confusions (Miller & Nicely 1955; Phatak & Allen 2007)
    base_asymmetries = {
        ("G", "D"): 0.75,   # /aga/ burst masked at 3 dB SNR often heard as /ada/
        ("D", "G"): 0.35,
        ("V", "B"): 0.85,   # Weak /v/ frication often heard as stop /b/
        ("B", "V"): 0.40,
        ("T", "K"): 0.60,   # /ata/ vs /aka/ burst confusion at 0 dB SNR
        ("K", "T"): 0.45,
        ("Z", "S"): 0.40,   # Devoicing in 0 dB noise
        ("Z", "D"): 0.45,
        ("S", "SH"): 0.45,
        ("SH", "S"): 0.35,
    }
    for (sc, rc), delta in base_asymmetries.items():
        logits[STIM_INDEX[sc], RESP_INDEX[rc]] += delta
    for (sc, rc), delta in spec.asymmetry.items():
        logits[STIM_INDEX[sc], RESP_INDEX[rc]] += delta

    # Softmax to mean probabilities, then sample Dirichlet row perturbations
    logits -= logits.max(axis=1, keepdims=True)
    mean_probs = np.exp(logits)
    mean_probs /= mean_probs.sum(axis=1, keepdims=True)

    gt_rows = np.zeros_like(mean_probs)
    tok0_rows = np.zeros_like(mean_probs)
    tok1_rows = np.zeros_like(mean_probs)

    for i in range(N_STIMULI):
        alpha_row = np.maximum(spec.dirichlet_conc * mean_probs[i], 0.03)
        base_row = rng.dirichlet(alpha_row)
        # Mild talker/utterance differences between _1_ and _2_ wav recordings
        tok0_rows[i] = rng.dirichlet(np.maximum(120.0 * base_row, 0.05))
        tok1_rows[i] = rng.dirichlet(np.maximum(120.0 * base_row, 0.05))
        gt_rows[i] = 0.5 * (tok0_rows[i] + tok1_rows[i])

    return VirtualListener(
        listener_id=listener_id,
        gt_matrix=gt_rows,
        archetype=spec.archetype,
        description=spec.description,
        token_matrices=np.stack([tok0_rows, tok1_rows], axis=0),
    )


def generate_synthetic_cohort(seed: int = 20260512) -> list[VirtualListener]:
    """
    Generate a diverse, clinically grounded cohort of 18 virtual listeners
    spanning 6 audiological profiles.
    """
    rng = np.random.default_rng(seed)

    specs: list[tuple[int, str, _ArchetypeSpec]] = [
        (
            3, "NH",
            _ArchetypeSpec(
                archetype="normal_hearing",
                description="Normal hearing: high overall ID (~85-92%), minor G/D, V/B, T/K confusions",
                self_lf=3.15, self_hf=2.85,
                w_voice=0.95, w_manner=1.05, w_place=0.55, w_sibilant=0.80,
                unk_lf=-3.4, unk_hf=-2.9,
                asymmetry={},
                dirichlet_conc=65.0,
            ),
        ),
        (
            3, "HF_Mild",
            _ArchetypeSpec(
                archetype="sloping_hf_mild",
                description="Mild-moderate sloping HF loss: good LF consonants, elevated S/SH & T/K confusions",
                self_lf=2.75, self_hf=1.35,
                w_voice=1.00, w_manner=0.95, w_place=0.25, w_sibilant=0.95,
                unk_lf=-3.0, unk_hf=-1.6,
                asymmetry={("S", "T"): 0.45, ("Z", "V"): 0.55, ("G", "D"): 0.35},
                dirichlet_conc=45.0,
            ),
        ),
        (
            3, "HF_Steep",
            _ArchetypeSpec(
                archetype="sloping_hf_steep",
                description="Steep ski-slope HF loss: preserved N/B, severe place loss & '???' on 0-dB HF tokens",
                self_lf=2.35, self_hf=0.20,
                w_voice=0.95, w_manner=0.75, w_place=0.05, w_sibilant=0.65,
                unk_lf=-2.5, unk_hf=-0.45,
                asymmetry={("S", "???"): 0.55, ("T", "K"): 0.50, ("Z", "V"): 0.80, ("G", "B"): 0.65},
                dirichlet_conc=40.0,
            ),
        ),
        (
            3, "Broad_Place",
            _ArchetypeSpec(
                archetype="spectral_smearing",
                description="Broad spectral smearing (CI / cochlear dead-region like): manner & voicing preserved, heavy place confusions",
                self_lf=1.65, self_hf=1.15,
                w_voice=1.15, w_manner=1.25, w_place=0.0, w_sibilant=0.75,
                unk_lf=-2.3, unk_hf=-1.7,
                asymmetry={("B", "D"): 0.55, ("D", "B"): 0.45, ("S", "SH"): 0.65, ("SH", "S"): 0.60},
                dirichlet_conc=38.0,
            ),
        ),
        (
            3, "Voicing_Deficit",
            _ArchetypeSpec(
                archetype="temporal_voicing",
                description="Temporal fine-structure / voicing deficit: cognate confusions (D/T, G/K, Z/S) and D/N blurring",
                self_lf=1.80, self_hf=1.55,
                w_voice=0.10, w_manner=0.90, w_place=0.80, w_sibilant=0.85,
                unk_lf=-2.2, unk_hf=-1.8,
                asymmetry={
                    ("D", "T"): 1.15, ("T", "D"): 0.95,
                    ("G", "K"): 1.10, ("K", "G"): 0.95,
                    ("Z", "S"): 1.20, ("S", "Z"): 0.85,
                    ("D", "N"): 0.70, ("B", "???"): 0.55,
                },
                dirichlet_conc=35.0,
            ),
        ),
        (
            3, "High_Noise",
            _ArchetypeSpec(
                archetype="poor_sin_high_guess",
                description="Severe speech-in-noise deficit: low overall ID with frequent '???' and broad guessing",
                self_lf=1.10, self_hf=0.35,
                w_voice=0.55, w_manner=0.55, w_place=0.15, w_sibilant=0.45,
                unk_lf=-1.1, unk_hf=-0.35,
                asymmetry={},
                dirichlet_conc=30.0,
            ),
        ),
    ]

    cohort: list[VirtualListener] = []
    for count, prefix, spec in specs:
        for k in range(1, count + 1):
            lid = f"{prefix}_{k:02d}"
            cohort.append(_build_synthetic_listener(lid, spec, rng))
    return cohort


# --------------------------------------------------------------------------
# Saving & Loading Ground Truth Profiles (CSV / JSON / Empirical CSV / MAT)
# --------------------------------------------------------------------------

RESP_COLUMNS: list[str] = [
    "p_UNK" if c == "???" else f"p_{c}" for c in DISPLAY_CONSONANTS
]
TOK0_COLUMNS: list[str] = [f"tok1_{col[2:]}" for col in RESP_COLUMNS]
TOK1_COLUMNS: list[str] = [f"tok2_{col[2:]}" for col in RESP_COLUMNS]


def save_cohort_csv(listeners: list[VirtualListener], csv_path: str | Path) -> None:
    """Export virtual listeners to a tidy ground-truth CSV (10 rows per listener)."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "listener_id", "archetype", "description", "target_consonant", "snr_db",
        *RESP_COLUMNS, *TOK0_COLUMNS, *TOK1_COLUMNS,
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for listener in listeners:
            for i, stim_c in enumerate(STIMULI_CONSONANTS):
                row: dict[str, str | float] = {
                    "listener_id": listener.listener_id,
                    "archetype": listener.archetype,
                    "description": listener.description,
                    "target_consonant": stim_c,
                    "snr_db": CONSONANT_SNR_DB[stim_c],
                }
                for j, col in enumerate(RESP_COLUMNS):
                    row[col] = f"{listener.gt_matrix[i, j]:.6f}"
                    row[TOK0_COLUMNS[j]] = f"{listener.token_matrices[0, i, j]:.6f}"
                    row[TOK1_COLUMNS[j]] = f"{listener.token_matrices[1, i, j]:.6f}"
                writer.writerow(row)


def save_cohort_json(listeners: list[VirtualListener], json_path: str | Path) -> None:
    """Export virtual listeners to JSON (similar to qCLS simulated_listeners_v2.json)."""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "listener_id": l.listener_id,
            "archetype": l.archetype,
            "description": l.description,
            "stimuli_consonants": STIMULI_CONSONANTS,
            "display_consonants": DISPLAY_CONSONANTS,
            "gt_matrix": np.round(l.gt_matrix, 6).tolist(),
            "token_matrices": np.round(l.token_matrices, 6).tolist(),
        }
        for l in listeners
    ]
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _listener_from_empirical_trial_csv(csv_path: Path, smoothing: float = 0.05) -> VirtualListener:
    """
    Build a VirtualListener from an exported qVCV web-app trial log
    (columns: Trial, Phase, Stimulus, Target, Response).
    """
    counts_tok = np.full((2, N_STIMULI, N_RESPONSES), smoothing, dtype=float)
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            target = row.get("Target", "").strip().upper()
            resp = row.get("Response", "").strip().upper()
            stim_file = row.get("Stimulus", "").strip()
            if target not in STIM_INDEX or resp not in RESP_INDEX:
                continue
            tok = parse_token_index(stim_file) if stim_file else 0
            counts_tok[tok, STIM_INDEX[target], RESP_INDEX[resp]] += 1.0

    tok_mats = np.stack([normalize_rows(counts_tok[0]), normalize_rows(counts_tok[1])], axis=0)
    gt_mat = 0.5 * (tok_mats[0] + tok_mats[1])
    lid = csv_path.stem.replace("qVCV_", "")
    return VirtualListener(
        listener_id=lid,
        gt_matrix=gt_mat,
        archetype="empirical_csv",
        description=f"Loaded from empirical qVCV trial log {csv_path.name}",
        token_matrices=tok_mats,
    )


class GroundTruthProfiles:
    """Collection of VirtualListeners indexed by listener_id (mirrors qCLS GroundTruthProfiles)."""

    def __init__(self, listeners: list[VirtualListener]):
        if not listeners:
            raise ValueError("GroundTruthProfiles requires at least one VirtualListener.")
        self._models: dict[str, VirtualListener] = {l.listener_id: l for l in listeners}

    def __getitem__(self, listener_id: str) -> np.ndarray:
        return self._models[listener_id].full_profile()

    def __len__(self) -> int:
        return len(self._models)

    def keys(self) -> list[str]:
        return list(self._models.keys())

    def items(self) -> Iterator[tuple[str, np.ndarray]]:
        return ((lid, model.full_profile()) for lid, model in self._models.items())

    def get_model(self, listener_id: str) -> VirtualListener:
        return self._models[listener_id]

    def new_virtual_listener(
        self, listener_id: str, rng: np.random.Generator
    ) -> VirtualListener:
        base = self._models[listener_id]
        return VirtualListener(
            listener_id=base.listener_id,
            gt_matrix=base.gt_matrix,
            archetype=base.archetype,
            description=base.description,
            token_matrices=base.token_matrices,
            rng=rng,
        )


def load_ground_truth(source: Optional[str | Path] = None) -> GroundTruthProfiles:
    """
    Load ground-truth virtual listeners from:
      - None or missing default ``virtual_listeners.csv``: auto-generates and saves
        the 18-listener synthetic clinical cohort
      - Profile CSV (``virtual_listeners.csv``)
      - Profile JSON (``virtual_listeners.json``)
      - Empirical web-app trial CSV or directory of ``qVCV_*.csv`` files
      - MATLAB ``.mat`` file containing ``gt_matrices`` / ``confusion_matrices`` / ``pmat``
    """
    default_csv = Path(__file__).resolve().with_name("virtual_listeners.csv")
    if source is None:
        source_path = default_csv
        if not source_path.exists():
            cohort = generate_synthetic_cohort()
            save_cohort_csv(cohort, source_path)
            save_cohort_json(cohort, source_path.with_suffix(".json"))
            return GroundTruthProfiles(cohort)
    else:
        source_path = Path(source)
        if not source_path.exists() and source_path == default_csv:
            cohort = generate_synthetic_cohort()
            save_cohort_csv(cohort, source_path)
            save_cohort_json(cohort, source_path.with_suffix(".json"))
            return GroundTruthProfiles(cohort)

    if source_path.is_dir():
        csv_files = sorted(source_path.glob("*.csv"))
        if not csv_files:
            raise ValueError(f"No CSV files found in directory {source_path}")
        return GroundTruthProfiles([_listener_from_empirical_trial_csv(p) for p in csv_files])

    suffix = source_path.suffix.lower()
    if suffix == ".json":
        with open(source_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        listeners = [
            VirtualListener(
                listener_id=str(rec["listener_id"]),
                gt_matrix=np.array(rec["gt_matrix"], dtype=float),
                archetype=str(rec.get("archetype", "json")),
                description=str(rec.get("description", "")),
                token_matrices=(
                    np.array(rec["token_matrices"], dtype=float)
                    if "token_matrices" in rec
                    else None
                ),
            )
            for rec in records
        ]
        return GroundTruthProfiles(listeners)

    if suffix == ".mat":
        import scipy.io as sio

        mat = sio.loadmat(str(source_path))
        arr = None
        for key in ("gt_matrices", "confusion_matrices", "pmat", "theta"):
            if key in mat:
                arr = np.asarray(mat[key], dtype=float)
                break
        if arr is None:
            raise ValueError(
                f"MATLAB file {source_path} must contain one of: "
                "gt_matrices, confusion_matrices, pmat, theta"
            )
        if arr.ndim == 2 and arr.shape == (N_STIMULI, N_RESPONSES):
            arr = arr[None, :, :]
        elif arr.ndim == 3 and arr.shape[:2] == (N_STIMULI, N_RESPONSES) and arr.shape[2] != N_RESPONSES:
            arr = np.transpose(arr, (2, 0, 1))
        if arr.ndim != 3 or arr.shape[1:] != (N_STIMULI, N_RESPONSES):
            raise ValueError(
                f"Expected .mat array shape (N, 10, 11) or (10, 11, N), got {arr.shape}"
            )
        listeners = [
            VirtualListener(
                listener_id=f"mat_listener_{idx:03d}",
                gt_matrix=arr[idx],
                archetype="mat_import",
            )
            for idx in range(arr.shape[0])
        ]
        return GroundTruthProfiles(listeners)

    if suffix == ".csv":
        with open(source_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            headers = set(reader.fieldnames or [])
            rows = list(reader)

        # Case 1: Empirical single-session web-app export (Trial,Phase,Stimulus,Target,Response)
        if {"Target", "Response"}.issubset(headers):
            return GroundTruthProfiles([_listener_from_empirical_trial_csv(source_path)])

        # Case 2: Ground-truth profile catalog CSV
        required = {"listener_id", "target_consonant", *RESP_COLUMNS}
        missing = required.difference(headers)
        if missing:
            raise ValueError(
                f"{source_path} is missing required profile columns: {sorted(missing)}"
            )

        grouped: dict[str, list[dict[str, str]]] = {}
        for r in rows:
            grouped.setdefault(r["listener_id"], []).append(r)

        listeners = []
        has_tokens = set(TOK0_COLUMNS).issubset(headers) and set(TOK1_COLUMNS).issubset(headers)
        for lid, lrows in grouped.items():
            gt = np.zeros((N_STIMULI, N_RESPONSES), dtype=float)
            tok = np.zeros((2, N_STIMULI, N_RESPONSES), dtype=float) if has_tokens else None
            seen = set()
            archetype = lrows[0].get("archetype", "csv")
            description = lrows[0].get("description", "")
            for r in lrows:
                tc = r["target_consonant"].strip().upper()
                idx = STIM_INDEX[tc]
                seen.add(tc)
                gt[idx] = [float(r[col]) for col in RESP_COLUMNS]
                if tok is not None:
                    tok[0, idx] = [float(r[col]) for col in TOK0_COLUMNS]
                    tok[1, idx] = [float(r[col]) for col in TOK1_COLUMNS]
            if len(seen) != N_STIMULI:
                raise ValueError(
                    f"Listener {lid} in {source_path} has {len(seen)} target consonants; expected 10"
                )
            listeners.append(
                VirtualListener(
                    listener_id=lid,
                    gt_matrix=gt,
                    archetype=archetype,
                    description=description,
                    token_matrices=tok,
                )
            )
        return GroundTruthProfiles(listeners)

    raise ValueError(f"Unsupported ground-truth file extension: {source_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic virtual-listener catalog for qVCV.")
    parser.add_argument("--seed", type=int, default=20260512)
    parser.add_argument(
        "--out_csv", type=Path,
        default=Path(__file__).resolve().with_name("virtual_listeners.csv"),
    )
    parser.add_argument(
        "--out_json", type=Path,
        default=Path(__file__).resolve().with_name("virtual_listeners.json"),
    )
    args = parser.parse_args()

    cohort = generate_synthetic_cohort(seed=args.seed)
    save_cohort_csv(cohort, args.out_csv)
    save_cohort_json(cohort, args.out_json)

    print(f"Generated {len(cohort)} virtual listeners:")
    for listener in cohort:
        pc = 100.0 * float(np.mean(np.diag(listener.gt_matrix[:, :N_STIMULI])))
        unk = 100.0 * float(np.mean(listener.gt_matrix[:, -1]))
        it = relative_information_transmitted(listener.gt_matrix)
        print(
            f"  {listener.listener_id:<18s} [{listener.archetype:<18s}] "
            f"Pc={pc:5.1f}%  Unk={unk:4.1f}%  "
            f"IT(all={it['overall']:4.1f}%, voice={it['voicing']:4.1f}%, "
            f"manner={it['manner']:4.1f}%, place={it['place']:4.1f}%)"
        )
    print(f"\nSaved CSV : {args.out_csv}")
    print(f"Saved JSON: {args.out_json}")


if __name__ == "__main__":
    main()
