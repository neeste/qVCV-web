"""
qvcv_tracking_sim.py

Standalone numerical simulation harness to assess the ACCURACY, TEST-RETEST
RELIABILITY, BIAS, and CONVERGENCE of qVCV tracking and estimation procedures
(mirroring qCLS/sim/qcls_wasm_accuracy_sim.py).

Metrics
-------
1. Accuracy    : Per-listener error between estimated (10x11) confusion matrix
                 and ground truth:
                   - cell_mae_pct   : Mean absolute error across all 110 cells (percentage points)
                   - cell_rmse_pct  : Root-mean-square cell error (percentage points)
                   - mean_tvd_pct   : Mean row Total Variation Distance (0..100%)
                   - max_row_tvd_pct: Worst-consonant row TVD (pins down hardest confusions)
                   - diag_mae_pct   : Per-consonant identification rate MAE (percentage points)
                   - pc_abs_err_pct : Overall percent-correct absolute error (percentage points)
                   - js_div_bits    : Mean row Jensen-Shannon divergence (bits)
                   - it_*_err_pct   : Error in Miller-Nicely Relative Information Transmitted
                                      (overall, voicing, manner, place)
2. Reliability : Pairwise test-retest MAD across independent repeated sessions
                 on the same virtual listener (cell_mad_pct, diag_mad_pct, mean_tvd_pct).
3. Bias        : Per-(listener, target consonant, response) signed error
                 (estimate - truth), written to signed_bias_by_cell.csv.
4. Allocation  : Mean trial allocation per consonant by listener/archetype,
                 showing how adaptive tracking shifts trials onto confused consonants.
5. Convergence : Accuracy vs. trial count (--sweep_trials 20 40 60 80 100).

Usage
-----
    # Evaluate current web-app baseline (block randomization + empirical counts):
    python3 sim/qvcv_tracking_sim.py --mode block --n_trials 100 --n_reps 25

    # Evaluate coupled adaptive phonetic-feature tracker at 60 trials:
    python3 sim/qvcv_tracking_sim.py --mode adaptive_feature --n_trials 60 --n_reps 25

    # Convergence sweep across trial counts:
    python3 sim/qvcv_tracking_sim.py --mode adaptive_feature \\
        --sweep_trials 20 40 60 80 100 --n_reps 25
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zlib
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

try:
    from .qvcv_simulator import (
        DISPLAY_CONSONANTS,
        N_STIMULI,
        STIMULI_CONSONANTS,
        GroundTruthProfiles,
        load_ground_truth,
        normalize_rows,
        relative_information_transmitted,
    )
    from .qvcv_trackers import (
        ESTIMATOR_MODES,
        SELECTION_MODES,
        QVCVTrackerAdapter,
    )
except ImportError:
    from qvcv_simulator import (
        DISPLAY_CONSONANTS,
        N_STIMULI,
        STIMULI_CONSONANTS,
        GroundTruthProfiles,
        load_ground_truth,
        normalize_rows,
        relative_information_transmitted,
    )
    from qvcv_trackers import (
        ESTIMATOR_MODES,
        SELECTION_MODES,
        QVCVTrackerAdapter,
    )


# --------------------------------------------------------------------------
# Deterministic Seeding & CSV Utilities
# --------------------------------------------------------------------------

def make_rep_seed(base_seed: int, listener_id: str, rep: int) -> int:
    """Platform-independent deterministic seed for (base_seed, listener_id, rep)."""
    crc = zlib.crc32(f"{listener_id}:{rep}".encode("utf-8")) & 0xFFFFFFFF
    return int((base_seed * 1_000_003 + crc) & 0x7FFFFFFF)


def write_dict_rows_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def jensen_shannon_bits(p: np.ndarray, q: np.ndarray) -> float:
    """Mean row-wise Jensen-Shannon divergence in bits (bounded in [0, 1])."""
    p = normalize_rows(p)
    q = normalize_rows(q)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        mask = a > 1e-15
        ratio = np.where(mask, a / np.maximum(b, 1e-15), 1.0)
        return np.sum(np.where(mask, a * np.log2(ratio), 0.0), axis=1)

    js_rows = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    return float(np.mean(np.maximum(0.0, js_rows)))


def compute_accuracy_metrics(estimate: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Compute comprehensive confusion-matrix recovery metrics (in % points and bits)."""
    est = normalize_rows(estimate)
    gt = normalize_rows(truth)
    abs_diff = np.abs(est - gt)

    row_tvd = 0.5 * np.sum(abs_diff, axis=1)  # (10,) in [0, 1]
    est_diag = np.diag(est[:, :N_STIMULI])
    gt_diag = np.diag(gt[:, :N_STIMULI])

    it_est = relative_information_transmitted(est)
    it_gt = relative_information_transmitted(gt)

    return {
        "cell_mae_pct": 100.0 * float(np.mean(abs_diff)),
        "cell_rmse_pct": 100.0 * float(np.sqrt(np.mean((est - gt) ** 2))),
        "mean_tvd_pct": 100.0 * float(np.mean(row_tvd)),
        "max_row_tvd_pct": 100.0 * float(np.max(row_tvd)),
        "diag_mae_pct": 100.0 * float(np.mean(np.abs(est_diag - gt_diag))),
        "pc_abs_err_pct": 100.0 * float(abs(np.mean(est_diag) - np.mean(gt_diag))),
        "js_div_bits": jensen_shannon_bits(est, gt),
        "it_overall_err_pct": abs(it_est["overall"] - it_gt["overall"]),
        "it_voicing_err_pct": abs(it_est["voicing"] - it_gt["voicing"]),
        "it_manner_err_pct": abs(it_est["manner"] - it_gt["manner"]),
        "it_place_err_pct": abs(it_est["place"] - it_gt["place"]),
    }


def compute_pair_reliability(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Pairwise test-retest agreement between two independent session estimates."""
    pa = normalize_rows(a)
    pb = normalize_rows(b)
    abs_diff = np.abs(pa - pb)
    diag_a = np.diag(pa[:, :N_STIMULI])
    diag_b = np.diag(pb[:, :N_STIMULI])
    return {
        "cell_mad_pct": 100.0 * float(np.mean(abs_diff)),
        "mean_tvd_pct": 100.0 * float(np.mean(0.5 * np.sum(abs_diff, axis=1))),
        "diag_mad_pct": 100.0 * float(np.mean(np.abs(diag_a - diag_b))),
        "pc_diff_pct": 100.0 * float(abs(np.mean(diag_a) - np.mean(diag_b))),
    }


# --------------------------------------------------------------------------
# Visualization helper
# --------------------------------------------------------------------------

def plot_confusion_comparison(
    listener_id: str,
    repetition: int,
    truth: np.ndarray,
    estimate: np.ndarray,
    counts: Optional[np.ndarray] = None,
    save_path: Optional[Path] = None,
) -> None:
    """Plot side-by-side ground-truth vs estimated 10x11 confusion heatmaps."""
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qvcv_mplconfig")
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 3 if counts is not None else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6))

    panels = [
        (axes[0], 100.0 * truth, "Ground Truth (%)", "Oranges", 100.0),
        (axes[1], 100.0 * estimate, f"Estimate (rep {repetition}) (%)", "Oranges", 100.0),
    ]
    if counts is not None:
        row_n = counts.sum(axis=1)
        panels.append((axes[2], counts, f"Raw Trial Counts (N={int(row_n.sum())})", "Blues", float(np.max(counts) + 1e-6)))

    for ax, mat, title, cmap, vmax in panels:
        im = ax.imshow(mat, cmap=cmap, vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(DISPLAY_CONSONANTS)))
        ax.set_xticklabels(DISPLAY_CONSONANTS, fontsize=8)
        ax.set_yticks(range(len(STIMULI_CONSONANTS)))
        ax.set_yticklabels(STIMULI_CONSONANTS, fontsize=8)
        ax.set_xlabel("Response")
        ax.set_ylabel("Target Stimulus")
        ax.set_title(title, fontsize=10)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if val >= 1.0:
                    txt = f"{int(round(val))}"
                    color = "white" if val > 0.58 * vmax else "#222"
                    ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, color=color)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Listener {listener_id}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=160)
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------
# Core Simulation Runners
# --------------------------------------------------------------------------

def run_accuracy_reliability(
    listeners: GroundTruthProfiles,
    adapter_factory: Callable[[], QVCVTrackerAdapter],
    n_trials: int,
    n_reps: int,
    base_seed: int = 0,
    mode_label: str = "block",
    plot_dir: Optional[Path] = None,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """
    For each virtual listener, run ``n_reps`` independent simulated sessions.
    Returns (accuracy_rows, reliability_rows, bias_rows, allocation_rows).
    """
    acc_rows: list[dict[str, Any]] = []
    rel_rows: list[dict[str, Any]] = []
    bias_rows: list[dict[str, Any]] = []
    alloc_rows: list[dict[str, Any]] = []

    for lid, gt in listeners.items():
        base_model = listeners.get_model(lid)
        adapter = adapter_factory()
        fits: list[np.ndarray] = []
        counts_list: list[np.ndarray] = []

        for r in range(n_reps):
            seed = make_rep_seed(base_seed, lid, r)
            vlistener = listeners.new_virtual_listener(lid, np.random.default_rng(seed ^ 0x5A5A5A5A))
            res = adapter.run_session(vlistener, n_trials=n_trials, seed=seed)

            metrics = compute_accuracy_metrics(res.fitted_matrix, gt)
            acc_rows.append({
                "mode": mode_label,
                "listener_id": lid,
                "archetype": base_model.archetype,
                "rep": r,
                "n_trials": n_trials,
                **metrics,
            })
            fits.append(res.fitted_matrix)
            counts_list.append(res.counts)

            if plot_dir is not None and r == 0:
                plot_confusion_comparison(
                    listener_id=lid,
                    repetition=1,
                    truth=gt,
                    estimate=res.fitted_matrix,
                    counts=res.counts,
                    save_path=plot_dir / f"heatmap_{mode_label}_{lid}.png",
                )

        # Pairwise test-retest reliability across independent reps
        for i in range(len(fits)):
            for j in range(i + 1, len(fits)):
                rel_metrics = compute_pair_reliability(fits[i], fits[j])
                rel_rows.append({
                    "mode": mode_label,
                    "listener_id": lid,
                    "archetype": base_model.archetype,
                    "rep_i": i,
                    "rep_j": j,
                    **rel_metrics,
                })

        # Signed bias per cell averaged across reps
        stacked_fits = np.stack(fits, axis=0)              # (n_reps, 10, 11)
        mean_signed_pct = 100.0 * np.mean(stacked_fits - gt[None, :, :], axis=0)
        for si, sc in enumerate(STIMULI_CONSONANTS):
            for rj, rc in enumerate(DISPLAY_CONSONANTS):
                bias_rows.append({
                    "mode": mode_label,
                    "listener_id": lid,
                    "target_consonant": sc,
                    "response_category": rc,
                    "truth_pct": 100.0 * float(gt[si, rj]),
                    "signed_bias_pct": float(mean_signed_pct[si, rj]),
                })

        # Stimulus trial allocation per target consonant averaged across reps
        stacked_counts = np.stack(counts_list, axis=0)     # (n_reps, 10, 11)
        mean_row_trials = np.mean(stacked_counts.sum(axis=2), axis=0)  # (10,)
        sd_row_trials = np.std(stacked_counts.sum(axis=2), axis=0)
        true_row_entropy = -np.sum(gt * np.log2(np.maximum(gt, 1e-15)), axis=1)
        for si, sc in enumerate(STIMULI_CONSONANTS):
            alloc_rows.append({
                "mode": mode_label,
                "listener_id": lid,
                "archetype": base_model.archetype,
                "target_consonant": sc,
                "true_id_pct": 100.0 * float(gt[si, si]),
                "true_entropy_bits": float(true_row_entropy[si]),
                "mean_trials": float(mean_row_trials[si]),
                "sd_trials": float(sd_row_trials[si]),
            })

    return acc_rows, rel_rows, bias_rows, alloc_rows


def convergence_sweep(
    listeners: GroundTruthProfiles,
    adapter_factory: Callable[[], QVCVTrackerAdapter],
    trial_counts: list[int],
    n_reps: int,
    base_seed: int = 0,
    mode_label: str = "block",
) -> list[dict[str, Any]]:
    """Evaluate accuracy & test-retest reliability across cumulative trial counts."""
    sorted_points = sorted(set(trial_counts))
    # Store per-(nt) lists of metrics and per-(listener, nt) list of fits for reliability
    acc_by_nt: dict[int, list[dict[str, float]]] = {nt: [] for nt in sorted_points}
    fits_by_lid_nt: dict[tuple[str, int], list[np.ndarray]] = {
        (lid, nt): [] for lid, _ in listeners.items() for nt in sorted_points
    }

    for lid, gt in listeners.items():
        adapter = adapter_factory()
        for r in range(n_reps):
            seed = make_rep_seed(base_seed, lid, r)
            vlistener = listeners.new_virtual_listener(lid, np.random.default_rng(seed ^ 0x5A5A5A5A))
            fits_map = adapter.run_session_at_points(vlistener, sorted_points, seed=seed)
            for nt in sorted_points:
                fit = fits_map[nt]
                acc_by_nt[nt].append(compute_accuracy_metrics(fit, gt))
                fits_by_lid_nt[(lid, nt)].append(fit)

    summary_rows: list[dict[str, Any]] = []
    for nt in sorted_points:
        records = acc_by_nt[nt]
        rel_tvds: list[float] = []
        rel_cell_mads: list[float] = []
        for lid in listeners.keys():
            lfits = fits_by_lid_nt[(lid, nt)]
            for i in range(len(lfits)):
                for j in range(i + 1, len(lfits)):
                    pair_rel = compute_pair_reliability(lfits[i], lfits[j])
                    rel_tvds.append(pair_rel["mean_tvd_pct"])
                    rel_cell_mads.append(pair_rel["cell_mad_pct"])

        def _stat(key: str) -> tuple[float, float, float]:
            arr = np.array([rec[key] for rec in records], dtype=float)
            return float(np.mean(arr)), float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0), float(np.median(arr))

        mae_m, mae_sd, mae_med = _stat("cell_mae_pct")
        rmse_m, _, _ = _stat("cell_rmse_pct")
        tvd_m, tvd_sd, tvd_med = _stat("mean_tvd_pct")
        max_tvd_m, _, _ = _stat("max_row_tvd_pct")
        diag_m, _, _ = _stat("diag_mae_pct")
        js_m, _, _ = _stat("js_div_bits")
        it_all_m, _, _ = _stat("it_overall_err_pct")

        summary_rows.append({
            "mode": mode_label,
            "n_trials": nt,
            "mean_cell_mae_pct": mae_m,
            "sd_cell_mae_pct": mae_sd,
            "median_cell_mae_pct": mae_med,
            "mean_cell_rmse_pct": rmse_m,
            "mean_tvd_pct": tvd_m,
            "sd_tvd_pct": tvd_sd,
            "median_tvd_pct": tvd_med,
            "mean_max_row_tvd_pct": max_tvd_m,
            "mean_diag_mae_pct": diag_m,
            "mean_js_div_bits": js_m,
            "mean_it_overall_err_pct": it_all_m,
            "retest_cell_mad_pct": float(np.mean(rel_cell_mads)) if rel_cell_mads else 0.0,
            "retest_tvd_pct": float(np.mean(rel_tvds)) if rel_tvds else 0.0,
        })

    return summary_rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        description="Assess qVCV tracking & estimation accuracy, reliability, and convergence."
    )
    p.add_argument(
        "--ground_truth", type=Path, default=None,
        help="Path to virtual-listener CSV, JSON, MATLAB .mat, or empirical qVCV CSV/folder "
             "(defaults to sim/virtual_listeners.csv, auto-generated if absent).",
    )
    p.add_argument(
        "--mode", choices=SELECTION_MODES, default="adaptive_feature",
        help="Stimulus selection policy.",
    )
    p.add_argument(
        "--estimator", choices=ESTIMATOR_MODES, default="auto",
        help="Confusion-matrix estimator (default 'auto' uses each mode's standard estimator).",
    )
    p.add_argument(
        "--prior_conc", type=float, default=1.5,
        help="Prior pseudo-count concentration per row for uncoupled Dirichlet mode.",
    )
    p.add_argument(
        "--allow_consecutive_repeat", action="store_true",
        help="Allow the same target consonant to be presented on two consecutive trials.",
    )
    p.add_argument(
        "--n_trials", type=int, default=100,
        help="Trials per session (ignored if --sweep_trials is given).",
    )
    p.add_argument(
        "--n_reps", type=int, default=25,
        help="Independent repeated sessions per virtual listener.",
    )
    p.add_argument(
        "--sweep_trials", type=int, nargs="+", default=None,
        help="If given, run a cumulative convergence sweep over these trial counts.",
    )
    p.add_argument(
        "--save_plots", action="store_true",
        help="Save per-listener ground-truth vs estimated confusion-matrix PNG heatmaps.",
    )
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument(
        "--out_dir", type=Path,
        default=Path(__file__).resolve().with_name("sim_results"),
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    listeners = load_ground_truth(args.ground_truth)
    resolved_est = (
        ("empirical" if args.mode in ("block", "random") else ("feature_eb" if args.mode == "adaptive_feature" else "dirichlet"))
        if args.estimator == "auto"
        else args.estimator
    )
    mode_label = f"{args.mode}[{resolved_est}]"
    print(f"Loaded {len(listeners)} virtual listener(s) | mode={mode_label}")

    def adapter_factory() -> QVCVTrackerAdapter:
        return QVCVTrackerAdapter(
            mode=args.mode,
            estimator=args.estimator,
            prior_concentration=args.prior_conc,
            avoid_consecutive_repeat=not args.allow_consecutive_repeat,
        )

    if args.sweep_trials:
        print(f"Running convergence sweep over trial counts: {args.sweep_trials} ({args.n_reps} reps/listener)...")
        sweep_rows = convergence_sweep(
            listeners,
            adapter_factory=adapter_factory,
            trial_counts=args.sweep_trials,
            n_reps=args.n_reps,
            base_seed=args.base_seed,
            mode_label=mode_label,
        )
        sweep_csv = out_dir / "convergence_sweep.csv"
        write_dict_rows_csv(sweep_rows, sweep_csv)

        print(
            f"\n{'Trials':>6s} | {'Cell MAE%':>9s} | {'Mean TVD%':>9s} | "
            f"{'MaxRowTVD%':>10s} | {'Diag MAE%':>9s} | {'IT Err%':>7s} | {'RetestTVD%':>10s}"
        )
        print("-" * 76)
        for r in sweep_rows:
            print(
                f"{r['n_trials']:6d} | {r['mean_cell_mae_pct']:9.2f} | {r['mean_tvd_pct']:9.2f} | "
                f"{r['mean_max_row_tvd_pct']:10.2f} | {r['mean_diag_mae_pct']:9.2f} | "
                f"{r['mean_it_overall_err_pct']:7.2f} | {r['retest_tvd_pct']:10.2f}"
            )
        print(f"\nSaved: {sweep_csv}")
        return

    print(f"Running {args.n_reps} reps x {len(listeners)} listeners ({args.n_trials} trials/session)...")
    plot_dir = (out_dir / "plots") if args.save_plots else None
    acc_rows, rel_rows, bias_rows, alloc_rows = run_accuracy_reliability(
        listeners,
        adapter_factory=adapter_factory,
        n_trials=args.n_trials,
        n_reps=args.n_reps,
        base_seed=args.base_seed,
        mode_label=mode_label,
        plot_dir=plot_dir,
    )

    write_dict_rows_csv(acc_rows, out_dir / "accuracy_per_rep.csv")
    write_dict_rows_csv(rel_rows, out_dir / "reliability_per_pair.csv")
    write_dict_rows_csv(bias_rows, out_dir / "signed_bias_by_cell.csv")
    write_dict_rows_csv(alloc_rows, out_dir / "trial_allocation.csv")

    def _mean(rows: list[dict[str, Any]], key: str) -> float:
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    def _sd(rows: list[dict[str, Any]], key: str) -> float:
        vals = [r[key] for r in rows]
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

    summary = {
        "mode": args.mode,
        "estimator": resolved_est,
        "n_listeners": len(listeners),
        "n_trials": args.n_trials,
        "n_reps": args.n_reps,
        "accuracy_mean_cell_mae_pct": _mean(acc_rows, "cell_mae_pct"),
        "accuracy_sd_cell_mae_pct": _sd(acc_rows, "cell_mae_pct"),
        "accuracy_mean_tvd_pct": _mean(acc_rows, "mean_tvd_pct"),
        "accuracy_mean_max_row_tvd_pct": _mean(acc_rows, "max_row_tvd_pct"),
        "accuracy_mean_diag_mae_pct": _mean(acc_rows, "diag_mae_pct"),
        "accuracy_mean_pc_abs_err_pct": _mean(acc_rows, "pc_abs_err_pct"),
        "accuracy_mean_js_div_bits": _mean(acc_rows, "js_div_bits"),
        "accuracy_mean_it_overall_err_pct": _mean(acc_rows, "it_overall_err_pct"),
        "accuracy_mean_it_voicing_err_pct": _mean(acc_rows, "it_voicing_err_pct"),
        "accuracy_mean_it_manner_err_pct": _mean(acc_rows, "it_manner_err_pct"),
        "accuracy_mean_it_place_err_pct": _mean(acc_rows, "it_place_err_pct"),
        "reliability_mean_cell_mad_pct": _mean(rel_rows, "cell_mad_pct"),
        "reliability_mean_tvd_pct": _mean(rel_rows, "mean_tvd_pct"),
        "reliability_mean_diag_mad_pct": _mean(rel_rows, "diag_mad_pct"),
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) else f"  {k}: {v}")

    # Per-listener mean TVD (worst first, mirroring qCLS)
    by_lid: dict[str, list[float]] = {}
    for r in acc_rows:
        by_lid.setdefault(r["listener_id"], []).append(r["mean_tvd_pct"])
    ranked = sorted(((lid, float(np.mean(vals))) for lid, vals in by_lid.items()), key=lambda x: x[1], reverse=True)
    print("\n=== Per-listener mean row TVD % (worst first) ===")
    for lid, val in ranked:
        print(f"  {lid:<18s}: {val:6.2f}%")

    print(f"\nSaved outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
