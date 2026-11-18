#!/usr/bin/env python3
"""
exp_adaptive_vs_block.py

Head-to-head benchmark comparing qVCV's shipped 5x20 block-randomization
procedure against Bayesian adaptive consonant selection and phonetic-feature
Empirical-Bayes estimation (mirroring qCLS/tracking_sims/exp_gp_vs_bap.py).

Evaluates 6 arms across trial budgets N = [20, 40, 60, 80, 100, 140]:
  1. block[empirical]            : Current shipped qVCV web app
  2. random[empirical]           : Uniform random stimulus control
  3. block[feature_eb]           : Block selection + coupled phonetic-feature EB estimator
  4. adaptive_var[dirichlet]     : Adaptive Dirichlet MSE-reduction + uncoupled posterior
  5. adaptive_eig[dirichlet]     : Adaptive Expected Information Gain (BALD) + uncoupled posterior
  6. adaptive_feature[feature_eb]: Adaptive coupled phonetic-feature EB tracker + estimator
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from qvcv_simulator import STIMULI_CONSONANTS, load_ground_truth
from qvcv_trackers import QVCVTrackerAdapter
from qvcv_tracking_sim import (
    convergence_sweep,
    run_accuracy_reliability,
    write_dict_rows_csv,
)

BENCHMARK_ARMS: list[tuple[str, str, str]] = [
    ("block_empirical",      "block",            "empirical"),
    ("random_empirical",     "random",           "empirical"),
    ("adaptive_empirical",   "adaptive_var",     "empirical"),
    ("block_dirichlet",      "block",            "dirichlet"),
    ("adaptive_var",         "adaptive_var",     "dirichlet"),
    ("adaptive_eig",         "adaptive_eig",     "dirichlet"),
    ("block_feature_eb",     "block",            "feature_eb"),
    ("adaptive_feature",     "adaptive_feature", "feature_eb"),
]


def equivalent_trials_for_target(
    trial_counts: list[int],
    errors: list[float],
    target_error: float,
) -> float:
    """
    Interpolate the number of trials needed on a monotone-decreasing error curve
    to reach ``target_error`` (e.g., the 100-trial baseline error).
    """
    xs = np.asarray(trial_counts, dtype=float)
    ys = np.asarray(errors, dtype=float)
    if ys[0] <= target_error:
        return float(xs[0])
    if ys[-1] > target_error:
        return float("nan")
    for k in range(len(xs) - 1):
        if ys[k] >= target_error >= ys[k + 1]:
            if abs(ys[k] - ys[k + 1]) < 1e-12:
                return float(xs[k])
            frac = (ys[k] - target_error) / (ys[k] - ys[k + 1])
            return float(xs[k] + frac * (xs[k + 1] - xs[k]))
    return float("nan")


def save_benchmark_figures(
    all_sweep_rows: list[dict[str, Any]],
    alloc_rows: list[dict[str, Any]],
    out_dir: Path,
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/qvcv_mplconfig")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    arm_styles = {
        "block_empirical":    ("Current Web App (Block + Raw)",  "#222222", "--", "o"),
        "random_empirical":   ("Random + Raw",                   "#9ca3af", ":",  "s"),
        "adaptive_empirical": ("Adaptive Var + Raw",             "#6b7280", "-",  "P"),
        "block_dirichlet":    ("Block + Dirichlet",              "#d97706", "--", "^"),
        "adaptive_var":       ("Adaptive Var + Dirichlet",       "#f59e0b", "-",  "D"),
        "adaptive_eig":       ("Adaptive EIG / BALD",            "#10b981", "-",  "v"),
        "block_feature_eb":   ("Block + Feature-EB",             "#2563eb", "-.", "^"),
        "adaptive_feature":   ("Adaptive Feature-EB (Coupled)",  "#dc2626", "-",  "o"),
    }

    # Figure 1: Convergence across trial counts (4 key metrics)
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2))
    metrics = [
        (axes[0, 0], "mean_tvd_pct",         "Mean Row TVD (%) [Lower = More Accurate]"),
        (axes[0, 1], "mean_max_row_tvd_pct", "Worst-Consonant Row TVD (%) [Hardest Row]"),
        (axes[1, 0], "mean_diag_mae_pct",    "Per-Consonant ID Accuracy MAE (%)"),
        (axes[1, 1], "retest_tvd_pct",       "Test-Retest Disagreement TVD (%) [Reliability]"),
    ]

    for arm_key, _, _ in BENCHMARK_ARMS:
        sub = [r for r in all_sweep_rows if r["mode"] == arm_key]
        if not sub:
            continue
        label, color, ls, marker = arm_styles[arm_key]
        xs = [r["n_trials"] for r in sub]
        for ax, mkey, _ in metrics:
            ys = [r[mkey] for r in sub]
            ax.plot(xs, ys, linestyle=ls, marker=marker, color=color, linewidth=2.0, markersize=5.5, label=label)

    for ax, _, title in metrics:
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Number of Trials (N)")
        ax.set_ylabel("Error / Disagreement (%)")
        ax.grid(True, alpha=0.28)
    axes[0, 0].legend(fontsize=8.2, loc="upper right")
    fig.suptitle("qVCV Tracking Simulation: Convergence & Test-Retest Reliability vs. Trial Budget", fontsize=12.5, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "convergence_comparison.png", dpi=170)
    plt.close(fig)

    # Figure 2: Trial allocation across the 10 consonants by clinical archetype (at N=100)
    archetypes = sorted({r["archetype"] for r in alloc_rows})
    fig2, ax2 = plt.subplots(figsize=(10.8, 5.2))
    x = np.arange(len(STIMULI_CONSONANTS))
    width = 0.8 / max(1, len(archetypes))

    palette = ["#1d4ed8", "#0ea5e9", "#dc2626", "#9333ea", "#ea580c", "#16a34a"]
    for idx, arch in enumerate(archetypes):
        means = []
        for sc in STIMULI_CONSONANTS:
            vals = [r["mean_trials"] for r in alloc_rows if r["archetype"] == arch and r["target_consonant"] == sc]
            means.append(float(np.mean(vals)) if vals else 0.0)
        offset = (idx - 0.5 * (len(archetypes) - 1)) * width
        ax2.bar(x + offset, means, width=width * 0.92, label=arch, color=palette[idx % len(palette)], alpha=0.88)

    ax2.axhline(10.0, color="#222", linestyle="--", linewidth=1.4, label="Block Uniform Baseline (10 trials/consonant)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(STIMULI_CONSONANTS, fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Trials Allocated (out of N=100)")
    ax2.set_xlabel("Target Consonant")
    ax2.set_title("Adaptive Trial Allocation by Clinical Archetype (adaptive_feature, N=100)", fontsize=11.5, fontweight="bold")
    ax2.legend(fontsize=8.5, ncol=2)
    ax2.grid(True, axis="y", alpha=0.25)
    fig2.tight_layout()
    fig2.savefig(out_dir / "adaptive_allocation_by_archetype.png", dpi=170)
    plt.close(fig2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark adaptive qVCV tracking vs 5x20 block baseline.")
    parser.add_argument("--ground_truth", type=Path, default=None)
    parser.add_argument("--trial_counts", type=int, nargs="+", default=[20, 40, 60, 80, 100, 140])
    parser.add_argument("--n_reps", type=int, default=30)
    parser.add_argument("--base_seed", type=int, default=42)
    parser.add_argument(
        "--out_dir", type=Path,
        default=Path(__file__).resolve().with_name("benchmark_results"),
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    listeners = load_ground_truth(args.ground_truth)
    print(f"Loaded {len(listeners)} virtual listeners | reps={args.n_reps} | sweeps={args.trial_counts}\n")

    all_sweep_rows: list[dict[str, Any]] = []
    for arm_name, sel_mode, est_mode in BENCHMARK_ARMS:
        t_arm = time.perf_counter()
        rows = convergence_sweep(
            listeners,
            adapter_factory=lambda sm=sel_mode, em=est_mode: QVCVTrackerAdapter(mode=sm, estimator=em),
            trial_counts=args.trial_counts,
            n_reps=args.n_reps,
            base_seed=args.base_seed,
            mode_label=arm_name,
        )
        all_sweep_rows.extend(rows)
        dt = time.perf_counter() - t_arm
        r100 = next((r for r in rows if r["n_trials"] == 100), rows[-1])
        print(
            f"  [{arm_name:<18s}] ({dt:4.1f}s) @ N={r100['n_trials']:3d}: "
            f"TVD={r100['mean_tvd_pct']:5.2f}% | MaxRowTVD={r100['mean_max_row_tvd_pct']:5.2f}% | "
            f"DiagMAE={r100['mean_diag_mae_pct']:5.2f}% | RetestTVD={r100['retest_tvd_pct']:5.2f}%"
        )

    write_dict_rows_csv(all_sweep_rows, out_dir / "benchmark_convergence.csv")

    # Detailed trial-allocation run for adaptive_feature at N=100
    _, _, _, alloc_rows = run_accuracy_reliability(
        listeners,
        adapter_factory=lambda: QVCVTrackerAdapter(mode="adaptive_feature", estimator="feature_eb"),
        n_trials=100,
        n_reps=args.n_reps,
        base_seed=args.base_seed,
        mode_label="adaptive_feature",
    )
    write_dict_rows_csv(alloc_rows, out_dir / "adaptive_allocation_N100.csv")

    # Compute equivalent trials needed by each arm to match N=100 block_empirical
    base_rows = [r for r in all_sweep_rows if r["mode"] == "block_empirical"]
    ref_nt = 100 if any(r["n_trials"] == 100 for r in base_rows) else base_rows[-1]["n_trials"]
    ref_row = next(r for r in base_rows if r["n_trials"] == ref_nt)

    efficiency_summary: dict[str, Any] = {
        "reference_arm": "block_empirical",
        "reference_n_trials": ref_nt,
        "reference_mean_tvd_pct": ref_row["mean_tvd_pct"],
        "reference_max_row_tvd_pct": ref_row["mean_max_row_tvd_pct"],
        "reference_diag_mae_pct": ref_row["mean_diag_mae_pct"],
        "reference_retest_tvd_pct": ref_row["retest_tvd_pct"],
        "arms": {},
    }

    print(f"\n=== EQUIVALENT TRIAL BUDGET TO MATCH {ref_nt}-TRIAL SHIPPED WEB APP (block_empirical) ===")
    print(f"{'Arm':<20s} | {'EqTrials (MeanTVD)':>18s} | {'EqTrials (MaxRowTVD)':>20s} | {'EqTrials (DiagMAE)':>18s}")
    print("-" * 86)
    for arm_name, _, _ in BENCHMARK_ARMS:
        sub = sorted([r for r in all_sweep_rows if r["mode"] == arm_name], key=lambda x: x["n_trials"])
        nts = [r["n_trials"] for r in sub]
        eq_tvd = equivalent_trials_for_target(nts, [r["mean_tvd_pct"] for r in sub], ref_row["mean_tvd_pct"])
        eq_max = equivalent_trials_for_target(nts, [r["mean_max_row_tvd_pct"] for r in sub], ref_row["mean_max_row_tvd_pct"])
        eq_diag = equivalent_trials_for_target(nts, [r["mean_diag_mae_pct"] for r in sub], ref_row["mean_diag_mae_pct"])
        efficiency_summary["arms"][arm_name] = {
            "eq_trials_mean_tvd": eq_tvd,
            "eq_trials_max_row_tvd": eq_max,
            "eq_trials_diag_mae": eq_diag,
        }
        s_tvd = f"{eq_tvd:5.1f} ({100.0 * (1.0 - eq_tvd / ref_nt):+4.0f}%)" if not np.isnan(eq_tvd) else f">{max(nts)}"
        s_max = f"{eq_max:5.1f} ({100.0 * (1.0 - eq_max / ref_nt):+4.0f}%)" if not np.isnan(eq_max) else f">{max(nts)}"
        s_diag = f"{eq_diag:5.1f} ({100.0 * (1.0 - eq_diag / ref_nt):+4.0f}%)" if not np.isnan(eq_diag) else f">{max(nts)}"
        print(f"{arm_name:<20s} | {s_tvd:>18s} | {s_max:>20s} | {s_diag:>18s}")

    with open(out_dir / "benchmark_summary.json", "w", encoding="utf-8") as f:
        json.dump(efficiency_summary, f, indent=2)

    save_benchmark_figures(all_sweep_rows, alloc_rows, out_dir)
    elapsed = time.perf_counter() - t0
    print(f"\nCompleted full benchmark in {elapsed:.1f}s. Results saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
