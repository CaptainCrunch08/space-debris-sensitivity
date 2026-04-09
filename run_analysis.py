"""
Main entry point for the space debris sensitivity analysis.

Usage
-----
    python run_analysis.py                  # full analysis (may take ~10–30 min)
    python run_analysis.py --quick          # fast mode: short horizon, small samples
    python run_analysis.py --sweep-only     # only run the parameter sweep
    python run_analysis.py --sobol-only     # only run the Sobol analysis

All figures are saved to results/.
All data are saved to results/ as CSV/NPZ.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

# Ensure project root is on path when running from any directory
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.cascade_model import CascadeModel
from models.multishell_model import MultiShellModel
from analysis.sensitivity_analysis import SolarFluxSensitivity
from analysis.plotting import (
    plot_baseline_trajectories,
    plot_sweep,
    plot_sweep_absolute,
    plot_sobol_bars,
    plot_output_distributions,
)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Space debris sensitivity analysis")
    p.add_argument("--quick", action="store_true",
                   help="Fast mode: 50-yr horizon, 10-pt sweep, N=64 Sobol")
    p.add_argument("--sweep-only", action="store_true")
    p.add_argument("--sobol-only", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.quick:
        years, sweep_n, sobol_n, n_ens = 50, 11, 64, 10
        print("Running in QUICK mode (reduced fidelity for testing).")
    else:
        years, sweep_n, sobol_n, n_ens = 200, 33, 256, 30
        print("Running in FULL mode.")

    t0 = time.time()

    # ------------------------------------------------------------------
    # 0. Baseline trajectories
    # ------------------------------------------------------------------
    print("\n[1/4] Generating baseline trajectories (F10.7=150, PMD=90%)…")
    c_base = CascadeModel()
    c_result = c_base.run(years=years)

    ms_base = MultiShellModel(seed=0)
    ms_ens = ms_base.run_ensemble(years=years, n_runs=n_ens)
    ms_result = {
        "t": ms_ens["t"],
        "N_total": ms_ens["N_total_mean"],
        "R_col_mean": ms_ens["R_col_mean"],
    }

    fig_base = plot_baseline_trajectories(c_result, ms_result)
    print(f"  Cascade  final N = {c_result['N'][-1]:,.0f}")
    print(f"  MultiSh  final N = {ms_ens['N_total_mean'][-1]:,.0f}")

    # ------------------------------------------------------------------
    # 1. Parameter sweep
    # ------------------------------------------------------------------
    if not args.sobol_only:
        print("\n[2/4] Running F10.7 parameter sweep…")
        sa = SolarFluxSensitivity(
            years=years,
            n_ensemble=n_ens,
            sweep_n_points=sweep_n,
            sobol_n_base=sobol_n,
        )
        sweep_df = sa.run_sweep()
        sweep_df.to_csv(RESULTS_DIR / "sweep_results.csv", index=False)
        print(f"  Sweep saved -> results/sweep_results.csv")

        plot_sweep(sweep_df)
        plot_sweep_absolute(sweep_df)

        # Sensitivity ratio: how much does each model's output change
        # going from solar min (70) to solar max (230)?
        min_row = sweep_df.iloc[sweep_df["F10_7"].sub(70).abs().argmin()]
        max_row = sweep_df.iloc[sweep_df["F10_7"].sub(230).abs().argmin()]

        print("\n  Sensitivity ratio (solar-min -> solar-max):")
        for m, key in [("Cascade", "cascade_N_final"), ("Multi-shell", "ms_N_final")]:
            ratio = max_row[key] / min_row[key]
            print(f"    {m}: N_final ratio = {ratio:.3f}  "
                  f"({min_row[key]:,.0f} -> {max_row[key]:,.0f})")

    # ------------------------------------------------------------------
    # 2. Sobol analysis
    # ------------------------------------------------------------------
    if not args.sweep_only:
        print("\n[3/4] Running multi-parameter Sobol analysis…")
        sa2 = SolarFluxSensitivity(
            years=years,
            n_ensemble=n_ens,
            sobol_n_base=sobol_n,
        )
        sobol_results = sa2.run_sobol_multi()

        # Persist results
        for model_key in ("cascade", "multishell"):
            d = sobol_results[model_key]
            np.save(
                RESULTS_DIR / f"sobol_{model_key}_outputs.npy",
                d["outputs"],
            )
        np.save(RESULTS_DIR / "sobol_param_values.npy", sobol_results["param_values"])
        with open(RESULTS_DIR / "sobol_indices.json", "w") as f:
            serialisable = {
                model_key: {
                    "S1": {k: float(v) for k, v in sobol_results[model_key]["S1"].items()},
                    "ST": {k: float(v) for k, v in sobol_results[model_key]["ST"].items()},
                    "S1_conf": {k: float(v) for k, v in sobol_results[model_key]["S1_conf"].items()},
                    "ST_conf": {k: float(v) for k, v in sobol_results[model_key]["ST_conf"].items()},
                }
                for model_key in ("cascade", "multishell")
            }
            json.dump(serialisable, f, indent=2)
        print("  Sobol indices saved -> results/sobol_indices.json")

        plot_sobol_bars(sobol_results)
        plot_output_distributions(sobol_results)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    elapsed = time.time() - t0
    print(f"\n[4/4] Analysis complete in {elapsed/60:.1f} min.  All figures -> results/")


if __name__ == "__main__":
    main()
