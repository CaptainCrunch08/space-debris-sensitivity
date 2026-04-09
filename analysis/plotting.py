"""
Plotting utilities for the debris sensitivity analysis.
All figures are saved to results/ and also returned for inline display.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

STYLE = {
    "cascade": {"color": "#2563EB", "label": "Cascade (Kessler-type)"},
    "multishell": {"color": "#DC2626", "label": "Multi-Shell Stochastic"},
}


def _save(fig: plt.Figure, name: str) -> None:
    path = RESULTS_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
# Baseline trajectory comparison
# ---------------------------------------------------------------------------

def plot_baseline_trajectories(
    cascade_result: dict,
    multishell_result: dict,
    save: bool = True,
) -> plt.Figure:
    """Plot total debris population over time for both models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Population
    ax = axes[0]
    ax.plot(cascade_result["t"], cascade_result["N"],
            color=STYLE["cascade"]["color"], lw=2, label=STYLE["cascade"]["label"])
    ax.plot(multishell_result["t"], multishell_result["N_total"],
            color=STYLE["multishell"]["color"], lw=2, ls="--", label=STYLE["multishell"]["label"])
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Total objects (S + D)")
    ax.set_title("Baseline debris population evolution")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    ax.grid(alpha=0.3)

    # Collision rate
    ax = axes[1]
    ax.plot(cascade_result["t"], cascade_result["R_col"],
            color=STYLE["cascade"]["color"], lw=2, label=STYLE["cascade"]["label"])
    ax.plot(multishell_result["t"], multishell_result["R_col_mean"],
            color=STYLE["multishell"]["color"], lw=2, ls="--", label=STYLE["multishell"]["label"])
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel("Collision rate [events/yr]")
    ax.set_title("Baseline collision rate")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save:
        _save(fig, "01_baseline_trajectories.png")
    return fig


# ---------------------------------------------------------------------------
# Parameter sweep
# ---------------------------------------------------------------------------

def plot_sweep(sweep_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot response curves from the F10.7 parameter sweep."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    metrics = [
        ("N_final", "Final total population"),
        ("R_col_final", "Final collision rate [/yr]"),
        ("N_integrated", "Integrated population [obj·yr]"),
    ]

    for ax, (metric, ylabel) in zip(axes, metrics):
        c_col = f"cascade_{metric}"
        ms_col = f"ms_{metric}"

        # Normalise to value at F10.7 = 150 for cleaner comparison
        ref_idx = (sweep_df["F10_7"] - 150.0).abs().idxmin()
        c_ref = sweep_df.loc[ref_idx, c_col]
        ms_ref = sweep_df.loc[ref_idx, ms_col]

        ax.plot(sweep_df["F10_7"], sweep_df[c_col] / c_ref,
                color=STYLE["cascade"]["color"], lw=2, label=STYLE["cascade"]["label"])
        ax.plot(sweep_df["F10_7"], sweep_df[ms_col] / ms_ref,
                color=STYLE["multishell"]["color"], lw=2, ls="--",
                label=STYLE["multishell"]["label"])

        ax.axvline(70, color="gray", ls=":", lw=1, label="Solar min (70)")
        ax.axvline(230, color="orange", ls=":", lw=1, label="Solar max (230)")
        ax.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.5)
        ax.set_xlabel("F10.7 [sfu]")
        ax.set_ylabel(f"Normalised {ylabel}")
        ax.set_title(ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("F10.7 sensitivity sweep — normalised to F10.7 = 150", fontsize=13)
    fig.tight_layout()
    if save:
        _save(fig, "02_sweep_response_curves.png")
    return fig


def plot_sweep_absolute(sweep_df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Plot absolute (non-normalised) sweep outputs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(sweep_df["F10_7"], sweep_df["cascade_N_final"],
            color=STYLE["cascade"]["color"], lw=2, label=STYLE["cascade"]["label"])
    ax2 = ax.twinx()
    ax2.plot(sweep_df["F10_7"], sweep_df["ms_N_final"],
             color=STYLE["multishell"]["color"], lw=2, ls="--",
             label=STYLE["multishell"]["label"])
    ax.set_xlabel("F10.7 [sfu]")
    ax.set_ylabel("Cascade final N", color=STYLE["cascade"]["color"])
    ax2.set_ylabel("Multi-shell final N", color=STYLE["multishell"]["color"])
    ax.set_title("Final total population vs F10.7")
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(sweep_df["F10_7"], sweep_df["cascade_R_col_final"],
            color=STYLE["cascade"]["color"], lw=2, label=STYLE["cascade"]["label"])
    ax.plot(sweep_df["F10_7"], sweep_df["ms_R_col_final"],
            color=STYLE["multishell"]["color"], lw=2, ls="--",
            label=STYLE["multishell"]["label"])
    ax.set_xlabel("F10.7 [sfu]")
    ax.set_ylabel("Collision rate [/yr]")
    ax.set_title("Final collision rate vs F10.7")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save:
        _save(fig, "03_sweep_absolute.png")
    return fig


# ---------------------------------------------------------------------------
# Sobol bar chart
# ---------------------------------------------------------------------------

def plot_sobol_bars(sobol_results: dict, save: bool = True) -> plt.Figure:
    """Bar chart of first-order and total Sobol indices for both models."""
    names = sobol_results["problem"]["names"]
    x = np.arange(len(names))
    width = 0.2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, model_key, style in [
        (axes[0], "cascade", STYLE["cascade"]),
        (axes[1], "multishell", STYLE["multishell"]),
    ]:
        s1 = [sobol_results[model_key]["S1"][n] for n in names]
        st = [sobol_results[model_key]["ST"][n] for n in names]
        s1_conf = [sobol_results[model_key]["S1_conf"][n] for n in names]
        st_conf = [sobol_results[model_key]["ST_conf"][n] for n in names]

        bars1 = ax.bar(x - width / 2, s1, width, label="S1 (first-order)",
                       color=style["color"], alpha=0.85, yerr=s1_conf, capsize=4)
        bars2 = ax.bar(x + width / 2, st, width, label="ST (total-effect)",
                       color=style["color"], alpha=0.45, yerr=st_conf, capsize=4,
                       hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=15, ha="right")
        ax.set_ylim(0, 1.1)
        ax.set_ylabel("Sobol index")
        ax.set_title(style["label"])
        ax.axhline(1.0, color="gray", ls=":", lw=0.8)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Sobol variance-based sensitivity indices\n(output: final total debris population)",
                 fontsize=12)
    fig.tight_layout()
    if save:
        _save(fig, "04_sobol_bar_chart.png")
    return fig


# ---------------------------------------------------------------------------
# Output distribution comparison
# ---------------------------------------------------------------------------

def plot_output_distributions(sobol_results: dict, save: bool = True) -> plt.Figure:
    """Histogram of model outputs across the Sobol sample space."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for ax, model_key, style in [
        (axes[0], "cascade", STYLE["cascade"]),
        (axes[1], "multishell", STYLE["multishell"]),
    ]:
        vals = sobol_results[model_key]["outputs"]
        ax.hist(vals, bins=40, color=style["color"], alpha=0.75, edgecolor="white")
        ax.axvline(np.median(vals), color="black", lw=1.5, label=f"Median: {np.median(vals):,.0f}")
        ax.set_xlabel("Final total debris population")
        ax.set_ylabel("Count")
        ax.set_title(style["label"])
        ax.legend()
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    fig.suptitle("Distribution of final debris population across all parameter samples", fontsize=12)
    fig.tight_layout()
    if save:
        _save(fig, "05_output_distributions.png")
    return fig
