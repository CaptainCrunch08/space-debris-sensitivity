"""
Generate 4 publication-quality figures for the paper.

Figure order follows the results-section narrative:
  Fig 1 — Normalized sensitivity     (central structural finding; lead with this)
  Fig 2 — Raw final population        (absolute scale context)
  Fig 3 — Sobol ST comparison         (parameter importance ranking)
  Fig 4 — Sobol S1 vs ST              (interaction-dominated structure)

Reads:
    results/sweep_results.csv
    results/sobol_indices.json

Writes:
    results/fig1_normalized_sensitivity.png
    results/fig2_raw_population.png
    results/fig3_sobol_st_comparison.png
    results/fig4_sobol_s1_st.png

Run from the project root:
    python make_paper_figures.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

RESULTS = Path("results")

# ---------------------------------------------------------------------------
# Global style — larger fonts, clean look
# ---------------------------------------------------------------------------
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

CASCADE_COLOR = "#2563EB"
MS_COLOR = "#DC2626"


def save(fig: plt.Figure, name: str) -> None:
    path = RESULTS / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  saved -> {path}")


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(RESULTS / "sweep_results.csv")

with open(RESULTS / "sobol_indices.json") as f:
    sobol = json.load(f)

# ---------------------------------------------------------------------------
# Figure 1 — Normalized population (N / N at F10.7 = 70)
# Central structural finding: cascade is more sensitive than multi-shell.
# ---------------------------------------------------------------------------
ref_idx = df["F10_7"].sub(70).abs().idxmin()
cascade_norm = df["cascade_N_final"] / df.loc[ref_idx, "cascade_N_final"]
ms_norm = df["ms_N_final"] / df.loc[ref_idx, "ms_N_final"]

fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(df["F10_7"], cascade_norm,
        color=CASCADE_COLOR, lw=2.5, label="Cascade (PDCM)")
ax.plot(df["F10_7"], ms_norm,
        color=MS_COLOR, lw=2.5, ls="--", label="Multi-Shell (MSEM)")

ax.axhline(1.0, color="black", ls=":", lw=0.8, alpha=0.35)
ax.axvline(150, color="gray", ls=":", lw=1.2, alpha=0.7, label="Baseline (F10.7 = 150)")
ax.set_xlabel("F10.7 Solar Flux [sfu]")
ax.set_ylabel("Normalized Population  (N / N\u2080, F10.7=70)")
ax.set_title("Figure 1.  Normalized Sensitivity to Solar Activity")
ax.legend()
ax.grid(alpha=0.25)
fig.tight_layout()
save(fig, "fig1_normalized_sensitivity.png")

# ---------------------------------------------------------------------------
# Figure 2 — Raw final population vs F10.7 (log scale)
# Absolute scale context: shows why normalization was needed.
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(df["F10_7"], df["cascade_N_final"],
        color=CASCADE_COLOR, lw=2.5, label="Cascade (PDCM)")
ax.plot(df["F10_7"], df["ms_N_final"],
        color=MS_COLOR, lw=2.5, ls="--", label="Multi-Shell (MSEM)")

ax.axvline(150, color="gray", ls=":", lw=1.2, alpha=0.7, label="Baseline (F10.7 = 150)")
ax.set_xlabel("F10.7 Solar Flux [sfu]")
ax.set_ylabel("Final Debris Population (objects)")
ax.set_title("Figure 2.  Final Debris Population vs Solar Activity")
ax.set_yscale("log")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.legend()
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
save(fig, "fig2_raw_population.png")

# ---------------------------------------------------------------------------
# Shared Sobol data for Figures 3 and 4
# ---------------------------------------------------------------------------
param_labels = ["F10.7", "Launch Rate", "PMD Compliance"]
param_keys = ["F10_7", "launch_rate", "pmd_compliance"]

casc_s1 = [sobol["cascade"]["S1"][k] for k in param_keys]
casc_st = [sobol["cascade"]["ST"][k] for k in param_keys]
ms_s1 = [sobol["multishell"]["S1"][k] for k in param_keys]
ms_st = [sobol["multishell"]["ST"][k] for k in param_keys]

x = np.arange(len(param_labels))

# ---------------------------------------------------------------------------
# Figure 3 — Total-order ST side-by-side: clean parameter ranking
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))

w = 0.3
ax.bar(x - w / 2, casc_st, w, label="Cascade (PDCM)",
       color=CASCADE_COLOR, alpha=0.9)
ax.bar(x + w / 2, ms_st, w, label="Multi-Shell (MSEM)",
       color=MS_COLOR, alpha=0.9)

ax.set_xticks(x)
ax.set_xticklabels(param_labels)
ax.set_ylabel("Total-Order Sobol Index (ST)")
ax.set_title("Figure 3.  Parameter Importance Comparison (Total-Order ST)")
ax.axhline(0, color="black", lw=0.8)
ax.legend()
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
save(fig, "fig3_sobol_st_comparison.png")

# ---------------------------------------------------------------------------
# Figure 4 — S1 and ST grouped: shows interaction-dominated structure
# ---------------------------------------------------------------------------
w = 0.18

fig, ax = plt.subplots(figsize=(9, 5))

ax.bar(x - 1.5 * w, casc_s1, w, label="Cascade S1",
       color=CASCADE_COLOR, alpha=1.0)
ax.bar(x - 0.5 * w, casc_st, w, label="Cascade ST",
       color=CASCADE_COLOR, alpha=0.5, hatch="//", edgecolor=CASCADE_COLOR)
ax.bar(x + 0.5 * w, ms_s1, w, label="Multi-Shell S1",
       color=MS_COLOR, alpha=1.0)
ax.bar(x + 1.5 * w, ms_st, w, label="Multi-Shell ST",
       color=MS_COLOR, alpha=0.5, hatch="//", edgecolor=MS_COLOR)

ax.set_xticks(x)
ax.set_xticklabels(param_labels)
ax.set_ylabel("Sobol Index")
ax.set_title("Figure 4.  First-Order (S1) and Total-Order (ST) Sobol Sensitivity Indices")
ax.axhline(0, color="black", lw=0.8)
ax.legend(fontsize=9, ncol=2)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
save(fig, "fig4_sobol_s1_st.png")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\nSensitivity ratios (N at solar max / N at solar min):")
min_row = df.loc[df["F10_7"].sub(70).abs().idxmin()]
max_row = df.loc[df["F10_7"].sub(230).abs().idxmin()]
print(f"  Cascade:     {max_row['cascade_N_final'] / min_row['cascade_N_final']:.3f}")
print(f"  Multi-Shell: {max_row['ms_N_final'] / min_row['ms_N_final']:.3f}")

print("\nTotal-order Sobol ST:")
print(f"  {'':20s} {'Cascade':>10} {'Multi-Shell':>12}")
for label, key in zip(param_labels, param_keys):
    print(f"  {label:20s} {sobol['cascade']['ST'][key]:>10.3f} {sobol['multishell']['ST'][key]:>12.3f}")

print("\nAll paper figures saved to results/")
