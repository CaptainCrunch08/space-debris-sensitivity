"""
Solar-flux sensitivity analysis for cascade and multi-shell debris models.

Two complementary approaches are implemented:

1. Parameter sweep  — simple, transparent, easy to visualise.
   Run both models across a grid of F10.7 values and record scalar
   output metrics (final population, final collision rate, integrated
   population).  Compare response curves.

2. Sobol variance-based global sensitivity analysis  — statistically
   rigorous.  Computes first-order (S1) and total-effect (ST) Sobol
   indices for each uncertain parameter using the Saltelli sampling
   scheme (SALib library).  Reports what fraction of output variance
   each parameter accounts for.

   Primary uncertain parameter: F10.7 (solar flux).
   Secondary parameters (optionally included): launch_rate, pmd_compliance.

References
----------
Sobol (2001) — global sensitivity indices and Monte Carlo estimation.
Pianosi et al. (2016) — sensitivity analysis workflow for env. models.
SALib: Herman & Usher (2017), doi:10.21105/joss.00097.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from SALib.sample import saltelli
from SALib.analyze import sobol

from models.cascade_model import CascadeModel
from models.multishell_model import MultiShellModel


# ---------------------------------------------------------------------------
# Scalar output extractors
# ---------------------------------------------------------------------------

def _cascade_outputs(model: CascadeModel, years: int) -> dict[str, float]:
    res = model.run(years=years)
    return {
        "N_final": float(res["N"][-1]),
        "D_final": float(res["D"][-1]),
        "R_col_final": float(res["R_col"][-1]),
        "N_integrated": float(np.trapezoid(res["N"], res["t"])),
    }


def _multishell_outputs(model: MultiShellModel, years: int, n_runs: int) -> dict[str, float]:
    ens = model.run_ensemble(years=years, n_runs=n_runs)
    return {
        "N_final": float(ens["N_total_mean"][-1]),
        "R_col_final": float(ens["R_col_mean"][-1]),
        "N_integrated": float(np.trapezoid(ens["N_total_mean"], ens["t"])),
    }


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class SolarFluxSensitivity:
    """Cross-model sensitivity analysis focused on the solar flux variable.

    Parameters
    ----------
    years : int
        Simulation horizon [yr].
    n_ensemble : int
        Monte Carlo realisations per parameter set for the multi-shell model.
    f107_range : tuple[float, float]
        (min, max) F10.7 values to explore.  NOAA/NGDC long-record range
        spans ~70 sfu (deep solar minimum) to ~230 sfu (solar maximum).
    sweep_n_points : int
        Number of grid points for the parameter sweep.
    sobol_n_base : int
        Base sample size N for Saltelli/Sobol sampling.
        Total evaluations = N * (2*D + 2) where D = number of parameters.
    cascade_kwargs : dict
        Additional keyword arguments forwarded to CascadeModel (excluding
        F10_7, which is varied by the analysis).
    multishell_kwargs : dict
        Additional keyword arguments forwarded to MultiShellModel.
    """

    def __init__(
        self,
        years: int = 200,
        n_ensemble: int = 30,
        f107_range: tuple[float, float] = (70.0, 230.0),
        sweep_n_points: int = 33,
        sobol_n_base: int = 256,
        cascade_kwargs: dict | None = None,
        multishell_kwargs: dict | None = None,
    ):
        self.years = years
        self.n_ensemble = n_ensemble
        self.f107_range = f107_range
        self.sweep_n_points = sweep_n_points
        self.sobol_n_base = sobol_n_base
        self.cascade_kwargs = cascade_kwargs or {}
        self.multishell_kwargs = multishell_kwargs or {}

    # ------------------------------------------------------------------
    # 1. Parameter sweep
    # ------------------------------------------------------------------

    def run_sweep(self) -> pd.DataFrame:
        """Run a univariate F10.7 sweep on both models.

        Returns
        -------
        DataFrame with columns:
            F10_7, cascade_N_final, cascade_D_final, cascade_R_col_final,
            cascade_N_integrated, ms_N_final, ms_R_col_final,
            ms_N_integrated
        """
        f107_grid = np.linspace(self.f107_range[0], self.f107_range[1], self.sweep_n_points)
        records = []

        for f107 in f107_grid:
            c_model = CascadeModel(F10_7=f107, **self.cascade_kwargs)
            c_out = _cascade_outputs(c_model, self.years)

            ms_model = MultiShellModel(F10_7=f107, **self.multishell_kwargs)
            ms_out = _multishell_outputs(ms_model, self.years, self.n_ensemble)

            row = {"F10_7": f107}
            row.update({f"cascade_{k}": v for k, v in c_out.items()})
            row.update({f"ms_{k}": v for k, v in ms_out.items()})
            records.append(row)
            print(f"  sweep F10.7={f107:.1f}  cascade_N={c_out['N_final']:.0f}  ms_N={ms_out['N_final']:.0f}")

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # 2. Sobol variance-based sensitivity (single parameter: F10.7)
    # ------------------------------------------------------------------

    def run_sobol_single(self) -> dict:
        """Sobol analysis varying only F10.7.

        Because only one parameter is varied, the first-order index S1
        is by definition 1.0 for both models.  This run is therefore
        mainly used to characterise the *magnitude* of the response and
        to compute uncertainty bounds for the mean output via bootstrap.

        Returns
        -------
        dict with 'cascade' and 'multishell' sub-dicts containing
        the Sobol indices and output arrays.
        """
        problem = {
            "num_vars": 1,
            "names": ["F10_7"],
            "bounds": [list(self.f107_range)],
        }
        param_values = saltelli.sample(problem, self.sobol_n_base, calc_second_order=False)

        c_outputs = []
        ms_outputs = []

        for i, row in enumerate(param_values):
            f107 = float(row[0])
            c_m = CascadeModel(F10_7=f107, **self.cascade_kwargs)
            c_outputs.append(_cascade_outputs(c_m, self.years)["N_final"])

            ms_m = MultiShellModel(F10_7=f107, **self.multishell_kwargs)
            ms_outputs.append(_multishell_outputs(ms_m, self.years, self.n_ensemble)["N_final"])

            if i % 50 == 0:
                print(f"  sobol_single sample {i}/{len(param_values)}")

        c_arr = np.array(c_outputs)
        ms_arr = np.array(ms_outputs)

        c_si = sobol.analyze(problem, c_arr, calc_second_order=False, print_to_console=False)
        ms_si = sobol.analyze(problem, ms_arr, calc_second_order=False, print_to_console=False)

        return {
            "problem": problem,
            "param_values": param_values,
            "cascade": {"outputs": c_arr, "S1": c_si["S1"], "ST": c_si["ST"]},
            "multishell": {"outputs": ms_arr, "S1": ms_si["S1"], "ST": ms_si["ST"]},
        }

    # ------------------------------------------------------------------
    # 3. Sobol multi-parameter analysis (F10.7, launch_rate, pmd)
    # ------------------------------------------------------------------

    def run_sobol_multi(
        self,
        launch_rate_range: tuple[float, float] = (100.0, 400.0),
        pmd_range: tuple[float, float] = (0.50, 0.99),
    ) -> dict:
        """Three-parameter Sobol analysis: F10.7, launch_rate, pmd_compliance.

        First-order (S1) and total-effect (ST) indices reveal:
          - How much variance each parameter independently explains (S1).
          - How much variance each parameter explains including interactions (ST).
          - Whether S1 ≪ ST, indicating strong parameter interactions.

        Returns
        -------
        dict containing Sobol indices for both models and all parameters.
        """
        problem = {
            "num_vars": 3,
            "names": ["F10_7", "launch_rate", "pmd_compliance"],
            "bounds": [
                list(self.f107_range),
                list(launch_rate_range),
                list(pmd_range),
            ],
        }
        param_values = saltelli.sample(problem, self.sobol_n_base, calc_second_order=False)

        c_N_final = np.zeros(len(param_values))
        ms_N_final = np.zeros(len(param_values))

        for i, row in enumerate(param_values):
            f107, lam, pmd = float(row[0]), float(row[1]), float(row[2])

            c_kw = {**self.cascade_kwargs, "F10_7": f107, "launch_rate": lam, "pmd_compliance": pmd}
            c_m = CascadeModel(**c_kw)
            c_N_final[i] = _cascade_outputs(c_m, self.years)["N_final"]

            ms_kw = {**self.multishell_kwargs, "F10_7": f107, "launch_rate": lam, "pmd_compliance": pmd}
            ms_m = MultiShellModel(**ms_kw)
            ms_N_final[i] = _multishell_outputs(ms_m, self.years, self.n_ensemble)["N_final"]

            if i % 100 == 0:
                print(f"  sobol_multi sample {i}/{len(param_values)}")

        c_si = sobol.analyze(problem, c_N_final, calc_second_order=False, print_to_console=False)
        ms_si = sobol.analyze(problem, ms_N_final, calc_second_order=False, print_to_console=False)

        results = {
            "problem": problem,
            "param_values": param_values,
            "cascade": {
                "outputs": c_N_final,
                "S1": dict(zip(problem["names"], c_si["S1"])),
                "ST": dict(zip(problem["names"], c_si["ST"])),
                "S1_conf": dict(zip(problem["names"], c_si["S1_conf"])),
                "ST_conf": dict(zip(problem["names"], c_si["ST_conf"])),
            },
            "multishell": {
                "outputs": ms_N_final,
                "S1": dict(zip(problem["names"], ms_si["S1"])),
                "ST": dict(zip(problem["names"], ms_si["ST"])),
                "S1_conf": dict(zip(problem["names"], ms_si["S1_conf"])),
                "ST_conf": dict(zip(problem["names"], ms_si["ST_conf"])),
            },
        }

        _print_sobol_summary(results)
        return results


def _print_sobol_summary(results: dict) -> None:
    """Pretty-print Sobol index comparison table."""
    names = results["problem"]["names"]
    print("\n=== Sobol Sensitivity Indices — Final Debris Population ===")
    print(f"{'Parameter':<20} {'Casc S1':>10} {'Casc ST':>10} {'MS S1':>10} {'MS ST':>10}")
    print("-" * 62)
    for name in names:
        cs1 = results["cascade"]["S1"][name]
        cst = results["cascade"]["ST"][name]
        ms1 = results["multishell"]["S1"][name]
        mst = results["multishell"]["ST"][name]
        print(f"{name:<20} {cs1:>10.3f} {cst:>10.3f} {ms1:>10.3f} {mst:>10.3f}")
    print()
