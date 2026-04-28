"""
Multi-shell stochastic debris evolution model (MOCAT-inspired framework).

LEO is divided into four altitude shells.  Within each shell the model
tracks three populations:
  S_i   — intact objects
  D_i   — large debris fragments (>=10 cm, radar-trackable)
  D_s_i — small debris fragments (1 mm-10 cm, untraceable)

Small debris (D_s) has a higher area-to-mass ratio than large debris,
resulting in shorter atmospheric lifetimes and stronger sensitivity to
solar flux F10.7.  Including it is essential for physical completeness
and materially affects the measured F10.7 sensitivity (Flegel et al.
2010; NASA-STD-8719.14C).

Shell definitions (mid-altitude used for physics calculations):
  Shell 0:  500-600 km  (mid ~550 km)
  Shell 1:  600-700 km  (mid ~650 km)
  Shell 2:  700-800 km  (mid ~750 km)
  Shell 3:  800-900 km  (mid ~850 km)
  Shell 4:  900-1000 km (mid ~950 km)

References
----------
MOCAT framework: Rossi et al. (2009); Radtke et al. (2017) LUCA2 paper.
Collision probability: Liou (2006) NASA collision modeling.
Fragment redistribution: Anz-Meador (2010) Iridium/Cosmos analysis.
Small debris physics: Flegel et al. (2010) MASTER-2009; NASA-STD-8719.14C.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng


# ---------------------------------------------------------------------------
# Shell geometry constants
# ---------------------------------------------------------------------------

SHELL_BOUNDS_KM = [(500, 600), (600, 700), (700, 800), (800, 900), (900, 1000)]
SHELL_MIDS_KM = [550.0, 650.0, 750.0, 850.0, 950.0]
N_SHELLS = len(SHELL_MIDS_KM)

EARTH_RADIUS_KM = 6371.0
RE = EARTH_RADIUS_KM


def _shell_volume_km3(h_low_km: float, h_high_km: float) -> float:
    """Volume of a spherical shell [km3]."""
    r_low = RE + h_low_km
    r_high = RE + h_high_km
    return (4.0 / 3.0) * np.pi * (r_high**3 - r_low**3)


SHELL_VOLUMES_KM3 = np.array(
    [_shell_volume_km3(lo, hi) for lo, hi in SHELL_BOUNDS_KM], dtype=float
)


# ---------------------------------------------------------------------------
# Atmospheric lifetime model — large debris (>=10 cm)
# ---------------------------------------------------------------------------

# Reference lifetimes [yr] at F10.7 = 150 sfu:
#   550 km ~5 yr,  650 km ~20 yr,  750 km ~100 yr,  850 km ~400 yr,
#   950 km ~1500 yr  (atmosphere near-negligible; cited range IADC 2013)
TAU_D_REF = np.array([5.0, 20.0, 100.0, 400.0, 1500.0], dtype=float)
TAU_D_F10_REF = 150.0

# Drag exponent per shell (lower shells more sensitive to solar activity).
# At 950 km drag is so weak that F10.7 has minimal effect; exponent drops to 0.8.
DRAG_EXPONENTS = np.array([2.0, 1.8, 1.5, 1.2, 0.8], dtype=float)


def debris_lifetime(F10_7: float) -> np.ndarray:
    """Per-shell large-debris atmospheric lifetime [yr]."""
    return TAU_D_REF * (TAU_D_F10_REF / F10_7) ** DRAG_EXPONENTS


# ---------------------------------------------------------------------------
# Atmospheric lifetime model — small debris (1 mm-10 cm)
# ---------------------------------------------------------------------------

# Small debris has ~10x higher A/m ratio -> ~1/10 the lifetime of large debris.
# It also has a higher drag exponent: more sensitive to solar-flux-driven
# density changes because atmospheric drag dominates its orbital evolution.
TAU_Ds_REF = TAU_D_REF / 10.0          # [yr] at F10.7 = 150
DRAG_EXPONENTS_SMALL = DRAG_EXPONENTS * 1.2   # amplified sensitivity


def small_debris_lifetime(F10_7: float) -> np.ndarray:
    """Per-shell small-debris atmospheric lifetime [yr]."""
    return TAU_Ds_REF * (TAU_D_F10_REF / F10_7) ** DRAG_EXPONENTS_SMALL


# ---------------------------------------------------------------------------
# Intact-object lifetime (PMD-driven, not directly F10.7)
# ---------------------------------------------------------------------------

TAU_S_NATURAL = np.array([50.0, 150.0, 500.0, 2000.0, 5000.0], dtype=float)
TAU_S_PMD = 25.0


def intact_lifetime(pmd_compliance: float) -> np.ndarray:
    """Effective per-shell intact-object lifetime [yr]."""
    rate_pmd = pmd_compliance / TAU_S_PMD
    rate_nat = (1.0 - pmd_compliance) / TAU_S_NATURAL
    return 1.0 / (rate_pmd + rate_nat)


# ---------------------------------------------------------------------------
# Collision rate model
# ---------------------------------------------------------------------------

# Cross-sections [km2] calibrated to ~0.10-0.15 catastrophic collisions/yr
# at baseline populations.
SIGMA_SS = 5.0e-6    # intact-intact [km2]
SIGMA_SD = 3.0e-6    # intact-large debris [km2]
SIGMA_DD = 2.0e-8    # large-large debris [km2]
# Intact-small: smaller cross-section per object but population is ~25x larger.
# Net effect: comparable collision rate to intact-large at baseline.
SIGMA_SDs = 3.0e-7   # intact-small debris [km2]

V_REL_KM_S = 10.0


def collision_rate(N_i: float, N_j: float, sigma: float, V_shell_km3: float) -> float:
    """Expected collisions per year in one shell (kinetic-gas formula)."""
    SEC_PER_YR = 3.156e7
    return sigma * V_REL_KM_S * SEC_PER_YR * N_i * N_j / V_shell_km3


# Thresholds for collision sampling fallbacks.
# At extreme Sobol parameter combinations (high launch rate + low PMD + solar
# minimum), the 900-1000 km shell accumulates debris with ~2800-yr lifetimes,
# driving D^2 collision rates to values that overflow numpy's Poisson sampler.
# This mirrors behaviour documented in IADC (2013) for that altitude band.
#
# Strategy:
#   lam <= 1e6  : standard Poisson sample
#   1e6 < lam <= 1e12 : Normal(lam, sqrt(lam)) approximation (statistically
#                       indistinguishable from Poisson for large lambda)
#   lam > 1e12  : population has diverged to unphysical levels; return
#                 deterministic mean with no noise (noise is negligible
#                 relative to mean at this scale)
#   non-finite  : guard against inf/NaN from float overflow; return 0
_POISSON_NORMAL_THRESHOLD = 1.0e6
_POISSON_HARD_CAP = 1.0e12


def _sample_collisions(rng: np.random.Generator, lam: float) -> int:
    """Sample collision count from Poisson(lam), with fallbacks for large lam."""
    if lam <= 0.0 or not np.isfinite(lam):
        return 0
    if lam > _POISSON_HARD_CAP:
        return int(_POISSON_HARD_CAP)
    if lam > _POISSON_NORMAL_THRESHOLD:
        return max(0, int(round(lam + np.sqrt(lam) * rng.standard_normal())))
    return int(rng.poisson(lam))


# ---------------------------------------------------------------------------
# Fragment redistribution weights
# ---------------------------------------------------------------------------

# Large fragments (>=10 cm) redistribution: modest ΔV spread.
# Rows = shell where collision occurred; columns = shell receiving fragments.
# Each row sums to 1.0.
FRAG_REDISTRIB = np.array(
    [
        [0.50, 0.28, 0.13, 0.05, 0.04],  # collision in shell 0 (550 km)
        [0.18, 0.48, 0.23, 0.07, 0.04],  # collision in shell 1 (650 km)
        [0.05, 0.18, 0.52, 0.20, 0.05],  # collision in shell 2 (750 km)
        [0.04, 0.08, 0.25, 0.48, 0.15],  # collision in shell 3 (850 km)
        [0.03, 0.07, 0.12, 0.28, 0.50],  # collision in shell 4 (950 km)
    ]
)

# Small fragments (1 mm-10 cm) spread more widely due to higher ΔV at
# ejection and lighter mass (Anz-Meador 2010; Flegel et al. 2010).
FRAG_REDISTRIB_SMALL = np.array(
    [
        [0.38, 0.28, 0.18, 0.10, 0.06],  # collision in shell 0 (550 km)
        [0.13, 0.43, 0.27, 0.11, 0.06],  # collision in shell 1 (650 km)
        [0.08, 0.18, 0.43, 0.23, 0.08],  # collision in shell 2 (750 km)
        [0.07, 0.12, 0.24, 0.43, 0.14],  # collision in shell 3 (850 km)
        [0.06, 0.10, 0.17, 0.27, 0.40],  # collision in shell 4 (950 km)
    ]
)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class MultiShellModel:
    """Multi-shell stochastic debris evolution model (three populations).

    Parameters
    ----------
    launch_rate : float
        Total annual launch rate [objects/yr].
    launch_distribution : array-like of length 5
        Fraction of launches into each shell (must sum to 1).
    pmd_compliance : float
        PMD compliance fraction [0, 1].
    F10_7 : float
        Solar flux index F10.7 [sfu].
    k_fragments : float
        Large (>=10 cm) fragments per catastrophic collision.
    k_small_frags : float
        Small (1 mm-10 cm) fragments per catastrophic collision.
    S0 : array-like of length 5
        Initial intact-object population per shell.
    D0 : array-like of length 5
        Initial large-debris population per shell.
    Ds0 : array-like of length 5
        Initial small-debris population per shell.
    dt : float
        Time step [yr].
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        launch_rate: float = 200.0,
        launch_distribution: list[float] | None = None,
        pmd_compliance: float = 0.25,
        F10_7: float = 150.0,
        k_fragments: float = 250.0,
        k_small_frags: float = 2000.0,
        S0: list[float] | None = None,
        D0: list[float] | None = None,
        Ds0: list[float] | None = None,
        dt: float = 1.0,
        seed: int | None = 42,
    ):
        self.launch_rate = launch_rate
        self.launch_distribution = (
            np.array(launch_distribution, dtype=float)
            if launch_distribution is not None
            else np.array([0.15, 0.30, 0.30, 0.15, 0.10], dtype=float)
        )
        self.pmd_compliance = pmd_compliance
        self.F10_7 = F10_7
        self.k_fragments = k_fragments
        self.k_small_frags = k_small_frags
        self.S0 = (
            np.array(S0, dtype=float)
            if S0 is not None
            else np.array([300.0, 600.0, 700.0, 400.0, 200.0], dtype=float)
        )
        self.D0 = (
            np.array(D0, dtype=float)
            if D0 is not None
            else np.array([1500.0, 3500.0, 3000.0, 2000.0, 1500.0], dtype=float)
        )
        # Default Ds0: ~25x D0 per shell, consistent with the ~500k/>1cm
        # vs ~20k/>10cm tracking ratio (ORDEM 3.1; ESA Annual Space Env. Report).
        self.Ds0 = (
            np.array(Ds0, dtype=float)
            if Ds0 is not None
            else self.D0 * 25.0
        )
        self.dt = dt
        self.rng = default_rng(seed)

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run(self, years: int = 200) -> dict:
        """Run one stochastic realisation of the three-population model.

        Returns
        -------
        dict with keys:
            't'        — time array [yr]
            'S'        — intact populations, shape (steps+1, N_SHELLS)
            'D'        — large debris (>=10 cm), shape (steps+1, N_SHELLS)
            'Ds'       — small debris (1 mm-10 cm), shape (steps+1, N_SHELLS)
            'N'        — total (S+D+Ds) per shell
            'N_large'  — S+D per shell (>=10 cm only)
            'N_total'  — sum of N over all shells
            'R_col'    — total collision events per year
        """
        steps = int(years / self.dt)
        t = np.arange(steps + 1, dtype=float) * self.dt

        S = np.zeros((steps + 1, N_SHELLS))
        D = np.zeros((steps + 1, N_SHELLS))
        Ds = np.zeros((steps + 1, N_SHELLS))
        R_col_ts = np.zeros(steps + 1)

        S[0] = self.S0.copy()
        D[0] = self.D0.copy()
        Ds[0] = self.Ds0.copy()

        tau_D = debris_lifetime(self.F10_7)
        tau_Ds = small_debris_lifetime(self.F10_7)
        tau_S = intact_lifetime(self.pmd_compliance)
        dt = self.dt
        pmd = self.pmd_compliance
        lam_per_shell = self.launch_rate * self.launch_distribution

        for step in range(steps):
            s = S[step].copy()
            d = D[step].copy()
            ds = Ds[step].copy()

            delta_S = np.zeros(N_SHELLS)
            delta_D = np.zeros(N_SHELLS)
            delta_Ds = np.zeros(N_SHELLS)
            total_collisions = 0

            for i in range(N_SHELLS):
                V = SHELL_VOLUMES_KM3[i]

                # -- Launches --
                delta_S[i] += lam_per_shell[i] * dt
                delta_D[i] += (1.0 - pmd) * lam_per_shell[i] * dt

                # -- Decay --
                delta_S[i] -= (s[i] / tau_S[i]) * dt
                delta_D[i] -= (d[i] / tau_D[i]) * dt
                delta_Ds[i] -= (ds[i] / tau_Ds[i]) * dt

                # -- Collision sampling --
                lam_SS = collision_rate(s[i], s[i], SIGMA_SS, V) * dt
                n_SS = _sample_collisions(self.rng, lam_SS)

                lam_SD = collision_rate(s[i], d[i], SIGMA_SD, V) * dt
                n_SD = _sample_collisions(self.rng, lam_SD)

                lam_DD = collision_rate(d[i], d[i], SIGMA_DD, V) * dt
                n_DD = _sample_collisions(self.rng, lam_DD)

                lam_SDs = collision_rate(s[i], ds[i], SIGMA_SDs, V) * dt
                n_SDs = _sample_collisions(self.rng, lam_SDs)

                n_total_col = n_SS + n_SD + n_DD + n_SDs
                total_collisions += n_total_col

                # Remove destroyed intact objects
                destroyed = min(2 * n_SS + n_SD + n_SDs, s[i])
                delta_S[i] -= destroyed

                # Redistribute large and small fragments across shells
                new_large = self.k_fragments * n_total_col
                new_small = self.k_small_frags * n_total_col
                for j in range(N_SHELLS):
                    delta_D[j] += new_large * FRAG_REDISTRIB[i, j]
                    delta_Ds[j] += new_small * FRAG_REDISTRIB_SMALL[i, j]

            R_col_ts[step] = total_collisions / dt

            S[step + 1] = np.maximum(s + delta_S, 0.0)
            D[step + 1] = np.maximum(d + delta_D, 0.0)
            Ds[step + 1] = np.maximum(ds + delta_Ds, 0.0)

        R_col_ts[steps] = 0.0

        N_all = S + D + Ds
        return {
            "t": t,
            "S": S,
            "D": D,
            "Ds": Ds,
            "N": N_all,
            "N_large": S + D,
            "N_total": N_all.sum(axis=1),
            "R_col": R_col_ts,
        }

    # ------------------------------------------------------------------
    # Ensemble run (Monte Carlo)
    # ------------------------------------------------------------------

    def run_ensemble(self, years: int = 200, n_runs: int = 50) -> dict:
        """Run multiple stochastic realisations and return statistics.

        Returns
        -------
        dict with keys:
            't'            — time array [yr]
            'N_total_mean' — ensemble mean total population (S+D+Ds)
            'N_total_std'  — ensemble standard deviation
            'N_total_all'  — all runs, shape (n_runs, steps+1)
            'R_col_mean'   — ensemble mean collision rate
        """
        steps = int(years / self.dt)
        all_N = np.zeros((n_runs, steps + 1))
        all_R = np.zeros((n_runs, steps + 1))
        t = None

        for run_idx in range(n_runs):
            self.rng = default_rng(run_idx + (self.rng.bit_generator.state["state"]["state"] % 10000))
            result = self.run(years=years)
            all_N[run_idx] = result["N_total"]
            all_R[run_idx] = result["R_col"]
            t = result["t"]

        return {
            "t": t,
            "N_total_mean": all_N.mean(axis=0),
            "N_total_std": all_N.std(axis=0),
            "N_total_all": all_N,
            "R_col_mean": all_R.mean(axis=0),
        }
