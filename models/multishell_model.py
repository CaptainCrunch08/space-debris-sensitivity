"""
Multi-shell stochastic debris evolution model (MOCAT-inspired framework).

LEO is divided into four altitude shells.  Within each shell the model
tracks populations of intact objects and debris fragments.  Collisions
are drawn stochastically from a Poisson distribution; fragments are
redistributed across shells based on a simplified ΔV spread.

Shell definitions (mid-altitude used for physics calculations):
  Shell 0:  500–600 km  (mid ≈ 550 km)
  Shell 1:  600–700 km  (mid ≈ 650 km)
  Shell 2:  700–800 km  (mid ≈ 750 km)
  Shell 3:  800–900 km  (mid ≈ 850 km)

Solar flux F10.7 controls atmospheric drag at each shell altitude.
Higher F10.7 → denser atmosphere → shorter debris lifetimes, especially
in lower shells.  The altitude-dependent lifetime model is adapted from
NRLMSISE-00/JB2008 comparisons (Sagnieres & Sharf 2017).

References
----------
MOCAT framework: Rossi et al. (2009); Radtke et al. (2017) LUCA2 paper.
Collision probability: Liou (2006) NASA collision modeling.
Fragment redistribution: Anz-Meador (2010) Iridium/Cosmos analysis.
"""

from __future__ import annotations

import numpy as np
from numpy.random import default_rng


# ---------------------------------------------------------------------------
# Shell geometry constants
# ---------------------------------------------------------------------------

SHELL_BOUNDS_KM = [(500, 600), (600, 700), (700, 800), (800, 900)]
SHELL_MIDS_KM = [550.0, 650.0, 750.0, 850.0]
N_SHELLS = len(SHELL_MIDS_KM)

EARTH_RADIUS_KM = 6371.0
RE = EARTH_RADIUS_KM

def _shell_volume_km3(h_low_km: float, h_high_km: float) -> float:
    """Volume of a spherical shell [km³]."""
    r_low = RE + h_low_km
    r_high = RE + h_high_km
    return (4.0 / 3.0) * np.pi * (r_high**3 - r_low**3)

SHELL_VOLUMES_KM3 = np.array(
    [_shell_volume_km3(lo, hi) for lo, hi in SHELL_BOUNDS_KM], dtype=float
)


# ---------------------------------------------------------------------------
# Atmospheric lifetime model
# ---------------------------------------------------------------------------

# Reference lifetimes [yr] at F10.7 = 150 sfu (moderate solar activity),
# calibrated roughly against published LEO debris decay data:
#   550 km → ~5 yr,  650 km → ~20 yr,  750 km → ~100 yr,  850 km → ~400 yr
TAU_D_REF = np.array([5.0, 20.0, 100.0, 400.0], dtype=float)
TAU_D_F10_REF = 150.0  # sfu reference point

# Drag exponent per shell: lower shells are more sensitive to solar activity.
# Calibrated from Sagnieres & Sharf (2017) uncertainty ranges.
DRAG_EXPONENTS = np.array([2.0, 1.8, 1.5, 1.2], dtype=float)


def debris_lifetime(F10_7: float) -> np.ndarray:
    """Per-shell debris atmospheric lifetime [yr] as a function of F10.7.

    τ_D_i(F10.7) = τ_D_ref_i × (F10.7_ref / F10.7)^exponent_i
    """
    return TAU_D_REF * (TAU_D_F10_REF / F10_7) ** DRAG_EXPONENTS


# ---------------------------------------------------------------------------
# Intact-object lifetime (depends on PMD, not directly on F10.7)
# ---------------------------------------------------------------------------

# Natural long-term lifetime per shell (objects that skip PMD).
# Higher shells have far longer natural lifetimes.
TAU_S_NATURAL = np.array([50.0, 150.0, 500.0, 2000.0], dtype=float)
TAU_S_PMD = 25.0  # years — the standard 25-yr disposal rule


def intact_lifetime(pmd_compliance: float) -> np.ndarray:
    """Effective per-shell intact-object lifetime [yr] given PMD compliance.

    Blend of the 25-yr PMD rule (fraction pmd) and natural decay (1-pmd).
    """
    rate_pmd = pmd_compliance / TAU_S_PMD
    rate_nat = (1.0 - pmd_compliance) / TAU_S_NATURAL
    return 1.0 / (rate_pmd + rate_nat)


# ---------------------------------------------------------------------------
# Collision rate model
# ---------------------------------------------------------------------------

# Collision cross-sections [km²] calibrated to produce ~0.10–0.15 catastrophic
# collisions per year across all shells at baseline (2010-era) populations.
# intact–intact: large effective area (~5 m²) but spatially sparse population.
# intact–debris: intermediate; debris is abundant, intact objects are larger.
# debris–debris: small cross-section (~0.02 m²) but large population squared.
SIGMA_SS = 5.0e-6   # intact–intact [km²]
SIGMA_SD = 3.0e-6   # intact–debris [km²]
SIGMA_DD = 2.0e-8   # debris–debris [km²]

# Mean relative velocity [km/s] in LEO
V_REL_KM_S = 10.0


def collision_rate(N_i: float, N_j: float, sigma: float, V_shell_km3: float) -> float:
    """Expected collisions per year in a shell.

    Uses the kinetic-gas analogy: rate = n_i * n_j * σ * v_rel * V_shell
    where n = N/V is number density.

    Simplified form: rate = σ * v_rel * N_i * N_j / V_shell  [yr⁻¹]
    (factor of 3.156×10⁷ s/yr applied implicitly via pre-scaling).
    """
    SEC_PER_YR = 3.156e7
    return sigma * V_REL_KM_S * SEC_PER_YR * N_i * N_j / V_shell_km3


# ---------------------------------------------------------------------------
# Fragment redistribution weights
# ---------------------------------------------------------------------------

# After a collision in shell i, fragments are redistributed across all shells.
# Weights are based on a simplified ΔV Gaussian spread (Anz-Meador 2010).
# Most fragments stay near the collision shell; a fraction spreads to adjacent.
FRAG_REDISTRIB = np.array(
    [
        [0.50, 0.30, 0.15, 0.05],  # collision in shell 0
        [0.20, 0.50, 0.25, 0.05],  # collision in shell 1
        [0.05, 0.20, 0.55, 0.20],  # collision in shell 2
        [0.05, 0.10, 0.30, 0.55],  # collision in shell 3
    ]
)


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

class MultiShellModel:
    """Multi-shell stochastic debris evolution model.

    Parameters
    ----------
    launch_rate : float
        Total annual launch rate [objects/yr].
    launch_distribution : array-like of length 4
        Fraction of launches into each shell (must sum to 1).
    pmd_compliance : float
        PMD compliance fraction [0, 1].
    F10_7 : float
        Solar flux index F10.7 [sfu].
    k_fragments : float
        Mean trackable fragments per catastrophic collision.
    S0 : array-like of length 4
        Initial intact-object population per shell.
    D0 : array-like of length 4
        Initial debris population per shell.
    dt : float
        Time step [yr].
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(
        self,
        launch_rate: float = 200.0,
        launch_distribution: list[float] | None = None,
        pmd_compliance: float = 0.90,
        F10_7: float = 150.0,
        k_fragments: float = 250.0,
        S0: list[float] | None = None,
        D0: list[float] | None = None,
        dt: float = 1.0,
        seed: int | None = 42,
    ):
        self.launch_rate = launch_rate
        self.launch_distribution = (
            np.array(launch_distribution, dtype=float)
            if launch_distribution is not None
            else np.array([0.15, 0.35, 0.35, 0.15], dtype=float)
        )
        self.pmd_compliance = pmd_compliance
        self.F10_7 = F10_7
        self.k_fragments = k_fragments
        self.S0 = (
            np.array(S0, dtype=float)
            if S0 is not None
            else np.array([300.0, 600.0, 700.0, 400.0], dtype=float)
        )
        self.D0 = (
            np.array(D0, dtype=float)
            if D0 is not None
            else np.array([1500.0, 3500.0, 3000.0, 2000.0], dtype=float)
        )
        self.dt = dt
        self.rng = default_rng(seed)

    # ------------------------------------------------------------------
    # Single run
    # ------------------------------------------------------------------

    def run(self, years: int = 200) -> dict:
        """Run one stochastic realisation of the multi-shell model.

        Returns
        -------
        dict with keys:
            't'       — time array [yr]
            'S'       — intact populations shape (steps+1, N_SHELLS)
            'D'       — debris populations shape (steps+1, N_SHELLS)
            'N'       — total (S+D) per shell
            'N_total' — sum over all shells
            'R_col'   — total collision events per year
        """
        steps = int(years / self.dt)
        t = np.arange(steps + 1, dtype=float) * self.dt

        S = np.zeros((steps + 1, N_SHELLS))
        D = np.zeros((steps + 1, N_SHELLS))
        R_col_ts = np.zeros(steps + 1)

        S[0] = self.S0.copy()
        D[0] = self.D0.copy()

        tau_D = debris_lifetime(self.F10_7)
        tau_S = intact_lifetime(self.pmd_compliance)
        dt = self.dt
        pmd = self.pmd_compliance
        lam_per_shell = self.launch_rate * self.launch_distribution

        for step in range(steps):
            s = S[step].copy()
            d = D[step].copy()

            delta_S = np.zeros(N_SHELLS)
            delta_D = np.zeros(N_SHELLS)
            total_collisions = 0

            for i in range(N_SHELLS):
                V = SHELL_VOLUMES_KM3[i]

                # -- Launches --
                delta_S[i] += lam_per_shell[i] * dt
                delta_D[i] += (1.0 - pmd) * lam_per_shell[i] * dt

                # -- Intact-object decay (PMD + natural) --
                delta_S[i] -= (s[i] / tau_S[i]) * dt

                # -- Debris atmospheric decay --
                delta_D[i] -= (d[i] / tau_D[i]) * dt

                # -- Collisions: intact–intact --
                lam_SS = collision_rate(s[i], s[i], SIGMA_SS, V) * dt
                n_SS = self.rng.poisson(max(lam_SS, 0.0))

                # -- Collisions: intact–debris --
                lam_SD = collision_rate(s[i], d[i], SIGMA_SD, V) * dt
                n_SD = self.rng.poisson(max(lam_SD, 0.0))

                # -- Collisions: debris–debris (fragment-producing) --
                lam_DD = collision_rate(d[i], d[i], SIGMA_DD, V) * dt
                n_DD = self.rng.poisson(max(lam_DD, 0.0))

                n_total_col = n_SS + n_SD + n_DD
                total_collisions += n_total_col

                # Remove colliding intact objects
                destroyed = min(2 * n_SS + n_SD, s[i])
                delta_S[i] -= destroyed

                # Generate and redistribute fragments
                new_frags = self.k_fragments * n_total_col
                for j in range(N_SHELLS):
                    delta_D[j] += new_frags * FRAG_REDISTRIB[i, j]

            R_col_ts[step] = total_collisions / dt

            S[step + 1] = np.maximum(s + delta_S, 0.0)
            D[step + 1] = np.maximum(d + delta_D, 0.0)

        R_col_ts[steps] = 0.0  # last time point not computed in loop

        return {
            "t": t,
            "S": S,
            "D": D,
            "N": S + D,
            "N_total": (S + D).sum(axis=1),
            "R_col": R_col_ts,
        }

    # ------------------------------------------------------------------
    # Ensemble run (Monte Carlo)
    # ------------------------------------------------------------------

    def run_ensemble(self, years: int = 200, n_runs: int = 50) -> dict:
        """Run multiple stochastic realisations and return statistics.

        Parameters
        ----------
        years : int
            Simulation duration [yr].
        n_runs : int
            Number of independent realisations.

        Returns
        -------
        dict with keys:
            't'            — time array [yr]
            'N_total_mean' — ensemble mean total population
            'N_total_std'  — ensemble standard deviation
            'N_total_all'  — all runs, shape (n_runs, steps+1)
            'R_col_mean'   — ensemble mean collision rate
        """
        steps = int(years / self.dt)
        all_N = np.zeros((n_runs, steps + 1))
        all_R = np.zeros((n_runs, steps + 1))
        t = None

        for run_idx in range(n_runs):
            # Use a different seed per run while remaining reproducible
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
