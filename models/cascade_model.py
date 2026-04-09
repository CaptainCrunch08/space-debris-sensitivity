"""
Population-dynamics cascade model (Kessler-type framework).

Tracks two aggregate populations over LEO as a whole:
  S(t)  — intact objects (active + derelict spacecraft, rocket bodies)
  D(t)  — hazardous debris fragments (≥10 cm)

Governing discrete-time equations (Δt = 1 year):

  S(t+1) = S(t) + Λ  - S(t)/τ_S  - R_SS(t) - R_SD(t)
  D(t+1) = D(t) + k·[R_SS(t) + R_SD(t) + R_DD(t)]
                - D(t)/τ_D(F10.7)
                + (1 - PMD)·Λ

Where:
  Λ         — annual launch rate [objects/yr]
  τ_S       — effective lifetime of intact objects [yr] (tied to PMD)
  τ_D(F107) — debris lifetime driven by atmospheric drag; decreases
               as solar flux F10.7 increases (more drag)
  R_ij      — pairwise collision rates between populations i and j
  k         — mean fragments produced per catastrophic collision
  PMD       — post-mission disposal compliance fraction [0,1]

Solar-flux effect on debris lifetime (after Sagnieres & Sharf 2017;
NASA-STD-8719.14C):
  τ_D(F10.7) = τ_D_ref * (F10.7_ref / F10.7) ** drag_exponent

Reference: Kessler & Cour-Palais (1978); Liou (2008).
"""

import numpy as np


class CascadeModel:
    """Deterministic Kessler-type cascade model for LEO debris evolution.

    Parameters
    ----------
    launch_rate : float
        Mean annual launch rate [intact objects / yr].  Default ≈ 200/yr,
        roughly consistent with 2010-era traffic before mega-constellations.
    pmd_compliance : float
        Post-mission disposal compliance fraction [0, 1].
    F10_7 : float
        Solar flux proxy F10.7 index [sfu].  Typical range 70–230.
    tau_S_base : float
        Base lifetime of intact objects at full PMD compliance [yr].
        At compliance p, the effective decay rate scales accordingly.
    tau_D_ref : float
        Debris atmospheric lifetime [yr] at F10.7 = F10.7_ref.
    F10_7_ref : float
        Reference solar flux for tau_D calibration [sfu].
    drag_exponent : float
        Power-law exponent linking F10.7 to debris lifetime.
        τ_D ∝ (F10.7_ref/F10.7)^drag_exponent.
        Calibrated from NRLMSISE-00/JB2008 comparisons (Sagnieres 2017).
    alpha_SS, alpha_SD, alpha_DD : float
        Pairwise collision coefficients [yr^-1 per object^2], encoding
        spatial density, cross-section, and relative velocity.
    k_fragments : float
        Mean number of trackable fragments (≥10 cm) per catastrophic
        collision.  Informed by NASA Standard Breakup Model and Anz-Meador
        (2010) analysis of the Iridium 33 / Cosmos 2251 event.
    S0, D0 : float
        Initial intact-object and debris populations.
    dt : float
        Time step [yr].
    """

    def __init__(
        self,
        launch_rate: float = 200.0,
        pmd_compliance: float = 0.90,
        F10_7: float = 150.0,
        tau_S_base: float = 25.0,
        tau_D_ref: float = 40.0,
        F10_7_ref: float = 150.0,
        drag_exponent: float = 1.5,
        alpha_SS: float = 2.0e-9,
        alpha_SD: float = 5.0e-9,
        alpha_DD: float = 2.0e-11,
        k_fragments: float = 250.0,
        S0: float = 2000.0,
        D0: float = 10000.0,
        dt: float = 1.0,
    ):
        self.launch_rate = launch_rate
        self.pmd_compliance = pmd_compliance
        self.F10_7 = F10_7
        self.tau_S_base = tau_S_base
        self.tau_D_ref = tau_D_ref
        self.F10_7_ref = F10_7_ref
        self.drag_exponent = drag_exponent
        self.alpha_SS = alpha_SS
        self.alpha_SD = alpha_SD
        self.alpha_DD = alpha_DD
        self.k_fragments = k_fragments
        self.S0 = S0
        self.D0 = D0
        self.dt = dt

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def tau_D(self) -> float:
        """Debris atmospheric lifetime [yr] as a function of F10.7.

        Higher solar flux → denser upper atmosphere → shorter lifetime.
        Model: τ_D = τ_D_ref * (F10.7_ref / F10.7)^drag_exponent
        """
        return self.tau_D_ref * (self.F10_7_ref / self.F10_7) ** self.drag_exponent

    def tau_S_eff(self) -> float:
        """Effective intact-object lifetime accounting for PMD compliance.

        PMD-compliant objects deorbit within tau_S_base years (25-yr rule).
        Non-compliant objects persist for a much longer natural lifetime
        (modeled here as 200 yr as a conservative upper bound).
        The effective mean lifetime is the harmonic blend:
          1/τ_S_eff = PMD/tau_S_base + (1-PMD)/tau_natural
        """
        tau_natural = 200.0
        rate = self.pmd_compliance / self.tau_S_base + (1 - self.pmd_compliance) / tau_natural
        return 1.0 / rate

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self, years: int = 200) -> dict:
        """Integrate the cascade model forward in time.

        Parameters
        ----------
        years : int
            Simulation duration [yr].

        Returns
        -------
        dict with keys:
            't'     — time array [yr]
            'S'     — intact-object population over time
            'D'     — debris population over time
            'N'     — total population (S + D)
            'R_col' — cumulative collision rate (collisions/yr) over time
        """
        steps = int(years / self.dt)
        t = np.arange(steps + 1, dtype=float) * self.dt

        S = np.zeros(steps + 1)
        D = np.zeros(steps + 1)
        R_col_ts = np.zeros(steps + 1)

        S[0] = self.S0
        D[0] = self.D0

        tau_D_val = self.tau_D()
        tau_S_val = self.tau_S_eff()
        lam = self.launch_rate
        pmd = self.pmd_compliance
        dt = self.dt

        for i in range(steps):
            s = S[i]
            d = D[i]

            # Pairwise collision rates [collisions / yr]
            r_SS = self.alpha_SS * s * s
            r_SD = self.alpha_SD * s * d
            r_DD = self.alpha_DD * d * d

            r_total = r_SS + r_SD + r_DD
            R_col_ts[i] = r_total

            # Intact-object update
            # Gains: launches
            # Losses: PMD-driven decay + collision removal
            S[i + 1] = (
                s
                + lam * dt
                - (s / tau_S_val) * dt
                - (r_SS + r_SD) * dt
            )
            S[i + 1] = max(S[i + 1], 0.0)

            # Debris update
            # Gains: fragments from all collision types + non-compliant launches
            # Losses: atmospheric drag decay
            D[i + 1] = (
                d
                + self.k_fragments * r_total * dt
                - (d / tau_D_val) * dt
                + (1 - pmd) * lam * dt
            )
            D[i + 1] = max(D[i + 1], 0.0)

        R_col_ts[steps] = (
            self.alpha_SS * S[steps] ** 2
            + self.alpha_SD * S[steps] * D[steps]
            + self.alpha_DD * D[steps] ** 2
        )

        return {
            "t": t,
            "S": S,
            "D": D,
            "N": S + D,
            "R_col": R_col_ts,
        }
