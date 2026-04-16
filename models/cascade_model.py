"""
Population-dynamics cascade model (Kessler-type framework).

Tracks three aggregate populations over LEO as a whole:
  S(t)   — intact objects (active + derelict spacecraft, rocket bodies)
  D(t)   — large debris fragments (>=10 cm, radar-trackable)
  D_s(t) — small debris fragments (1 mm–10 cm, untraceable)

Small debris (D_s) is the dominant population by count (~500,000 objects
>1 cm vs ~20,000 objects >10 cm) and poses the primary collision risk to
operational satellites because it cannot be tracked or avoided.  It also
has a much higher area-to-mass ratio (~1 m²/kg vs ~0.05 m²/kg for large
fragments), making it significantly more sensitive to atmospheric drag and
therefore to solar flux F10.7.

Governing discrete-time equations (dt = 1 year):

  S(t+1)   = S(t)   + Lambda
                     - S/tau_S
                     - (R_SS + R_SD + R_SDs) * dt

  D(t+1)   = D(t)   + k_large * (R_SS + R_SD + R_DD + R_SDs) * dt
                     - D/tau_D(F10.7) * dt
                     + (1-PMD) * Lambda * dt

  D_s(t+1) = D_s(t) + k_small * (R_SS + R_SD + R_DD + R_SDs) * dt
                     - D_s/tau_Ds(F10.7) * dt

Collision terms:
  R_SS  — intact–intact   (produces large + small fragments)
  R_SD  — intact–large    (produces large + small fragments)
  R_DD  — large–large     (produces large + small fragments)
  R_SDs — intact–small    (catastrophic at 10 km/s; produces large + small fragments)

Fragment ratios from NASA Standard Breakup Model (Anz-Meador 2010):
  k_large ~  250  trackable fragments (>=10 cm) per catastrophic collision
  k_small ~ 2000  sub-trackable fragments (1 mm–10 cm) per collision

Solar-flux effect on debris lifetimes (Sagnieres & Sharf 2017;
NASA-STD-8719.14C):
  tau_D(F10.7)  = tau_D_ref  * (F10.7_ref / F10.7) ** drag_exponent
  tau_Ds(F10.7) = tau_Ds_ref * (F10.7_ref / F10.7) ** drag_exponent_small

Small fragments use a higher drag exponent because their elevated A/m ratio
amplifies the response to atmospheric density changes driven by solar flux.

References: Kessler & Cour-Palais (1978); Liou (2008);
            Flegel et al. (2010) MASTER-2009.
"""

import numpy as np


class CascadeModel:
    """Deterministic Kessler-type cascade model for LEO debris evolution.

    Parameters
    ----------
    launch_rate : float
        Mean annual launch rate [intact objects / yr].
    pmd_compliance : float
        Post-mission disposal compliance fraction [0, 1].
    F10_7 : float
        Solar flux proxy F10.7 index [sfu].  Typical range 70-230.
    tau_S_base : float
        Intact-object PMD disposal lifetime [yr] (25-yr rule).
    tau_D_ref : float
        Large-debris atmospheric lifetime [yr] at F10.7 = F10.7_ref.
    tau_Ds_ref : float
        Small-debris atmospheric lifetime [yr] at F10.7 = F10.7_ref.
        Approximately tau_D_ref / 10 due to higher area-to-mass ratio.
    F10_7_ref : float
        Reference solar flux for lifetime calibration [sfu].
    drag_exponent : float
        Power-law exponent for large-debris lifetime vs F10.7.
    drag_exponent_small : float
        Power-law exponent for small-debris lifetime vs F10.7.
        Higher than drag_exponent because small fragments are more
        sensitive to density changes (higher A/m ratio).
    alpha_SS, alpha_SD, alpha_DD : float
        Pairwise collision coefficients for large-object interactions.
    alpha_SDs : float
        Intact-small-debris collision coefficient.
    k_fragments : float
        Large (>=10 cm) fragments per catastrophic collision.
    k_small_frags : float
        Small (1 mm–10 cm) fragments per catastrophic collision.
    S0, D0, Ds0 : float
        Initial populations: intact, large debris, small debris.
    dt : float
        Time step [yr].
    """

    def __init__(
        self,
        launch_rate: float = 200.0,
        pmd_compliance: float = 0.25,
        F10_7: float = 150.0,
        tau_S_base: float = 25.0,
        tau_D_ref: float = 40.0,
        tau_Ds_ref: float = 4.0,
        F10_7_ref: float = 150.0,
        drag_exponent: float = 1.5,
        drag_exponent_small: float = 1.8,
        alpha_SS: float = 2.0e-9,
        alpha_SD: float = 5.0e-9,
        alpha_DD: float = 2.0e-11,
        alpha_SDs: float = 1.0e-10,
        k_fragments: float = 250.0,
        k_small_frags: float = 2000.0,
        S0: float = 2000.0,
        D0: float = 10000.0,
        Ds0: float = 250000.0,
        dt: float = 1.0,
    ):
        self.launch_rate = launch_rate
        self.pmd_compliance = pmd_compliance
        self.F10_7 = F10_7
        self.tau_S_base = tau_S_base
        self.tau_D_ref = tau_D_ref
        self.tau_Ds_ref = tau_Ds_ref
        self.F10_7_ref = F10_7_ref
        self.drag_exponent = drag_exponent
        self.drag_exponent_small = drag_exponent_small
        self.alpha_SS = alpha_SS
        self.alpha_SD = alpha_SD
        self.alpha_DD = alpha_DD
        self.alpha_SDs = alpha_SDs
        self.k_fragments = k_fragments
        self.k_small_frags = k_small_frags
        self.S0 = S0
        self.D0 = D0
        self.Ds0 = Ds0
        self.dt = dt

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------

    def tau_D(self) -> float:
        """Large-debris atmospheric lifetime [yr] as a function of F10.7."""
        return self.tau_D_ref * (self.F10_7_ref / self.F10_7) ** self.drag_exponent

    def tau_Ds(self) -> float:
        """Small-debris atmospheric lifetime [yr] as a function of F10.7.

        Shorter than tau_D due to higher area-to-mass ratio; more sensitive
        to solar-flux-driven density changes (higher drag_exponent_small).
        """
        return self.tau_Ds_ref * (self.F10_7_ref / self.F10_7) ** self.drag_exponent_small

    def tau_S_eff(self) -> float:
        """Effective intact-object lifetime accounting for PMD compliance."""
        tau_natural = 200.0
        rate = self.pmd_compliance / self.tau_S_base + (1 - self.pmd_compliance) / tau_natural
        return 1.0 / rate

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def run(self, years: int = 200) -> dict:
        """Integrate the three-population cascade model forward in time.

        Returns
        -------
        dict with keys:
            't'      — time array [yr]
            'S'      — intact-object population
            'D'      — large debris (>=10 cm) population
            'Ds'     — small debris (1 mm-10 cm) population
            'N'      — total population (S + D + Ds)
            'N_large'— S + D only (for comparison with >=10 cm-only models)
            'R_col'  — collision rate (collisions/yr)
        """
        steps = int(years / self.dt)
        t = np.arange(steps + 1, dtype=float) * self.dt

        S = np.zeros(steps + 1)
        D = np.zeros(steps + 1)
        Ds = np.zeros(steps + 1)
        R_col_ts = np.zeros(steps + 1)

        S[0] = self.S0
        D[0] = self.D0
        Ds[0] = self.Ds0

        tau_D_val = self.tau_D()
        tau_Ds_val = self.tau_Ds()
        tau_S_val = self.tau_S_eff()
        lam = self.launch_rate
        pmd = self.pmd_compliance
        dt = self.dt

        for i in range(steps):
            s = S[i]
            d = D[i]
            ds = Ds[i]

            # Collision rates [collisions / yr]
            r_SS = self.alpha_SS * s * s
            r_SD = self.alpha_SD * s * d
            r_DD = self.alpha_DD * d * d
            r_SDs = self.alpha_SDs * s * ds   # intact hit by small debris

            r_total = r_SS + r_SD + r_DD + r_SDs
            R_col_ts[i] = r_total

            # Intact-object update
            S[i + 1] = max(
                s + lam * dt
                - (s / tau_S_val) * dt
                - (r_SS + r_SD + r_SDs) * dt,
                0.0,
            )

            # Large debris update
            D[i + 1] = max(
                d
                + self.k_fragments * r_total * dt
                - (d / tau_D_val) * dt
                + (1 - pmd) * lam * dt,
                0.0,
            )

            # Small debris update
            Ds[i + 1] = max(
                ds
                + self.k_small_frags * r_total * dt
                - (ds / tau_Ds_val) * dt,
                0.0,
            )

        R_col_ts[steps] = (
            self.alpha_SS * S[steps] ** 2
            + self.alpha_SD * S[steps] * D[steps]
            + self.alpha_DD * D[steps] ** 2
            + self.alpha_SDs * S[steps] * Ds[steps]
        )

        return {
            "t": t,
            "S": S,
            "D": D,
            "Ds": Ds,
            "N": S + D + Ds,
            "N_large": S + D,
            "R_col": R_col_ts,
        }
