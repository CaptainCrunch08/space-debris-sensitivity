# Space Debris Population Evolution — Sensitivity Analysis

Cross-model sensitivity analysis comparing two classes of LEO debris
evolution models on their response to solar flux (F10.7).

## Research question

Do different modeling frameworks (deterministic cascade vs. multi-shell
stochastic) place different weights on the same environmental parameter
(solar flux / atmospheric drag)?

## Models implemented

| Model | Framework | Key reference |
|---|---|---|
| `CascadeModel` | Deterministic Kessler-type population ODE | Kessler & Cour-Palais (1978); Liou (2008) |
| `MultiShellModel` | Five-shell stochastic (MOCAT-inspired), 500–1000 km | Rossi et al. (2009); Radtke et al. (2017) |

Both models track three populations:
- **S** — intact objects (active + derelict spacecraft, rocket bodies)
- **D** — large debris fragments (>=10 cm, radar-trackable)
- **Ds** — small debris fragments (1 mm–10 cm, untraceable but hazardous)

## Project structure

```
models/
  cascade_model.py      Kessler-type three-population cascade model
  multishell_model.py   Five-shell stochastic evolution model
analysis/
  sensitivity_analysis.py   Parameter sweep + Sobol variance decomposition
  plotting.py               All figures
notebooks/
  01_model_exploration.ipynb   Interactive walkthrough
run_analysis.py          Main entry point
results/                 Output figures and data (auto-created)
```

## Quick start

```powershell
# Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# Run a quick test (50-yr horizon, reduced samples)
python run_analysis.py --quick

# Run the full analysis
python run_analysis.py
```

## Sensitivity parameters

| Parameter | Baseline | Sobol range | Units | Source |
|---|---|---|---|---|
| F10.7 (solar flux) | 150 | 70 – 230 | sfu | NOAA/NGDC long-record solar flux data |
| Launch rate | 200 | 100 – 600 | objects/yr | Zhang et al. (2022); pre-constellation to early deployment |
| PMD compliance | 0.25 | 0.20 – 0.99 | fraction | NASA OIG IG-21-011 (2021); Liou & Krisko (2013) |

### PMD compliance note
The baseline of 0.25 represents the midpoint of the 20–30% global historical
compliance range documented in NASA OIG report IG-21-011 (January 2021):
> "the global compliance rate has only averaged between 20 to 30 percent —
> much lower than the 90 percent required to slow the rate at which debris
> is generated in LEO."

The Sobol upper bound of 0.99 spans to near-full compliance, matching the
aspirational scenario tested by Liou & Krisko (2013), who showed that even
95% compliance does not fully stabilize the LEO environment.

### Launch rate note
The baseline of 200 objects/yr reflects pre-constellation (2010-era) traffic,
consistent with the initial population calibration. The upper bound of 600/yr
captures early constellation deployment scenarios. Full mega-constellation
deployment rates (>1,000/yr per Zhang et al. 2022) exceed the scope of these
simplified models and are a direction for future work.

## Dependencies

See `requirements.txt`.  Install with:
```powershell
.\.venv\Scripts\pip install -r requirements.txt
```
