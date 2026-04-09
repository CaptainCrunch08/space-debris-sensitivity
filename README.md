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
| `MultiShellModel` | Multi-shell stochastic (MOCAT-inspired) | Rossi et al. (2009); Radtke et al. (2017) |

## Project structure

```
models/
  cascade_model.py      Kessler-type two-population cascade model
  multishell_model.py   Four-shell stochastic evolution model
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

| Parameter | Range | Units |
|---|---|---|
| F10.7 (solar flux) | 70 – 230 | sfu |
| Launch rate | 100 – 400 | objects/yr |
| PMD compliance | 0.50 – 0.99 | fraction |

## Dependencies

See `requirements.txt`.  Install with:
```powershell
.\.venv\Scripts\pip install -r requirements.txt
```
