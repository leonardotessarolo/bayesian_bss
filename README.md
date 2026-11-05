# bayesian_bss

Bayesian blind source separation experiments (2×2 linear instantaneous mixtures).

## Structure

```
bayesian_bss/
├── src/                     # Source code (entry points: run_discrete.py, run_continuous.py)
├── discrete_experiment/     # Discrete experiment notebooks; results in output/
├── continuous_experiment/   # Continuous experiment notebooks; results in output/
├── legacy/                  # Old code (not used by the experiments)
├── run_discrete.sh          # Launcher for the discrete experiment
├── run_continuous.sh        # Launcher for the continuous experiment
└── requirements.txt
```

## Setup

Requires Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x run_*.sh
```

## Running

Set the arguments at the top of each launcher, then run it from anywhere:

```bash
./run_discrete.sh     # -> discrete_experiment/output/<EXPERIMENT_NAME>/
./run_continuous.sh   # -> continuous_experiment/output/<EXPERIMENT_NAME>/
```

| Variable | Meaning |
|---|---|
| `EXPERIMENT_NAME` | Output folder name |
| `N_REALIZATIONS` | Number of realizations |
| `SEED` | Master seed |
| `FIRST_RUN` | `--first-run` for a new experiment, `--no-first-run` to resume |
| `MODE` (discrete only) | `broad` or `narrow` |

- **Discrete `narrow` mode** needs `discrete_experiment/output/CBA_SBRT_broad_analysis/gmm_means_starting_points.pkl`, produced by a `broad` run plus the clustering notebook.
- **Resuming** (`--no-first-run`) skips realizations that already finished.
- Each run saves its environment to `requirements.lock.txt` inside the output folder. To reproduce a run exactly, install from that file.
