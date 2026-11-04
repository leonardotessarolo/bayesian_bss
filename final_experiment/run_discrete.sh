#!/usr/bin/env bash
set -euo pipefail

# Save as bayesian_bss/final_experiment/run_discrete.sh.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Resolve relative output paths from final_experiment.
cd "$SCRIPT_DIR"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Edit the arguments below. To resume, set FIRST_RUN=--no-first-run.
# Run as a module to avoid src/logging.py shadowing standard-library logging.
EXPERIMENT_NAME="my_discrete_experiment"
N_REALIZATIONS=2
MODE="narrow"
OUTPUT_DIR="./output"
SEED=100
FIRST_RUN="--first-run"

export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export PYTHONHASHSEED=0

mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"
"$PYTHON_BIN" -m pip freeze > "${OUTPUT_DIR}/${EXPERIMENT_NAME}/requirements.lock.txt"

exec "$PYTHON_BIN" -u -m src.run_discrete \
    --experiment-name "$EXPERIMENT_NAME" \
    --mode "$MODE" \
    --n-realizations "$N_REALIZATIONS" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    $FIRST_RUN \
    --create-folder-structure