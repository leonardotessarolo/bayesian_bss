#!/usr/bin/env bash
set -euo pipefail

# Save as bayesian_bss/run_continuous.sh (repo root).
PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# Every relative path (CLI and pickled configs) is resolved from the repo root.
cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# Edit the arguments below. To resume, set FIRST_RUN=--no-first-run.
# Run as a module to avoid src/logging.py shadowing standard-library logging.
EXPERIMENT_NAME="continuous_exp_150"
OUTPUT_DIR="./continuous_experiment/output"
N_REALIZATIONS=150
SEED=100
FIRST_RUN="--first-run"
export JAX_ENABLE_X64=True
export JAX_PLATFORMS=cpu
export PYTHONHASHSEED=0
mkdir -p "${OUTPUT_DIR}/${EXPERIMENT_NAME}"
"$PYTHON_BIN" -m pip freeze > "${OUTPUT_DIR}/${EXPERIMENT_NAME}/requirements.lock.txt"
exec "$PYTHON_BIN" -u -m src.run_continuous \
    --experiment-name "$EXPERIMENT_NAME" \
    --n-realizations "$N_REALIZATIONS" \
    --output-dir "$OUTPUT_DIR" \
    --seed "$SEED" \
    $FIRST_RUN \
    --create-folder-structure