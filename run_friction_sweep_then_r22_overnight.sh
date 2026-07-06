#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR=/private/tmp
export XDG_CACHE_HOME=/private/tmp
export KMP_DUPLICATE_LIB_OK=TRUE

echo "=== Overnight sequence started: $(date) ==="
echo "Working directory: $(pwd)"

echo
echo "=== 1/2 friction stopping v7/v8/v9 sweep: $(date) ==="
bash ./in_env python run_friction_stopping_rma_v7_v8_v9.py

echo
echo "=== 2/2 R22 discrete calibrated uncertainty: $(date) ==="
bash ./in_env python run_r22_discrete_calibrated_uncertainty.py

echo
echo "=== Overnight sequence finished: $(date) ==="
