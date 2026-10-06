#!/usr/bin/env bash
# End-to-end Task 1 pipeline. Usage:  ./run_task1.sh  [--skip-ingest]
set -euo pipefail

ENV_PREFIX="${CONDA_PREFIX_DSA5208:-/opt/homebrew/Caskroom/miniforge/base/envs/dsa5208}"
export JAVA_HOME="$ENV_PREFIX/lib/jvm"
export SPARK_LOCAL_CORES="${SPARK_LOCAL_CORES:-12}"
PY="$ENV_PREFIX/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p logs outputs/figures outputs/tables

if [[ "${1:-}" != "--skip-ingest" ]]; then
  echo "[1/3] ingest + clean  (~4 min, 8 GB CSV -> 342 MB Parquet)"
  "$PY" src/01_ingest.py 2>&1 | tee logs/01_ingest.log | grep -vE '^[0-9]{2}/'
else
  echo "[1/3] ingest skipped"
fi

echo "[2/3] Task 1 analysis  (~1 min, writes outputs/tables/*.csv)"
"$PY" src/02_task1_analysis.py 2>&1 | tee logs/02_task1_analysis.log | grep -vE '^[0-9]{2}/'

echo "[3/3] figures          (writes outputs/figures/*.png and *.pdf)"
"$PY" src/03_task1_figures.py 2>&1 | grep -v findfont

echo "done."
