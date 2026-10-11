#!/usr/bin/env bash
# End-to-end Task 2 pipeline (needs data_parquet/ from stage 1 of ./run_task1.sh).
# Usage:  ./run_task2.sh  [--skip-features] [--refit]
#   --skip-features  re-use data_parquet/task2_features (and its eval/train sets)
#   --refit          ignore cached candidate models in models/_candidates/
set -euo pipefail

ENV_PREFIX="${CONDA_PREFIX_DSA5208:-/opt/homebrew/Caskroom/miniforge/base/envs/dsa5208}"
export JAVA_HOME="$ENV_PREFIX/lib/jvm"
export SPARK_LOCAL_CORES="${SPARK_LOCAL_CORES:-12}"
export PYTHONUNBUFFERED=1
PY="$ENV_PREFIX/bin/python"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
mkdir -p logs outputs/figures outputs/tables models

SKIP=0; REFIT=""
for a in "$@"; do
  case "$a" in
    --skip-features) SKIP=1 ;;
    --refit) REFIT="--refit" ;;
  esac
done

if [[ ! -d data_parquet/readings ]]; then
  echo "data_parquet/ not found - run ./run_task1.sh first (stage 1 builds it)"; exit 1
fi

if [[ $SKIP -eq 0 ]]; then
  echo "[1/4] feature table: forward target + trailing/neighbour features  (~5 min)"
  "$PY" src/04_task2_features.py 2>&1 | tee logs/04_task2_features.log | grep -vE '^[0-9]{2}/|^\s+at |WARN'
  rm -rf data_parquet/task2_sets          # sets are derived from the features: rebuild
else
  echo "[1/4] feature table skipped"
fi

echo "[2/4] leakage / label verification against an independent pandas recomputation  (~1 min)"
"$PY" src/verify_task2_features.py 2>&1 | tee logs/verify_task2_features.log | grep -vE '^[0-9]{2}/|^\s+at |WARN|Setting default|To adjust'

echo "[3/4] train, select on 2023, test once on 2024, save model  (see README for runtime)"
"$PY" src/05_task2_train.py $REFIT 2>&1 | tee logs/05_task2_train.log | grep -vE '^[0-9]{2}/|^\s+at |WARN|Arrow|Cannot convert|Attempting|warn\('

echo "[4/4] Task 2 figures"
"$PY" src/06_task2_figures.py 2>&1 | grep -v findfont

echo "done.  Try:  $PY src/predict_task2.py --station S77 --time \"2024-10-01 15:00:00\" --threshold 5"
