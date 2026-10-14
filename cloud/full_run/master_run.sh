#!/usr/bin/env bash
# Runs ON the Dataproc master node: executes the full notebook on the cluster and uploads the results.
# Started from Cloud Shell (see run_full_pipeline_dataproc.sh):  bash master_run.sh gs://<bucket> "<cluster description>"
set -uo pipefail
BUCKET="$1"; CLUSTER_DESC="${2:-Dataproc cluster}"
WORK=/tmp/dsa5208                      # same absolute path locally and on HDFS
LOG=$WORK/run.log
mkdir -p $WORK/data $WORK/reference && cd $WORK
stamp() { echo "[$(date '+%F %T')] $*" | tee -a $WORK/progress.log; }

stamp "copy notebook, coastline file and the eight CSVs from $BUCKET"
gsutil -q cp "$BUCKET/code/full_run/DSA5208_Project2_dataproc.ipynb" .
gsutil -q cp "$BUCKET/code/full_run/singapore_boundary.geojson" reference/
gsutil -m -q cp "$BUCKET/data/*.csv" data/
stamp "put the CSVs on HDFS at $WORK/data"
hdfs dfs -mkdir -p $WORK/data && hdfs dfs -put -f data/*.csv $WORK/data/
hdfs dfs -ls $WORK/data | tail -8 | tee -a $WORK/progress.log

stamp "Python environment for the notebook driver (same library versions as the laptop run)"
/opt/conda/default/bin/python -m venv /tmp/dsa-venv
/tmp/dsa-venv/bin/pip install -q --upgrade pip
/tmp/dsa-venv/bin/pip install -q pandas==3.0.6 numpy==1.26.4 matplotlib==3.11.2 pyarrow==25.0.0 nbconvert==7.17.1 nbclient ipykernel
/tmp/dsa-venv/bin/python -m ipykernel install --user --name dsa5208 >/dev/null
/tmp/dsa-venv/bin/python -c "import pandas, numpy, matplotlib, pyarrow; print('driver libs', pandas.__version__, numpy.__version__, matplotlib.__version__, pyarrow.__version__)" | tee -a $WORK/progress.log

export SPARK_HOME=/usr/lib/spark
export PYTHONPATH=$SPARK_HOME/python:$(ls $SPARK_HOME/python/lib/py4j-*-src.zip)
export PYSPARK_PYTHON=/opt/conda/default/bin/python      # exists on every node; same Python minor version as the venv
export SPARK_MASTER=yarn SPARK_DRIVER_MEMORY=16g CV_PARALLELISM=4 RERUN_CV=1 CLUSTER_DESC="$CLUSTER_DESC"
stamp "execute the notebook on YARN (log: $LOG)"
/usr/bin/time -p /tmp/dsa-venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=dsa5208 DSA5208_Project2_dataproc.ipynb > $LOG 2>&1
echo "exit $?" >> $LOG
stamp "notebook finished: $(tail -1 $LOG)"

stamp "collect results: outputs/, models (metadata locally, Spark models from HDFS), notebook, logs"
mkdir -p results && cp -r DSA5208_Project2_dataproc.ipynb outputs run.log progress.log results/ 2>/dev/null
mkdir -p results/models && cp -r models/* results/models/ 2>/dev/null
for m in task2_baseline_lr task2_final_rf; do hdfs dfs -copyToLocal -f $WORK/models/$m/* results/models/$m/ 2>/dev/null; done
yarn application -list -appStates FINISHED 2>/dev/null | tail -3 > results/yarn_apps.txt
hdfs dfs -du -s -h $WORK/data_parquet/* > results/hdfs_sizes.txt 2>/dev/null
tar -czf dataproc_full_run.tar.gz -C results .
gsutil -q cp dataproc_full_run.tar.gz "$BUCKET/full_run/"
stamp "uploaded $BUCKET/full_run/dataproc_full_run.tar.gz ($(du -h dataproc_full_run.tar.gz | cut -f1))"
gsutil -q cp $WORK/progress.log "$BUCKET/full_run/progress.log"
echo DONE | gsutil -q cp - "$BUCKET/full_run/DONE"
