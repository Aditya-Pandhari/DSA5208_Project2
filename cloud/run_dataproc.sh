#!/usr/bin/env bash
# DSA5208 Project 2 — run the scalability benchmark on a Google Cloud Dataproc cluster.
# Run the blocks one at a time from the repository root (needs the gcloud CLI: https://cloud.google.com/sdk/docs/install).
# Cost guide (asia-southeast1, Oct 2026 list prices): 1 master + 2 workers n2-standard-4 ~ US$1.0/hour,
# with 4 workers ~ US$1.7/hour. The whole experiment is well under US$5 if the cluster is deleted at the end.
set -euo pipefail

# ---- 0. one-time settings -----------------------------------------------------------------------
export PROJECT_ID="YOUR-PROJECT-ID"              # from the Cloud console (the project the $50 coupon was applied to)
export REGION="asia-southeast1"                  # Singapore
export ZONE="asia-southeast1-b"
export BUCKET="gs://dsa5208-rain-${USER}"        # bucket names are global; change if taken
export CLUSTER="dsa5208-rain"

gcloud auth login
gcloud config set project "$PROJECT_ID"
gcloud services enable dataproc.googleapis.com compute.googleapis.com storage.googleapis.com

# ---- 1. bucket + data (7.5 GB upload; ~10-20 min on a fast connection) --------------------------
gcloud storage buckets create "$BUCKET" --location="$REGION" --uniform-bucket-level-access
gcloud storage cp data/HistoricalRainfallacrossSingapore20*.csv "$BUCKET/data/"
gcloud storage cp cloud/dataproc_benchmark.py "$BUCKET/code/"
gcloud storage ls -l "$BUCKET/data/"              # expect 8 files, ~7.5 GB

# ---- 2. cluster with 2 workers (Spark 3.5 image) --------------------------------------------------
gcloud dataproc clusters create "$CLUSTER" --region="$REGION" --zone="$ZONE" \
  --image-version=2.2-debian12 \
  --master-machine-type=n2-standard-4 --master-boot-disk-size=100 \
  --num-workers=2 --worker-machine-type=n2-standard-4 --worker-boot-disk-size=100 \
  --max-idle=30m                                   # auto-delete after 30 idle minutes (cost protection)

# ---- 3. benchmark on 2 workers (ingest + cleaning job) ------------------------------------------
gcloud dataproc jobs submit pyspark "$BUCKET/code/dataproc_benchmark.py" \
  --cluster="$CLUSTER" --region="$REGION" -- --bucket "$BUCKET" --label "2 workers x n2-standard-4 (8 vCPU)"

# ---- 4. scale out to 4 workers and repeat (bronze is reused) --------------------------------------
gcloud dataproc clusters update "$CLUSTER" --region="$REGION" --num-workers=4
gcloud dataproc jobs submit pyspark "$BUCKET/code/dataproc_benchmark.py" \
  --cluster="$CLUSTER" --region="$REGION" -- --bucket "$BUCKET" --label "4 workers x n2-standard-4 (16 vCPU)" --skip-ingest

# ---- 5. (optional, ~+US$1) 8 workers ---------------------------------------------------------------
# gcloud dataproc clusters update "$CLUSTER" --region="$REGION" --num-workers=8
# gcloud dataproc jobs submit pyspark "$BUCKET/code/dataproc_benchmark.py" \
#   --cluster="$CLUSTER" --region="$REGION" -- --bucket "$BUCKET" --label "8 workers x n2-standard-4 (32 vCPU)" --skip-ingest

# ---- 6. collect the results and DELETE the cluster ------------------------------------------------
mkdir -p outputs/tables/cloud
gcloud storage cp "$BUCKET/results/cloud_scaling/*.csv" outputs/tables/cloud/
gcloud dataproc clusters describe "$CLUSTER" --region="$REGION" --format="yaml(config.masterConfig,config.workerConfig,config.softwareConfig.imageVersion)" > outputs/tables/cloud/cluster_config.yaml
gcloud dataproc clusters delete "$CLUSTER" --region="$REGION" --quiet
# keep the bucket only if you plan to rerun; otherwise: gcloud storage rm -r "$BUCKET"
