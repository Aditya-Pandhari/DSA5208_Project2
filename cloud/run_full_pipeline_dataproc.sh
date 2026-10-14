#!/usr/bin/env bash
# DSA5208 Project 2 — run the COMPLETE notebook on a Google Cloud Dataproc cluster (from Cloud Shell, block by block).
# Cluster: master + 2 workers, n2-highmem-4 (4 vCPU, 32 GB) = 12 vCPUs, the education project's quota; SSD-backed
# pd-balanced disks 100 + 75 + 75 GB = the 250 GB SSD quota. Cost ~US$1/hour; the run takes ~3-5 hours.
# --max-age=12h deletes the cluster after 12 hours even if it is forgotten.

# ---- 0. every new Cloud Shell session ---------------------------------------------------------------
gcloud config set project dsa5208-project2-511504
export REGION=asia-southeast1 ZONE=asia-southeast1-b BUCKET=gs://dsa5208-rain-juhirishu16 CLUSTER=dsa5208-full
gcloud storage ls $BUCKET/code/full_run/         # expect: DSA5208_Project2_dataproc.ipynb, master_run.sh, singapore_boundary.geojson

# ---- 1. create the cluster (~2 min) ---------------------------------------------------------------
gcloud dataproc clusters create $CLUSTER --region=$REGION --zone=$ZONE --image-version=2.2-debian12 \
  --master-machine-type=n2-highmem-4 --master-boot-disk-type=pd-balanced --master-boot-disk-size=100 \
  --num-workers=2 --worker-machine-type=n2-highmem-4 --worker-boot-disk-type=pd-balanced --worker-boot-disk-size=75 \
  --max-age=12h

# ---- 2. start the run on the master node (returns immediately; the run continues on the cluster) ---------
# The first `gcloud compute ssh` creates an SSH key: press Enter at the passphrase prompts.
gcloud compute ssh $CLUSTER-m --zone=$ZONE --command \
  "gsutil -q cp $BUCKET/code/full_run/master_run.sh /tmp/ && setsid nohup bash /tmp/master_run.sh $BUCKET \
   'Dataproc 2.2, master + 2 workers n2-highmem-4 (4 vCPU, 32 GB), YARN, HDFS' > /tmp/master_run.out 2>&1 < /dev/null & echo started"

# ---- 3. check progress whenever you like (Cloud Shell may disconnect; the run does not stop) ------------
gcloud compute ssh $CLUSTER-m --zone=$ZONE --command "cat /tmp/dsa5208/progress.log; grep -o '\[[0-9.]*\. [^]]*\] [0-9.]* s' /tmp/dsa5208/run.log | tail -3"
gcloud storage ls $BUCKET/full_run/              # DONE appears here when everything is uploaded

# ---- 4. when DONE exists: download the results and DELETE the cluster ----------------------------------
gcloud storage cp $BUCKET/full_run/dataproc_full_run.tar.gz .       # then download it to your laptop (Cloud Shell ⋮ > Download)
gcloud dataproc clusters delete $CLUSTER --region=$REGION --quiet
gcloud dataproc clusters list --region=$REGION                     # should list nothing
