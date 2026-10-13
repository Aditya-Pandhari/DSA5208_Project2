# Dataproc scalability runs

`t43_cloud_scaling.csv` holds the timings of `cloud/dataproc_benchmark.py` on Google Cloud Dataproc
(project `dsa5208-project2-511504`, region `asia-southeast1`, 13 Oct 2026), transcribed from the job logs;
the Dataproc job IDs are in the last column. Cluster: master and workers `n2-standard-2` (2 vCPU, 8 GB),
50 GB pd-standard boot disks, image `2.2-debian12` (Spark 3.5), data in a Cloud Storage bucket.
The 2-worker job ingested the CSVs and timed the cleaning job; the 4-worker job reused the bronze layer
(`--skip-ingest`). Each cleaning job ran once as a warm-up (not recorded) and three times timed.
Both cluster sizes return the same usable station-day total as the laptop (1,195,661.7 mm).
