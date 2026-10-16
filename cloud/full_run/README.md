# The complete notebook executed on Google Cloud Dataproc

`DSA5208_Project2_dataproc.ipynb` is the notebook as it stood on 14 October 2026, generated from the submitted
notebook by `cloud/make_dataproc_notebook.py` and executed on a Dataproc 2.2 cluster (master + 2 workers
`n2-highmem-4`, Spark 3.5.3 on YARN, data layers on HDFS) by `cloud/master_run.sh`. Its outputs (`outputs/`),
model metadata (`models/`), run log, HDFS sizes and the YARN application record are in this folder.

It was run **before** the last revisions of the submitted notebook (the Spark plan cell 12.4, the strong/weak-scaling
rewrite of Section 20, the Stage 2 train/test boundary purge and the fixed-grid `predict()`), so its code differs from
the submitted notebook in those cells. The analysis it reproduces — cleaning, Task 1, Stage 1, the Stage 2
cross-validation, the final model and the 2024 evaluation — is the same, and the report's Section 8.6 compares its
results with the laptop run. The cluster-trained Spark model folders were not retrieved before the cluster was
deleted (a quoting bug in the collection step, since fixed in `master_run.sh`); the submitted model is the laptop-trained
one.
