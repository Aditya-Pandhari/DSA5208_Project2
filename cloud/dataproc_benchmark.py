"""Cloud scalability benchmark for DSA5208 Project 2: the notebook's Section 20 job on a Dataproc cluster.

Runs two Spark jobs on the data in a GCS bucket and records their wall time:
  1. ingest  : read the eight annual CSVs with the explicit schema -> bronze Parquet (partitioned by year)
  2. clean   : read bronze -> snap timestamps -> de-duplicate (shuffle 1) -> station-days (shuffle 2) -> one number
               (identical to `bench_job` in Section 20 of the notebook; repeated REPEAT times, first run discarded)

Submit (see run_dataproc.sh):
  gcloud dataproc jobs submit pyspark cloud/dataproc_benchmark.py --cluster=... --region=... \
      -- --bucket gs://<bucket> --label "2 workers x n2-standard-4"
Results are appended to gs://<bucket>/results/cloud_scaling.csv (one row per run) and printed.
"""
import argparse
import json
import time
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType

READING_TYPE = "TB1 Rainfall 5 Minute Total F"
MIN_SLOTS = 231                     # usable station-day: >= 80 % of 288 five-minute slots (as in the notebook)
RAW_SCHEMA = StructType([
    StructField("date", StringType()), StructField("timestamp", StringType()), StructField("update_timestamp", StringType()),
    StructField("station_id", StringType()), StructField("station_name", StringType()), StructField("station_device_id", StringType()),
    StructField("location_longitude", DoubleType()), StructField("location_latitude", DoubleType()),
    StructField("reading_update_timestamp", StringType()), StructField("reading_value", DoubleType()),
    StructField("reading_type", StringType()), StructField("reading_unit", StringType()),
])


def ingest(spark, bucket):
    raw = spark.read.option("header", True).schema(RAW_SCHEMA).csv(f"{bucket}/data/*.csv")
    bronze = raw.select(
        "station_id", "station_name", "station_device_id",
        F.col("location_longitude").alias("lon"), F.col("location_latitude").alias("lat"),
        F.to_timestamp("timestamp").alias("ts_raw"), F.to_timestamp("update_timestamp").alias("update_ts"),
        F.to_timestamp("reading_update_timestamp").alias("upd"), F.col("reading_value").alias("rain_mm"),
        "reading_type", "reading_unit", F.col("date").alias("date_str"),
    ).withColumn("year", F.year("ts_raw"))
    bronze.write.mode("overwrite").partitionBy("year").parquet(f"{bucket}/bronze")
    return spark.read.parquet(f"{bucket}/bronze").count()


def clean_job(spark, bucket):
    """Bronze -> snapped timestamps -> de-duplicated (shuffle 1) -> station-days (shuffle 2) -> one number."""
    b = spark.read.parquet(f"{bucket}/bronze").filter(F.col("reading_type") == READING_TYPE)
    snapped = b.select("station_id", F.timestamp_seconds((F.round(F.unix_timestamp("ts_raw") / F.lit(300)) * F.lit(300))
                                                         .cast("long")).alias("ts"), "rain_mm", "upd")
    d = (snapped.groupBy("station_id", "ts").agg(F.max(F.struct(F.col("upd"), F.col("rain_mm"))).alias("pick"))
         .select("station_id", "ts", F.col("pick.rain_mm").alias("rain_mm")))
    sd = d.groupBy("station_id", F.to_date("ts").alias("date")).agg(F.sum("rain_mm").alias("day_mm"), F.count(F.lit(1)).alias("slots"))
    return sd.filter(F.col("slots") >= MIN_SLOTS).agg(F.sum("day_mm")).first()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True, help="gs://bucket (holds data/*.csv; bronze/ and results/ are written)")
    ap.add_argument("--label", required=True, help="cluster description, e.g. '2 workers x n2-standard-4'")
    ap.add_argument("--repeat", type=int, default=3, help="timed repetitions of the cleaning job (after one warm-up)")
    ap.add_argument("--skip-ingest", action="store_true", help="reuse an existing bronze/ in the bucket")
    args = ap.parse_args()

    spark = (SparkSession.builder.appName("dsa5208-cloud-benchmark")
             .config("spark.sql.session.timeZone", "Asia/Singapore")
             .config("spark.sql.shuffle.partitions", "96").getOrCreate())
    sc = spark.sparkContext
    cores = int(sc.getConf().get("spark.executor.instances", "0") or 0) * int(sc.getConf().get("spark.executor.cores", "0") or 0)
    print(f"[{datetime.now():%H:%M:%S}] cluster: {args.label}; default parallelism {sc.defaultParallelism}; "
          f"executor cores (from conf) {cores or 'dynamic'}")

    rows = []
    if not args.skip_ingest:
        t0 = time.time()
        n = ingest(spark, args.bucket)
        dt = time.time() - t0
        print(f"[{datetime.now():%H:%M:%S}] ingest: {n:,} rows -> bronze in {dt:.1f} s")
        rows.append({"label": args.label, "job": "ingest (CSV -> bronze Parquet)", "run": 1, "seconds": round(dt, 1), "result": float(n)})

    total = None
    for i in range(args.repeat + 1):
        t0 = time.time()
        total = clean_job(spark, args.bucket)
        dt = time.time() - t0
        tag = "warm-up (discarded)" if i == 0 else f"run {i}"
        print(f"[{datetime.now():%H:%M:%S}] clean job {tag}: {dt:.1f} s  (usable station-day total {total:,.1f} mm)")
        if i > 0:
            rows.append({"label": args.label, "job": "clean (bronze -> dedup -> station-days)", "run": i, "seconds": round(dt, 1), "result": total})

    out = spark.createDataFrame(rows, "label string, job string, run long, seconds double, result double").withColumn("parallelism", F.lit(sc.defaultParallelism)) \
        .withColumn("spark", F.lit(spark.version)).withColumn("when", F.lit(datetime.now().isoformat(timespec="seconds")))
    out.coalesce(1).write.mode("append").option("header", True).csv(f"{args.bucket}/results/cloud_scaling")
    print(json.dumps(rows, indent=1, default=float))
    spark.stop()


if __name__ == "__main__":
    main()
