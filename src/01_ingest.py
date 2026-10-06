"""
Stage 1 - Ingest & clean.

Reads the eight raw annual CSV exports of the data.gov.sg "Historical Rainfall
across Singapore" collection (~8 GB), applies record-level cleaning, and writes
a compact analysis-ready Parquet store partitioned by year.

Cleaning performed here
-----------------------
1. Schema is declared explicitly (no inference pass over 8 GB).
2. Only the TB1 5-minute rainfall total readings in mm are retained.
3. Observation timestamps are parsed as Asia/Singapore and *snapped* to the
   nearest 5-minute grid point, because the 2017 export is offset by a few
   seconds and drifts between polls.
4. Exact and near-duplicate records (the same station / 5-min slot re-published
   under a later `update_timestamp`) are collapsed to the most recently updated
   reading.
5. Physically impossible values (negative, or above PLAUSIBLE_MAX_5MIN) are
   dropped and counted in the audit table.
6. Station coordinates are reconciled: a station may appear with slightly
   different lon/lat across years, so the modal coordinate is used.

Outputs
-------
data_parquet/readings/          year-partitioned fact table
data_parquet/stations/          station dimension (id, name, lon, lat)
outputs/tables/clean_audit.csv  per-year record accounting for the report
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pyspark.sql.functions as F
from pyspark.sql.types import DoubleType, StringType, StructField, StructType
from pyspark.sql.window import Window

from common import DATA_PARQUET, DATA_RAW, PLAUSIBLE_MAX_5MIN, YEARS, spark_session, write_table

RAW_SCHEMA = StructType([
    StructField("date", StringType()),
    StructField("timestamp", StringType()),
    StructField("update_timestamp", StringType()),
    StructField("station_id", StringType()),
    StructField("station_name", StringType()),
    StructField("station_device_id", StringType()),
    StructField("location_longitude", DoubleType()),
    StructField("location_latitude", DoubleType()),
    StructField("reading_update_timestamp", StringType()),
    StructField("reading_value", DoubleType()),
    StructField("reading_type", StringType()),
    StructField("reading_unit", StringType()),
])

READING_TYPE = "TB1 Rainfall 5 Minute Total F"


def main() -> None:
    t0 = time.time()
    spark = spark_session("dsa5208-ingest", shuffle_partitions=96)

    files = [str(DATA_RAW / f"Historical Rainfall across Singapore ({y}).csv") for y in YEARS]
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        raise SystemExit(f"Missing raw files:\n  " + "\n  ".join(missing))

    raw = (
        spark.read.option("header", True)
        .option("mode", "PERMISSIVE")
        .schema(RAW_SCHEMA)
        .csv(files)
    )

    # --- parse & normalise -------------------------------------------------
    ts = F.to_timestamp("timestamp")
    parsed = raw.select(
        F.col("station_id"),
        F.col("station_name"),
        F.col("location_longitude").alias("lon"),
        F.col("location_latitude").alias("lat"),
        # snap to the 5-minute grid: round to nearest 300 s
        F.timestamp_seconds(
            (F.round(F.unix_timestamp(ts) / F.lit(300)) * F.lit(300)).cast("long")
        ).alias("ts"),
        F.col("reading_value").alias("rain_mm"),
        F.to_timestamp("reading_update_timestamp").alias("upd"),
        F.col("reading_type"),
    ).filter(F.col("reading_type") == READING_TYPE)

    parsed = parsed.withColumn("year", F.year("ts"))

    # --- per-year audit counters ------------------------------------------
    audit = parsed.groupBy("year").agg(
        F.count(F.lit(1)).alias("raw_records"),
        F.sum(F.col("ts").isNull().cast("int")).alias("null_timestamp"),
        F.sum(F.col("rain_mm").isNull().cast("int")).alias("null_value"),
        F.sum((F.col("rain_mm") < 0).cast("int")).alias("negative_value"),
        F.sum((F.col("rain_mm") > PLAUSIBLE_MAX_5MIN).cast("int")).alias("implausible_value"),
    )

    valid = parsed.filter(
        F.col("ts").isNotNull()
        & F.col("station_id").isNotNull()
        & F.col("rain_mm").isNotNull()
        & (F.col("rain_mm") >= 0)
        & (F.col("rain_mm") <= PLAUSIBLE_MAX_5MIN)
    )

    # --- de-duplicate: keep the most recently updated reading per slot ------
    # A max() over a struct performs the "argmax" in a single shuffle, which is
    # far cheaper than a sorted window over ~50 M rows.
    dedup = (
        valid.groupBy("station_id", "ts")
        .agg(
            F.max(F.struct(F.col("upd"), F.col("rain_mm"))).alias("pick"),
            F.count(F.lit(1)).alias("n_dupes"),
            F.first("station_name", ignorenulls=True).alias("station_name"),
            F.first("lon", ignorenulls=True).alias("lon"),
            F.first("lat", ignorenulls=True).alias("lat"),
        )
        .select(
            "station_id",
            "ts",
            F.col("pick.rain_mm").alias("rain_mm"),
            "n_dupes",
            "station_name",
            "lon",
            "lat",
        )
    )

    dedup_audit = dedup.groupBy(F.year("ts").alias("year")).agg(
        F.count(F.lit(1)).alias("clean_records"),
        F.sum(F.col("n_dupes") - F.lit(1)).alias("duplicates_removed"),
        F.countDistinct("station_id").alias("stations"),
    )

    # --- station dimension: modal coordinate per station -------------------
    coord_counts = (
        dedup.groupBy("station_id", "station_name", "lon", "lat")
        .agg(F.count(F.lit(1)).alias("n"))
    )
    w = Window.partitionBy("station_id").orderBy(F.col("n").desc(), F.col("station_name"))
    stations = (
        coord_counts.withColumn("rk", F.row_number().over(w))
        .filter(F.col("rk") == 1)
        .select("station_id", "station_name", "lon", "lat")
    )

    # --- write fact table ---------------------------------------------------
    fact = (
        dedup.select("station_id", "ts", "rain_mm")
        .withColumn("year", F.year("ts"))
        .withColumn("month", F.month("ts"))
        .withColumn("doy", F.dayofyear("ts"))
        .withColumn("date", F.to_date("ts"))
        .withColumn("hour", F.hour("ts"))
        .withColumn("minute_of_day", F.hour("ts") * F.lit(60) + F.minute("ts"))
    )

    out_fact = DATA_PARQUET / "readings"
    print(f"[ingest] writing fact table -> {out_fact}")
    (
        fact.repartition("year")
        .write.mode("overwrite")
        .partitionBy("year")
        .parquet(str(out_fact))
    )

    out_stn = DATA_PARQUET / "stations"
    stations.coalesce(1).write.mode("overwrite").parquet(str(out_stn))

    # --- audit table --------------------------------------------------------
    audit_pdf = (
        audit.join(dedup_audit, "year", "outer")
        .orderBy("year")
        .toPandas()
    )
    audit_pdf["retention_pct"] = (
        100.0 * audit_pdf["clean_records"] / audit_pdf["raw_records"]
    ).round(2)
    write_table(audit_pdf, "t01_cleaning_audit")
    print(audit_pdf.to_string(index=False))

    n_stations = stations.count()
    print(f"[ingest] stations: {n_stations}")
    print(f"[ingest] done in {time.time() - t0:0.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
