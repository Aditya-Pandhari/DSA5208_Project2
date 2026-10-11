"""
Stage 4 - Task 2 feature table.

Builds one row per (station, prediction time) from the cleaned Parquet store:
  * future_30min_mm  - rainfall over the NEXT 30 minutes (target source)
  * trailing local rainfall, calendar and neighbour features (all <= t)
  * split label (train / validation / test), purged at the year boundaries

Rows are kept only if (a) all six future slots exist and (b) the 3 h trailing
history is complete, so no label or feature is computed from a window that
silently stretched over missing observations. Rows dropped for either reason are
counted in outputs/tables/t25_task2_row_audit.csv.

Output: data_parquet/task2_features/  (partitioned by split; gitignored)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import pyspark.sql.functions as F

from common import DATA_PARQUET, spark_session, write_table
from task2_lib import (TRAILING_COLS, add_future_target, add_neighbour_features,
                       add_time_features, add_trailing_features, neighbour_map,
                       split_col)

FEATURES_PATH = DATA_PARQUET / "task2_features"


def main() -> None:
    t0 = time.time()
    spark = spark_session("dsa5208-task2-features", shuffle_partitions=96)
    stations = spark.read.parquet(str(DATA_PARQUET / "stations")).toPandas()
    stations = stations.merge(stations[["station_id"]], on="station_id")
    nbr = neighbour_map(stations)
    print(f"[task2] neighbour map: {len(nbr)} edges for {len(stations)} stations "
          f"(mean {len(nbr) / len(stations):.2f} per station)")

    df = (spark.read.parquet(str(DATA_PARQUET / "readings"))
          .select("station_id", "ts", "rain_mm")
          .withColumn("ts_s", F.col("ts").cast("long")))
    df = add_future_target(add_trailing_features(df))
    df = df.join(F.broadcast(spark.createDataFrame(
        stations[["station_id", "lat", "lon"]])), "station_id")
    df.cache()

    n_all = df.count()
    n_fut = df.filter(F.col("future_30min_mm").isNotNull()).count()
    n_hist = df.filter(F.col("acc180").isNotNull()).count()
    ok = df.filter(F.col("future_30min_mm").isNotNull() & F.col("acc180").isNotNull())
    n_ok = ok.count()
    print(f"[task2] rows {n_all:,}; complete future {n_fut:,}; complete 3 h history "
          f"{n_hist:,}; both {n_ok:,}")

    # neighbour lookup uses every row whose own trailing windows are complete to
    # 1 h; values are <= ts, so they are legitimately known at prediction time
    lookup = (df.filter(F.col("acc60").isNotNull())
              .select("station_id", "ts", "r5", "acc30", "acc60"))
    feats = add_neighbour_features(spark, ok, lookup, nbr)
    feats = add_time_features(feats).withColumn("split", split_col())
    feats = feats.filter(F.col("split").isNotNull())

    cols = (["station_id", "ts", "pred_time", "split", "lat", "lon", "season", "hour",
             "minute_of_day", "tod_sin", "tod_cos", "month", "doy_sin", "doy_cos"]
            + TRAILING_COLS
            + ["nbr_acc30_mean", "nbr_acc30_max", "nbr_acc60_mean", "nbr_r5_max",
               "n_nbr", "has_nbr", "future_30min_mm"])
    (feats.select(*cols)
          .repartition("split")
          .write.mode("overwrite").partitionBy("split").parquet(str(FEATURES_PATH)))

    out = spark.read.parquet(str(FEATURES_PATH))
    out.cache()
    audit = (out.groupBy("split")
             .agg(F.count(F.lit(1)).alias("rows"),
                  F.countDistinct("station_id").alias("stations"),
                  F.min("pred_time").alias("first_pred_time"),
                  F.max("pred_time").alias("last_pred_time"),
                  F.avg("has_nbr").alias("frac_with_neighbour"),
                  *[F.avg((F.col("future_30min_mm") > h).cast("double")).alias(f"pos_rate_gt_{h}")
                    for h in (0.2, 1.0, 5.0, 10.0, 20.0)],
                  *[F.sum((F.col("future_30min_mm") > h).cast("long")).alias(f"n_pos_gt_{h}")
                    for h in (0.2, 1.0, 5.0, 10.0, 20.0)])
             .toPandas().sort_values("split"))
    audit["rows_all_raw"] = n_all
    audit["rows_incomplete_future"] = n_all - n_fut
    audit["rows_incomplete_history_or_future"] = n_all - n_ok
    write_table(audit, "t25_task2_row_audit")
    print(audit.T.to_string())

    q = out.approxQuantile("future_30min_mm", [0.5, 0.9, 0.99, 0.999, 0.9999], 1e-5)
    dist = pd.DataFrame({"quantile": [0.5, 0.9, 0.99, 0.999, 0.9999], "future_30min_mm": q})
    write_table(dist, "t26_future30_quantiles")
    print(dist.to_string(index=False))
    print(f"[task2] features done in {time.time() - t0:0.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
