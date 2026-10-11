"""
Shared Task 2 definitions: target construction, feature definitions, splits.

Timestamp semantics (verified against the raw export)
-----------------------------------------------------
`reading_update_timestamp` trails `timestamp` by >= 284 s in every year sampled
(median ~5.5-11 min), so a row stamped T is published only after T + 5 min: the
stamp marks the START of its 5-minute interval, [T, T+5 min).

Every example is therefore anchored on the latest fully observed slot, with
stamp `ts`. The clock *prediction time* is

    t = ts + 5 min                      (the instant that slot completed)

The "next 30 minutes" are [t, t+30 min), i.e. the six slots stamped
ts+5, ts+10, ..., ts+30 min. In anchor terms this is exactly "t+5 ... t+30" of
the assignment, and everything at or before `ts` is information available at t.

All windows are TIME-based (`rangeBetween` on epoch seconds) with an explicit
count of slots present, never row-based, so missing slots cannot silently
stretch a window.
"""
from __future__ import annotations

import math

import numpy as np
import pyspark.sql.functions as F
from pyspark.sql.window import Window

SLOT_SEC = 300
PRED_LAG_SEC = 300                  # t = ts + 5 min (see module docstring)
HORIZON_SLOTS = 6                   # 30 min / 5 min
WET_5MIN_MM = 0.2

# chronological split, defined on the clock prediction time t
TRAIN_END = "2023-01-01 00:00:00"   # SGT; train: t + 30 min <= 2023-01-01
VAL_END = "2024-01-01 00:00:00"     # val:   2023-01-01 <= t, t + 30 min <= 2024-01-01
SPLITS = {"train": "2017-2022", "validation": "2023", "test": "2024"}

# neighbour map: nearest K gauges within MAX_KM (static coordinates only)
NBR_K = 4
NBR_MAX_KM = 10.0

SEASONS = [
    ("NE Monsoon (Dec-Mar)", [12, 1, 2, 3]),
    ("Inter-monsoon (Apr-May)", [4, 5]),
    ("SW Monsoon (Jun-Sep)", [6, 7, 8, 9]),
    ("Inter-monsoon (Oct-Nov)", [10, 11]),
]


def season_col(month_col):
    expr = F.lit(None).cast("string")
    for name, months in reversed(SEASONS):
        expr = F.when(month_col.isin(months), F.lit(name)).otherwise(expr)
    return expr


def _w(back_sec: int, fwd_sec: int):
    """Per-station window over epoch seconds, [ts - back, ts + fwd]."""
    return (Window.partitionBy("station_id").orderBy(F.col("ts_s"))
            .rangeBetween(-back_sec, fwd_sec))


def add_future_target(df):
    """Add `future_30min_mm` and `n_future` (slots found in ts+5..ts+30 min).

    `future_30min_mm` is NULL unless all six future slots exist, so an
    incomplete window can never be mistaken for a dry one. Expects columns
    station_id, ts, rain_mm.
    """
    if "ts_s" not in df.columns:
        df = df.withColumn("ts_s", F.col("ts").cast("long"))
    w = (Window.partitionBy("station_id").orderBy("ts_s")
         .rangeBetween(SLOT_SEC, HORIZON_SLOTS * SLOT_SEC))
    return (df.withColumn("n_future", F.count("rain_mm").over(w))
              .withColumn("_fsum", F.sum("rain_mm").over(w))
              .withColumn("future_30min_mm",
                          F.when(F.col("n_future") == HORIZON_SLOTS, F.col("_fsum")))
              .drop("_fsum"))


def add_trailing_features(df):
    """Rainfall known at prediction time: windows END at the anchor slot.

    A window is valid only if every slot in it is present; otherwise the
    feature is NULL and the row is dropped by `complete_rows`.
    """
    if "ts_s" not in df.columns:
        df = df.withColumn("ts_s", F.col("ts").cast("long"))
    out = df.withColumn("r5", F.col("rain_mm"))
    for name, slots in [("acc15", 3), ("acc30", 6), ("acc60", 12), ("acc180", 36)]:
        w = _w((slots - 1) * SLOT_SEC, 0)
        out = (out.withColumn(f"_n{name}", F.count("rain_mm").over(w))
                  .withColumn(name, F.when(F.col(f"_n{name}") == slots,
                                           F.sum("rain_mm").over(w)))
                  .drop(f"_n{name}"))
    w60 = _w(11 * SLOT_SEC, 0)
    out = (out.withColumn("max5_60", F.max("rain_mm").over(w60))
              .withColumn("wet_slots_60",
                          F.sum((F.col("rain_mm") >= WET_5MIN_MM).cast("int")).over(w60))
              .withColumn("is_wet", (F.col("rain_mm") >= WET_5MIN_MM).cast("double")))
    return out


TRAILING_COLS = ["r5", "acc15", "acc30", "acc60", "acc180", "max5_60", "wet_slots_60", "is_wet"]


def add_time_features(df):
    """Calendar features at the CLOCK prediction time t = ts + 5 min (SGT)."""
    t = F.col("ts") + F.expr("INTERVAL 5 MINUTES")
    mod = F.hour(t) * 60 + F.minute(t)
    doy = F.dayofyear(t)
    return (df.withColumn("pred_time", t)
              .withColumn("hour", F.hour(t))
              .withColumn("minute_of_day", mod)
              .withColumn("tod_sin", F.sin(2 * math.pi * mod / 1440.0))
              .withColumn("tod_cos", F.cos(2 * math.pi * mod / 1440.0))
              .withColumn("month", F.month(t))
              .withColumn("doy_sin", F.sin(2 * math.pi * doy / 365.25))
              .withColumn("doy_cos", F.cos(2 * math.pi * doy / 365.25))
              .withColumn("season", season_col(F.month(t))))


def neighbour_map(stations_pdf, k: int = NBR_K, max_km: float = NBR_MAX_KM):
    """Static nearest-K neighbour map (station_id, nbr_id, dist_km).

    Uses only station coordinates, never rainfall, so it carries no target
    information. Equirectangular distance is accurate to <0.1% at this scale.
    """
    ids = stations_pdf["station_id"].tolist()
    lat = stations_pdf["lat"].to_numpy(float)
    lon = stations_pdf["lon"].to_numpy(float)
    rows = []
    for i, s in enumerate(ids):
        dy = (lat - lat[i]) * 110.574
        dx = (lon - lon[i]) * 111.320 * np.cos(np.radians(lat[i]))
        d = np.hypot(dx, dy)
        order = [j for j in np.argsort(d) if j != i and d[j] <= max_km][:k]
        rows += [(s, ids[j], float(d[j])) for j in order]
    return rows


def add_neighbour_features(spark, feats, lookup, nbr_rows):
    """Neighbour summaries at the same anchor slot.

    `lookup` = (station_id, ts, r5, acc30, acc60) for the neighbours' own
    trailing windows (already restricted to <= ts, so no future leakage).
    Strategy: join the lookup table to the small reversed edge map (broadcast,
    no shuffle), then group by (target station, ts). The 48 M-row table is
    never cross-joined station-by-station.
    """
    edges = spark.createDataFrame(nbr_rows, ["station_id", "nbr_id", "dist_km"])
    rev = edges.select(F.col("nbr_id").alias("station_id"),
                       F.col("station_id").alias("target_id"))
    agg = (lookup.join(F.broadcast(rev), "station_id")
           .groupBy("target_id", "ts")
           .agg(F.avg("acc30").alias("nbr_acc30_mean"),
                F.max("acc30").alias("nbr_acc30_max"),
                F.avg("acc60").alias("nbr_acc60_mean"),
                F.max("r5").alias("nbr_r5_max"),
                F.count("acc30").alias("n_nbr"))
           .withColumnRenamed("target_id", "station_id"))
    out = feats.join(agg, ["station_id", "ts"], "left")
    out = out.withColumn("n_nbr", F.coalesce(F.col("n_nbr"), F.lit(0)))
    # no neighbour reporting -> neighbour signal is *absent*, encoded as 0 with
    # an explicit n_nbr == 0 indicator, never as a fake "dry neighbours" fact
    for c in ["nbr_acc30_mean", "nbr_acc30_max", "nbr_acc60_mean", "nbr_r5_max"]:
        out = out.withColumn(c, F.coalesce(F.col(c), F.lit(0.0)))
    return out.withColumn("has_nbr", (F.col("n_nbr") > 0).cast("double"))


def split_col(pred_time_col="pred_time"):
    """train / validation / test, purged so no label window crosses a boundary."""
    t = F.col(pred_time_col)
    end = t + F.expr("INTERVAL 30 MINUTES")
    return (F.when(end <= F.to_timestamp(F.lit(TRAIN_END)), "train")
             .when((t >= F.to_timestamp(F.lit(TRAIN_END))) & (end <= F.to_timestamp(F.lit(VAL_END))), "validation")
             .when(t >= F.to_timestamp(F.lit(VAL_END)), "test"))


def hash_frac(*cols, seed: int = 5208):
    """Deterministic uniform [0,1) per key (xxhash64 is stable across runs)."""
    h = F.xxhash64(*cols, F.lit(seed))
    return (F.abs(h % F.lit(1_000_000)).cast("double")) / 1_000_000.0


# ---------------------------------------------------------------------------
# Model inputs and Spark ML pipelines
# ---------------------------------------------------------------------------
THR_TRAIN = [0.2, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0, 15.0, 20.0, 30.0]
THR_EVAL = [0.2, 1.0, 2.0, 5.0, 10.0, 20.0]

LOG_COLS = {"l_r5": "r5", "l_acc15": "acc15", "l_acc30": "acc30", "l_acc60": "acc60",
            "l_acc180": "acc180", "l_max5_60": "max5_60",
            "l_nbr30_mean": "nbr_acc30_mean", "l_nbr30_max": "nbr_acc30_max",
            "l_nbr60_mean": "nbr_acc60_mean", "l_nbr_r5_max": "nbr_r5_max"}

LOCAL_NUM = ["lat", "lon", "tod_sin", "tod_cos", "doy_sin", "doy_cos",
             "l_r5", "l_acc15", "l_acc30", "l_acc60", "l_acc180", "l_max5_60",
             "wet_slots_60", "is_wet", "lh"]
NBR_NUM = ["l_nbr30_mean", "l_nbr30_max", "l_nbr60_mean", "l_nbr_r5_max", "has_nbr"]
# threshold x rainfall-state interactions (logistic regression only; trees learn
# these themselves)
INTERACT = ["l_r5", "l_acc15", "l_acc30", "l_acc60", "l_acc180", "l_nbr30_mean",
            "l_nbr30_max", "l_nbr_r5_max", "is_wet"]


def add_model_inputs(df):
    """log1p transforms, the threshold input and its interactions.

    Requires `threshold_mm` (a real numeric input of the model) in `df`.
    """
    for new, old in LOG_COLS.items():
        df = df.withColumn(new, F.log1p(F.col(old)))
    df = df.withColumn("lh", F.log1p(F.col("threshold_mm")))
    for c in INTERACT:
        df = df.withColumn(f"ix_{c}", F.col("lh") * F.col(c))
    return df


def feature_names(spatial: bool = True, interactions: bool = False):
    num = list(LOCAL_NUM) + (list(NBR_NUM) if spatial else [])
    if interactions:
        num += [f"ix_{c}" for c in INTERACT if spatial or "nbr" not in c]
    return num


def build_pipeline(kind: str, spatial: bool = True, interactions: bool = False, **hp):
    """kind = 'lr' | 'gbt'.

    LR: station and season one-hot encoded (handleInvalid='keep' so a gauge
    absent from training maps to an extra 'unknown' slot instead of failing).
    GBT: location enters through lat/lon, season as an indexed categorical.
    """
    from pyspark.ml import Pipeline
    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler

    num = feature_names(spatial, interactions)
    stages = [StringIndexer(inputCol="season", outputCol="season_idx", handleInvalid="keep")]
    if kind == "lr":
        stages += [
            StringIndexer(inputCol="station_id", outputCol="station_idx", handleInvalid="keep"),
            OneHotEncoder(inputCols=["station_idx", "season_idx"],
                          outputCols=["station_ohe", "season_ohe"], handleInvalid="keep"),
            VectorAssembler(inputCols=num + ["station_ohe", "season_ohe"], outputCol="features"),
            LogisticRegression(featuresCol="features", labelCol="label", maxIter=hp.get("maxIter", 60),
                               regParam=hp.get("regParam", 1e-5), elasticNetParam=0.0, tol=1e-6),
        ]
    else:
        stages += [
            VectorAssembler(inputCols=num + ["season_idx"], outputCol="features"),
            GBTClassifier(featuresCol="features", labelCol="label", maxIter=hp.get("maxIter", 60),
                          maxDepth=hp.get("maxDepth", 5), stepSize=hp.get("stepSize", 0.1),
                          maxBins=32, subsamplingRate=hp.get("subsamplingRate", 0.7), seed=5208),
        ]
    return Pipeline(stages=stages)


def with_label(df):
    return df.withColumn("label", (F.col("future_30min_mm") > F.col("threshold_mm")).cast("double"))
