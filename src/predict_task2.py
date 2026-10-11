"""
Task 2 prediction interface.

    python src/predict_task2.py --station S228 --time "2024-10-01 15:00:00" --threshold 5.0

Returns P(rain over the next 30 minutes > threshold) for one station at one
prediction time. The user supplies only those three inputs: every derived
feature (recent rainfall, neighbouring gauges, calendar terms) is rebuilt here
from the cleaned Parquet store using ONLY observations up to the prediction
time, with the same code that built the training table (task2_lib).

Conventions: times are Singapore time (SGT); a prediction time is snapped down
to the 5-minute grid; the latest information used is the 5-minute interval that
ended at the prediction time.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "task2_exceedance"
FIVE = pd.Timedelta(minutes=5)
FIRST_SLOT = pd.Timestamp("2017-01-01 00:00:00")
LAST_SLOT = pd.Timestamp("2024-12-31 23:55:00")
HISTORY_SLOTS = 36           # 3 h of complete 5-minute history


class PredictionError(ValueError):
    """A request that cannot be answered, with a user-facing reason."""


def check_inputs(station, time_str, threshold, meta, known_stations):
    if station not in known_stations:
        near = [s for s in known_stations if s.lower().startswith(station.lower()[:2])][:8]
        raise PredictionError(f"Unknown station '{station}'. {len(known_stations)} stations are known "
                              f"(e.g. {', '.join(sorted(known_stations)[:8])}...)." + (f" Similar: {near}" if near else ""))
    try:
        t = pd.Timestamp(time_str)
    except Exception:
        raise PredictionError(f"Cannot parse time '{time_str}'. Use e.g. '2024-10-01 15:00:00' (SGT).")
    if t.tzinfo is not None:
        t = t.tz_convert("Asia/Singapore").tz_localize(None)
    if not math.isfinite(threshold):
        raise PredictionError("Threshold must be a finite number of millimetres.")
    if threshold < 0:
        raise PredictionError(f"Threshold must not be negative (got {threshold} mm).")
    lo, hi = min(meta["threshold_grid"]), max(meta["threshold_grid"])
    if not lo <= threshold <= hi:
        raise PredictionError(f"Threshold {threshold} mm is outside the supported range {lo}-{hi} mm "
                              f"(gauge resolution is 0.2 mm; the model was trained on {lo}-{hi} mm).")
    t_snap = t.floor("5min")
    anchor = t_snap - FIVE                      # last fully observed interval
    if anchor > LAST_SLOT or anchor < FIRST_SLOT + (HISTORY_SLOTS - 1) * FIVE:
        raise PredictionError(
            f"Prediction time {t} is outside the available history: the cleaned data run "
            f"{FIRST_SLOT + HISTORY_SLOTS * FIVE} to {LAST_SLOT + FIVE} (3 h of history are needed).")
    return t, t_snap, anchor


def load(spark_app="dsa5208-predict"):
    from pyspark.ml import PipelineModel
    from common import spark_session
    if not (MODEL_DIR / "metadata.json").exists():
        raise PredictionError(f"No trained model at {MODEL_DIR}. Run ./run_task2.sh first.")
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    spark = spark_session(spark_app, shuffle_partitions=8)
    model = PipelineModel.load(str(MODEL_DIR / "pipeline_model"))
    return spark, model, meta


def features_at(spark, station, anchor):
    """One-row frame of prediction-time features, built from data <= anchor."""
    import pyspark.sql.functions as F
    from common import DATA_PARQUET
    from task2_lib import (add_neighbour_features, add_time_features,
                           add_trailing_features, neighbour_map)

    stations = spark.read.parquet(str(DATA_PARQUET / "stations")).toPandas()
    nbr = [e for e in neighbour_map(stations) if e[0] == station]
    ids = [station] + [e[1] for e in nbr]
    lo = anchor - 185 * pd.Timedelta(minutes=1)
    years = sorted({lo.year, anchor.year})
    rd = (spark.read.parquet(str(DATA_PARQUET / "readings"))
          .filter(F.col("year").isin(years) & F.col("station_id").isin(ids))
          .filter((F.col("ts") >= F.to_timestamp(F.lit(str(lo)))) & (F.col("ts") <= F.to_timestamp(F.lit(str(anchor)))))
          .select("station_id", "ts", "rain_mm"))           # nothing after the anchor is even loaded
    tr = add_trailing_features(rd).cache()
    row = tr.filter((F.col("station_id") == station) & (F.col("ts") == F.to_timestamp(F.lit(str(anchor)))))
    if row.count() == 0:
        raise PredictionError(f"No observation for {station} in the 5-minute interval ending at "
                              f"{anchor + FIVE}: the gauge was not reporting.")
    n_hist = tr.filter(F.col("station_id") == station).count()
    if row.filter(F.col("acc180").isNotNull()).count() == 0:
        raise PredictionError(f"Insufficient recent history for {station} before {anchor + FIVE}: "
                              f"{n_hist} of {HISTORY_SLOTS} five-minute readings in the previous 3 h "
                              f"(a complete record is required).")
    loc = stations[stations["station_id"] == station][["station_id", "lat", "lon"]]
    row = row.join(F.broadcast(spark.createDataFrame(loc)), "station_id")
    lookup = tr.filter(F.col("acc60").isNotNull()).select("station_id", "ts", "r5", "acc30", "acc60")
    row = add_neighbour_features(spark, row, lookup, [e for e in nbr])
    return add_time_features(row), len(nbr)


def predict(station, time_str, threshold, curve=False, _ctx=None):
    spark, model, meta = _ctx or load()
    import pyspark.sql.functions as F
    from pyspark.ml.functions import vector_to_array
    from task2_lib import add_model_inputs

    t, t_snap, anchor = check_inputs(station, time_str, float(threshold), meta, _known(spark))
    row, n_nbr = features_at(spark, station, anchor)
    grid = sorted(set(meta["threshold_grid"]) | {float(threshold)})
    ths = [g for g in grid if (g <= threshold or curve)] if (meta["monotone_postprocess_cummin"] or curve) else [float(threshold)]
    ths = sorted(set(ths) | {float(threshold)})
    x = row.crossJoin(F.broadcast(spark.createDataFrame([(float(g),) for g in ths], ["threshold_mm"])))
    out = (model.transform(add_model_inputs(x))
           .select("threshold_mm", vector_to_array("probability")[1].alias("p")).toPandas()
           .sort_values("threshold_mm"))
    if meta["monotone_postprocess_cummin"]:
        out["p"] = out["p"].cummin()          # P(R>h) cannot increase with h
    p = float(out.loc[(out["threshold_mm"] - threshold).abs().idxmin(), "p"])
    info = row.select("acc30", "acc60", "r5", "nbr_acc30_max", "n_nbr").first().asDict()
    return {"station": station, "time": t, "time_used": t_snap, "threshold": float(threshold),
            "probability": p, "curve": out if curve else None, "inputs": info, "meta": meta}


_STATIONS = None


def _known(spark):
    global _STATIONS
    if _STATIONS is None:
        from common import DATA_PARQUET
        _STATIONS = set(spark.read.parquet(str(DATA_PARQUET / "stations")).toPandas()["station_id"])
    return _STATIONS


def main() -> int:
    ap = argparse.ArgumentParser(description="P(rain in the next 30 min > threshold) for a Singapore gauge")
    ap.add_argument("--station", required=True, help="station id, e.g. S228")
    ap.add_argument("--time", required=True, help='prediction time in SGT, e.g. "2024-10-01 15:00:00"')
    ap.add_argument("--threshold", required=True, type=float, help="rainfall threshold in mm (0.2-30)")
    ap.add_argument("--curve", action="store_true", help="also print P for every grid threshold")
    a = ap.parse_args()
    try:
        r = predict(a.station, a.time, a.threshold, a.curve)
    except PredictionError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 2
    note = "" if r["time"] == r["time_used"] else f"  (snapped to the 5-minute grid: {r['time_used']:%H:%M})"
    print(f"Station: {r['station']}")
    print(f"Prediction time: {r['time']:%Y-%m-%d %H:%M} SGT{note}")
    print(f"Threshold: {r['threshold']:g} mm")
    print(f"Probability of >{r['threshold']:g} mm rainfall in the next 30 minutes: {r['probability']:.3f}")
    i = r["inputs"]
    print(f"(recent rain: last 5 min {i['r5']:.1f} mm, last 30 min {i['acc30']:.1f} mm, last 60 min {i['acc60']:.1f} mm; "
          f"{i['n_nbr']} neighbouring gauges reporting)")
    if r["curve"] is not None:
        print("\nthreshold_mm  probability")
        for x in r["curve"].itertuples():
            print(f"{x.threshold_mm:>10g}  {x.p:>10.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
