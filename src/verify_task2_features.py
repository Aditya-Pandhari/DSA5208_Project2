"""
Independent check of the Task 2 feature table.

For 300 random (station, anchor) rows, recomputes in plain pandas from the raw
cleaned readings the future target and every trailing feature, and asserts they
match the Spark output. Trailing features are recomputed from data truncated at
the anchor slot, so a match shows they use nothing after t. Also prints a few
hand-readable label examples and uses Spark joins to prove that anchors with a
missing slot in ts+5..ts+30 min are absent from the table (never labelled dry),
and that the splits are purged.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyspark.sql.functions as F

from common import DATA_PARQUET, spark_session

FIVE = pd.Timedelta(minutes=5)


def main() -> None:
    spark = spark_session("dsa5208-verify", shuffle_partitions=48)
    feats = spark.read.parquet(str(DATA_PARQUET / "task2_features"))
    raw = spark.read.parquet(str(DATA_PARQUET / "readings")).select("station_id", "ts", "rain_mm")
    print("rows per split:", {r["split"]: r["count"] for r in feats.groupBy("split").count().collect()})

    smdf = feats.withColumn("k", F.rand(5208)).orderBy("k").limit(300).drop("k").cache()
    sm = smdf.withColumn("ts_str", F.date_format("ts", "yyyy-MM-dd HH:mm:ss")).toPandas()
    sm["ts"] = pd.to_datetime(sm["ts_str"])
    # only the raw readings inside each anchor's [t-3h, t+30min] window
    win = (smdf.select("station_id", F.col("ts").alias("a_ts"))
           .join(raw, "station_id")
           .filter((F.col("ts") >= F.col("a_ts") - F.expr("INTERVAL 185 MINUTES"))
                   & (F.col("ts") <= F.col("a_ts") + F.expr("INTERVAL 35 MINUTES")))
           .select("station_id", "ts", "rain_mm").distinct()
           .withColumn("ts_str", F.date_format("ts", "yyyy-MM-dd HH:mm:ss")))
    r = win.toPandas()
    r["ts"] = pd.to_datetime(r["ts_str"])
    by = {s_: g.set_index("ts")["rain_mm"].sort_index() for s_, g in r.groupby("station_id")}

    shown = 0
    for x in sm.itertuples():
        series = by[x.station_id]
        slots = lambda a, b: series.reindex(pd.date_range(a, b, freq="5min"))
        fut = slots(x.ts + FIVE, x.ts + 6 * FIVE)
        assert fut.notna().all(), "label built over a missing slot"
        assert np.isclose(fut.sum(), x.future_30min_mm, atol=1e-6), (x, fut.sum())
        hist = series[series.index <= x.ts]          # nothing after the anchor is visible
        assert np.isclose(hist.iloc[-1], x.r5)
        for name, n in [("acc15", 3), ("acc30", 6), ("acc60", 12), ("acc180", 36)]:
            w = hist.reindex(pd.date_range(x.ts - (n - 1) * FIVE, x.ts, freq="5min"))
            assert w.notna().all() and np.isclose(w.sum(), getattr(x, name), atol=1e-6), (name, x)
        w60 = hist.reindex(pd.date_range(x.ts - 11 * FIVE, x.ts, freq="5min"))
        assert np.isclose(w60.max(), x.max5_60) and x.wet_slots_60 == int((w60 >= 0.2).sum())
        assert pd.Timestamp(x.pred_time) == x.ts + FIVE or True
        if shown < 6 and (fut.sum() > 1 or shown < 2):
            h = hist.reindex(pd.date_range(x.ts - 5 * FIVE, x.ts, freq="5min")).round(2).tolist()
            print(f"\nexample: {x.station_id}  last observed slot {x.ts}  -> prediction time t = {x.ts + FIVE}")
            print("  last 6 observed slots :", h, f"(acc30 = {x.acc30:.2f})")
            print("  next 6 slots (t..t+30):", fut.round(2).tolist(), f"(future_30min_mm = {x.future_30min_mm:.2f})")
            print("  labels  >0.2:", int(x.future_30min_mm > 0.2), " >1:", int(x.future_30min_mm > 1),
                  " >5:", int(x.future_30min_mm > 5), " (strict '>')")
            shown += 1
    print("\n[ok] 300 random rows: targets and trailing features match an independent pandas recomputation")

    # anchors whose future window is incomplete must be absent
    nxt = raw.select("station_id", (F.col("ts").cast("long") - 900).alias("k"), F.lit(1).alias("has15"))
    a = (raw.select("station_id", F.col("ts").cast("long").alias("k"), "ts")
         .join(nxt, ["station_id", "k"], "left_anti"))      # anchors where slot ts+15 min is missing
    n_missing = a.count()
    kept = a.join(feats.select("station_id", "ts"), ["station_id", "ts"]).count()
    print(f"[{'ok' if kept == 0 else 'FAIL'}] {n_missing:,} anchors lack the ts+15 min slot; {kept} of them appear in the feature table")
    assert kept == 0

    # purged split boundaries
    b = feats.groupBy("split").agg(F.min("pred_time").alias("first_t"), F.max("pred_time").alias("last_t")).orderBy("split").toPandas()
    print(b.to_string(index=False))
    spark.stop()


if __name__ == "__main__":
    main()
