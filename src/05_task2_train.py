"""
Stage 5 - Task 2 training, model selection on 2023, single pass on 2024.

Problem:  P( rain over the next 30 min > h | station, prediction time, state at t )
treated as ONE binary classifier whose input includes the threshold h.

Training data (2017-2022)
  Every (station, prediction time) row of the training split is a candidate.
  For each threshold h in THR_TRAIN a row is kept iff
        hash(station, time, h) / 2^64 < rate[h],
  with rate[h] = min(R_CAP, TARGET_POS / n_pos[h]) - a function of the threshold
  only, never of an individual label. Conditioning on h, the kept rows are
  therefore an unbiased draw, so no probability recalibration is needed (there
  is no label-dependent downsampling). Rare thresholds are sampled more densely.

Evaluation (2023 validation, 2024 test)
  A fixed hash-selected 20 % of prediction times, each paired with ALL training
  thresholds, so every model and threshold is scored on the same held-out
  times. Headline metrics are reported at THR_EVAL.

Selection uses 2023 only (macro log-loss skill vs the climatological baseline);
2024 is scored once afterwards and no decision depends on it.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.sql.window import Window

import common
from common import DATA_PARQUET, ROOT, spark_session, write_table
from task2_eval import (bin_aggregate, group_metrics, metrics, pr_points, prob_col,
                        reliability)
from task2_lib import (THR_EVAL, THR_TRAIN, add_model_inputs, build_pipeline,
                       feature_names, hash_frac, with_label)

FEATURES = DATA_PARQUET / "task2_features"
MODEL_DIR = ROOT / "models" / "task2_exceedance"
TARGET_POS = 8000        # target positives per training threshold ...
R_CAP = 0.06             # ... but never sample more than 6 % of times
EVAL_FRAC = 0.20
GBT_FRAC = 0.35          # GBT trains on a deterministic subset of the LR sample
MONO_TOL = 0.01          # post-process if > 1 % of adjacent threshold pairs violate
BASELINE_PSEUDO = 20     # additive smoothing weight (pseudo-observations)

CANDIDATES = {
    "LR_local":          dict(kind="lr", spatial=False, interactions=False, regParam=1e-5),
    "LR_spatial":        dict(kind="lr", spatial=True, interactions=False, regParam=1e-5),
    "LR_spatial_ix":     dict(kind="lr", spatial=True, interactions=True, regParam=1e-5),
    "LR_spatial_ix_reg": dict(kind="lr", spatial=True, interactions=True, regParam=1e-3),
    "GBT_spatial_d5":    dict(kind="gbt", spatial=True, maxDepth=5, maxIter=80),
    "GBT_spatial_d7":    dict(kind="gbt", spatial=True, maxDepth=7, maxIter=80),
}


def log(msg):
    print(f"[{datetime.now():%H:%M:%S}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# data sets
# ---------------------------------------------------------------------------
RAW_COLS = ["station_id", "ts", "pred_time", "lat", "lon", "season", "hour", "minute_of_day",
            "tod_sin", "tod_cos", "month", "doy_sin", "doy_cos", "r5", "acc15", "acc30",
            "acc60", "acc180", "max5_60", "wet_slots_60", "is_wet", "nbr_acc30_mean",
            "nbr_acc30_max", "nbr_acc60_mean", "nbr_r5_max", "n_nbr", "has_nbr",
            "future_30min_mm"]


def thr_frame(spark, thresholds, extra=None):
    rows = [(float(h),) + ((extra[h],) if extra else ()) for h in thresholds]
    return spark.createDataFrame(rows, ["threshold_mm"] + (["rate"] if extra else []))


def build_sets(spark, feats, sets_dir, scale, eval_frac, smoke):
    if (sets_dir / "_done").exists():
        log(f"re-using sets in {sets_dir}")
        return pd.read_csv(sets_dir / "design.csv")
    shutil.rmtree(sets_dir, ignore_errors=True)
    sets_dir.mkdir(parents=True)
    tr = feats.filter(F.col("split") == "train").select(*RAW_COLS)
    stats = tr.agg(*[F.sum((F.col("future_30min_mm") > h).cast("long")).alias(f"p{i}")
                     for i, h in enumerate(THR_TRAIN)], F.count(F.lit(1)).alias("n")).first()
    n_tr = stats["n"]
    rates = {h: min(R_CAP, TARGET_POS * scale / max(stats[f"p{i}"], 1)) for i, h in enumerate(THR_TRAIN)}
    design = pd.DataFrame({"threshold_mm": THR_TRAIN,
                           "n_pos_population": [stats[f"p{i}"] for i in range(len(THR_TRAIN))],
                           "population_pos_rate": [stats[f"p{i}"] / n_tr for i in range(len(THR_TRAIN))],
                           "sample_rate": [rates[h] for h in THR_TRAIN]})
    thr = thr_frame(spark, THR_TRAIN, rates)
    s = (tr.crossJoin(F.broadcast(thr))
         .filter(hash_frac("station_id", F.col("ts").cast("long"), "threshold_mm", seed=5208) < F.col("rate"))
         .drop("rate"))
    s = with_label(s).repartition(48)
    s.write.mode("overwrite").parquet(str(sets_dir / "train"))
    tsm = spark.read.parquet(str(sets_dir / "train"))
    agg = tsm.groupBy("threshold_mm").agg(F.count(F.lit(1)).alias("rows"), F.sum("label").alias("positives")).toPandas()
    design = design.merge(agg, on="threshold_mm")
    design["sample_pos_rate"] = design["positives"] / design["rows"]
    write_table(design, "t27_task2_training_design")
    log("training sample:\n" + design.round(5).to_string(index=False))

    allthr = thr_frame(spark, THR_TRAIN)
    for split in ("validation", "test"):
        ev = (feats.filter(F.col("split") == split).select(*RAW_COLS)
              .filter(hash_frac("station_id", F.col("ts").cast("long"), seed=2023) < eval_frac))
        ev = with_label(ev.crossJoin(F.broadcast(allthr))).repartition(24)
        ev.write.mode("overwrite").parquet(str(sets_dir / split))
    design.to_csv(sets_dir / "design.csv", index=False)
    (sets_dir / "_done").write_text("ok")
    return design


# ---------------------------------------------------------------------------
# baseline (Model 0): training-only climatology
# ---------------------------------------------------------------------------
def climatology(spark, feats):
    tr = feats.filter(F.col("split") == "train")
    parts = []
    for h in THR_TRAIN:
        y = (F.col("future_30min_mm") > h).cast("double")
        parts.append(tr.groupBy("season", "hour").agg(F.count(F.lit(1)).alias("n"), F.sum(y).alias("pos"))
                     .withColumn("threshold_mm", F.lit(float(h))))
    cell = parts[0]
    for q in parts[1:]:
        cell = cell.unionByName(q)
    cell = cell.toPandas()
    glob = cell.groupby("threshold_mm")[["n", "pos"]].sum()
    glob["p_h"] = glob["pos"] / glob["n"]
    cell = cell.merge(glob["p_h"].reset_index(), on="threshold_mm")
    cell["p_clim"] = (cell["pos"] + BASELINE_PSEUDO * cell["p_h"]) / (cell["n"] + BASELINE_PSEUDO)
    return (spark.createDataFrame(cell[["threshold_mm", "season", "hour", "p_clim"]]),
            spark.createDataFrame(glob.reset_index()[["threshold_mm", "p_h"]]))


# ---------------------------------------------------------------------------
# scoring / monotonicity helpers
# ---------------------------------------------------------------------------
def with_cummin(pred):
    w = (Window.partitionBy("station_id", "ts").orderBy("threshold_mm")
         .rowsBetween(Window.unboundedPreceding, Window.currentRow))
    return pred.withColumn("p", F.min("p").over(w))


def monotonic_stats(pred):
    w = Window.partitionBy("station_id", "ts").orderBy("threshold_mm")
    d = pred.withColumn("dp", F.col("p") - F.lag("p").over(w)).filter(F.col("dp").isNotNull())
    d = d.withColumn("viol", (F.col("dp") > 1e-9).cast("int")).withColumn("mag", F.when(F.col("dp") > 1e-9, F.col("dp")))
    a = d.agg(F.count(F.lit(1)).alias("pairs"), F.sum("viol").alias("viol_pairs"),
              F.avg("mag").alias("mean_violation"), F.max("mag").alias("max_violation"),
              F.expr("percentile_approx(mag, 0.95, 100000)").alias("p95_violation")).first()
    seq = d.groupBy("station_id", "ts").agg(F.max("viol").alias("any")).agg(
        F.count(F.lit(1)).alias("sequences"), F.sum("any").alias("seq_viol")).first()
    return {"sequences": seq["sequences"], "pct_sequences_with_violation": 100 * seq["seq_viol"] / seq["sequences"],
            "adjacent_pairs": a["pairs"], "pct_pairs_violating": 100 * a["viol_pairs"] / a["pairs"],
            "mean_violation_prob": a["mean_violation"], "p95_violation_prob": a["p95_violation"],
            "max_violation_prob": a["max_violation"]}


def macro(df, base):
    m = df.merge(base[["threshold_mm", "logloss", "brier"]].rename(
        columns={"logloss": "ll0", "brier": "br0"}), on="threshold_mm")
    return {"macro_pr_auc": m["pr_auc"].mean(), "macro_roc_auc": m["roc_auc"].mean(),
            "macro_logloss": m["logloss"].mean(), "macro_brier": m["brier"].mean(),
            "macro_logloss_skill": (1 - m["logloss"] / m["ll0"]).mean(),
            "macro_brier_skill": (1 - m["brier"] / m["br0"]).mean()}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny run for debugging; writes to scratch")
    ap.add_argument("--only", nargs="*", help="restrict to these candidate names")
    ap.add_argument("--refit", action="store_true", help="ignore cached candidate models")
    args = ap.parse_args()
    scale, eval_frac, tag = (0.03, 0.01, "_smoke") if args.smoke else (1.0, EVAL_FRAC, "")
    if args.smoke:
        scratch = ROOT / "logs" / "smoke_tables"
        scratch.mkdir(parents=True, exist_ok=True)
        common.TABLES = scratch
    sets_dir = DATA_PARQUET / f"task2_sets{tag}"
    cand_dir = ROOT / "models" / f"_candidates{tag}"
    t0 = time.time()

    spark = spark_session("dsa5208-task2-train", shuffle_partitions=64)
    feats = spark.read.parquet(str(FEATURES))
    stations = spark.read.parquet(str(DATA_PARQUET / "stations")).toPandas()
    design = build_sets(spark, feats, sets_dir, scale, eval_frac, args.smoke)

    train = spark.read.parquet(str(sets_dir / "train"))
    val = spark.read.parquet(str(sets_dir / "validation")).cache()
    test = spark.read.parquet(str(sets_dir / "test"))
    split_rows = {s: feats.filter(F.col("split") == s).count() for s in ("train", "validation", "test")}
    log(f"split rows (prediction times): {split_rows}; train sample rows {train.count():,}; "
        f"validation eval rows {val.count():,}")

    clim_cell, clim_thr = climatology(spark, feats)
    lon_cuts = np.quantile(stations["lon"], [1 / 3, 2 / 3])
    region = (F.when(F.col("lon") < float(lon_cuts[0]), "West")
              .when(F.col("lon") < float(lon_cuts[1]), "Central").otherwise("East"))
    band = F.concat(F.lpad((F.floor(F.col("hour") / 4) * 4).cast("string"), 2, "0"), F.lit("-"),
                    F.lpad((F.floor(F.col("hour") / 4) * 4 + 3).cast("string"), 2, "0"))

    def score(name, model, df):
        base_cols = ["station_id", "ts", "threshold_mm", "label", "season", "hour", "lon"]
        if name == "Climatology":
            out = df.join(F.broadcast(clim_cell), ["threshold_mm", "season", "hour"], "left").withColumnRenamed("p_clim", "p")
        elif name == "BaseRate_by_threshold":
            out = df.join(F.broadcast(clim_thr), "threshold_mm", "left").withColumnRenamed("p_h", "p")
        else:
            out = prob_col(model.transform(add_model_inputs(df)))
        return (out.select(*base_cols, "p").withColumn("region", region).withColumn("hour_band", band))

    # ---- baselines on validation ------------------------------------------
    results, mono_rows, val_cut = {}, {}, {}
    for nm in ("Climatology", "BaseRate_by_threshold"):
        pred = score(nm, None, val)
        b = bin_aggregate(pred.filter(F.col("threshold_mm").isin(THR_EVAL)), ["threshold_mm"])
        results[nm] = group_metrics(b, ["threshold_mm"])
    base = results["Climatology"]

    # ---- fit + validate candidates ----------------------------------------
    models, fit_time = {}, {}
    names = [n for n in CANDIDATES if not args.only or n in args.only]
    for nm in names:
        spec = dict(CANDIDATES[nm])
        path = cand_dir / nm
        if path.exists() and not args.refit:
            models[nm] = PipelineModel.load(str(path))
            log(f"loaded cached {nm}")
        else:
            tdf = train
            if spec["kind"] == "gbt":
                tdf = tdf.filter(hash_frac("station_id", F.col("ts").cast("long"), "threshold_mm", seed=77) < GBT_FRAC)
            tdf = add_model_inputs(tdf).cache()
            n_fit = tdf.count()
            log(f"fitting {nm} on {n_fit:,} rows ...")
            t1 = time.time()
            kind = spec.pop("kind")
            models[nm] = build_pipeline(kind, **spec).fit(tdf)
            fit_time[nm] = time.time() - t1
            tdf.unpersist()
            log(f"  fitted {nm} in {fit_time[nm]:.0f}s")
            models[nm].write().overwrite().save(str(path))
        pred = score(nm, models[nm], val)
        b = bin_aggregate(pred.filter(F.col("threshold_mm").isin(THR_EVAL)), ["threshold_mm"])
        results[nm] = group_metrics(b, ["threshold_mm"])
        mono_rows[nm] = monotonic_stats(pred)
        ms = macro(results[nm], base)
        log(f"  {nm}: val macro PR-AUC {ms['macro_pr_auc']:.4f}  logloss skill {ms['macro_logloss_skill']:.4f}  "
            f"monotonic-violating pairs {mono_rows[nm]['pct_pairs_violating']:.2f}%")

    # ---- validation table + selection (2023 only) --------------------------
    long = pd.concat([r.assign(model=m) for m, r in results.items()])
    write_table(long[["model", "threshold_mm"] + [c for c in long.columns if c not in ("model", "threshold_mm")]],
                "t28_task2_validation_metrics")
    summ = []
    for m, r in results.items():
        row = {"model": m, "split": "validation", **macro(r, base)}
        row.update({f"mono_{k}": v for k, v in mono_rows.get(m, {}).items()})
        row["fit_seconds"] = fit_time.get(m, np.nan)
        summ.append(row)
    summ = pd.DataFrame(summ)
    cands = summ[summ["model"].isin(names)].sort_values("macro_logloss_skill", ascending=False)
    final_name = cands.iloc[0]["model"]
    best_lr = cands[cands["model"].str.startswith("LR")].iloc[0]["model"] if cands["model"].str.startswith("LR").any() else None
    best_gbt = cands[cands["model"].str.startswith("GBT")].iloc[0]["model"] if cands["model"].str.startswith("GBT").any() else None
    use_mono = bool(mono_rows[final_name]["pct_pairs_violating"] / 100 > MONO_TOL)
    summ["selected"] = summ["model"] == final_name
    write_table(summ, "t29_task2_model_comparison_validation")
    log(f"SELECTED on 2023: {final_name}  (monotone post-processing: {use_mono})")

    # ---- final variants -----------------------------------------------------
    finals = {"Climatology": None, "BaseRate_by_threshold": None}
    for nm in {best_lr, best_gbt, final_name} - {None}:
        finals[nm] = models[nm]
    report_names = list(finals) + ([f"{final_name}+cummin"] if use_mono else [])

    def final_pred(name, df):
        if name.endswith("+cummin"):
            return with_cummin(score(final_name, models[final_name], df))
        return score(name, finals[name], df)

    # operating cutoffs: F1-optimal per (model, threshold) on VALIDATION
    cut, tabs = {}, {"rel": [], "pr": [], "metrics": [], "sub": []}
    vbin = {}
    for nm in report_names:
        pv = final_pred(nm, val).filter(F.col("threshold_mm").isin(THR_EVAL))
        vb = bin_aggregate(pv, ["threshold_mm"])
        vbin[nm] = vb
        gm = group_metrics(vb, ["threshold_mm"])
        for r in gm.itertuples():
            cut[(nm, r.threshold_mm)] = r.cutoff_bin
        tabs["metrics"].append(gm.assign(model=nm, split="validation", operating_point="F1-optimal on 2023 (in-sample for 2023)"))
        for h, g in vb.groupby("threshold_mm"):
            tabs["rel"].append(reliability(g).assign(model=nm, split="validation", threshold_mm=h))

    # ---- UNTOUCHED 2024 test, scored once, after selection -----------------
    log("scoring 2024 test with models selected on 2023")
    sub_bins = {}
    for nm in report_names:
        pt = final_pred(nm, test).filter(F.col("threshold_mm").isin(THR_EVAL))
        tb = bin_aggregate(pt, ["threshold_mm"])
        co = {(h,): cut[(nm, h)] for h in THR_EVAL}
        gm = group_metrics(tb, ["threshold_mm"], cutoffs=co)
        tabs["metrics"].append(gm.assign(model=nm, split="test", operating_point="F1-optimal cutoff chosen on 2023, applied unchanged"))
        for h, g in tb.groupby("threshold_mm"):
            tabs["rel"].append(reliability(g).assign(model=nm, split="test", threshold_mm=h))
            tabs["pr"].append(pr_points(g).assign(model=nm, split="test", threshold_mm=h))
        if nm in (final_name, f"{final_name}+cummin") and (nm.endswith("+cummin") or not use_mono):
            for dim in ("season", "hour_band", "region"):
                for split_name, df in (("validation", val), ("test", test)):
                    pp = final_pred(nm, df).filter(F.col("threshold_mm").isin(THR_EVAL))
                    sb = bin_aggregate(pp, ["threshold_mm", dim])
                    sg = group_metrics(sb, ["threshold_mm", dim])
                    sg = sg.rename(columns={dim: "group"}).assign(dimension=dim, model=nm, split=split_name)
                    tabs["sub"].append(sg)
    mt = pd.concat(tabs["metrics"])
    write_table(mt[["model", "split", "threshold_mm"] + [c for c in mt.columns if c not in ("model", "split", "threshold_mm")]],
                "t30_task2_final_metrics")
    write_table(pd.concat(tabs["rel"]), "t31_task2_reliability")
    write_table(pd.concat(tabs["pr"]), "t32_task2_pr_curves")
    sub = pd.concat(tabs["sub"])
    sub["small_sample"] = sub["n_pos"] < 100
    write_table(sub, "t33_task2_subgroups")

    # macro summary incl. test
    tb_base = mt[(mt["model"] == "Climatology") & (mt["split"] == "test")]
    ms_rows = []
    for (m, sp), g in mt.groupby(["model", "split"]):
        ref = tb_base if sp == "test" else base
        ms_rows.append({"model": m, "split": sp, **macro(g, ref)})
    write_table(pd.DataFrame(ms_rows), "t34_task2_macro_summary")

    # monotonicity on the final model, validation + test
    mrows = []
    for nm in report_names:
        for sp, df in (("validation", val), ("test", test)):
            if nm in (final_name, f"{final_name}+cummin", best_lr, best_gbt):
                mrows.append({"model": nm, "split": sp, **monotonic_stats(final_pred(nm, df))})
    write_table(pd.DataFrame(mrows), "t35_task2_monotonicity")

    # MLlib cross-check of the binned PR-AUC / ROC-AUC (final model, 2023, 5 mm)
    ev = BinaryClassificationEvaluator(rawPredictionCol="p", labelCol="label")
    pp = final_pred(final_name, val).filter(F.col("threshold_mm") == 5.0)
    ml_pr, ml_roc = ev.setMetricName("areaUnderPR").evaluate(pp), ev.setMetricName("areaUnderROC").evaluate(pp)
    mine = results[final_name].set_index("threshold_mm").loc[5.0]
    log(f"cross-check @5mm 2023: MLlib PR {ml_pr:.4f} vs binned AP {mine['pr_auc']:.4f}; "
        f"MLlib ROC {ml_roc:.4f} vs binned {mine['roc_auc']:.4f}")
    pd.DataFrame([{"model": final_name, "threshold_mm": 5.0, "mllib_areaUnderPR": ml_pr,
                   "binned_average_precision": mine["pr_auc"], "mllib_areaUnderROC": ml_roc,
                   "binned_roc_auc": mine["roc_auc"]}]).pipe(write_table, "t36_task2_metric_crosscheck")

    # feature effects ---------------------------------------------------------
    fm = models[final_name]
    probe = add_model_inputs(train.limit(200000))
    tr_df = fm.transform(probe)
    attrs = tr_df.schema["features"].metadata["ml_attr"]["attrs"]
    names_f = {a["idx"]: a["name"] for grp in attrs.values() for a in grp}
    imp = []
    last = fm.stages[-1]
    if hasattr(last, "featureImportances"):
        for i, v in enumerate(last.featureImportances.toArray()):
            imp.append({"model": final_name, "feature": names_f.get(i, str(i)), "importance": v})
    lr_name = best_lr
    lm = models[lr_name]
    from pyspark.ml.stat import Summarizer
    ldf = lm.transform(add_model_inputs(train.limit(300000)))
    sd = ldf.select(Summarizer.metrics("std").summary(F.col("features")).alias("s")).first()["s"]["std"].toArray()
    lattrs = ldf.schema["features"].metadata["ml_attr"]["attrs"]
    lnames = {a["idx"]: a["name"] for grp in lattrs.values() for a in grp}
    coefs = lm.stages[-1].coefficients.toArray()
    for i, c in enumerate(coefs):
        imp.append({"model": lr_name, "feature": lnames.get(i, str(i)), "coefficient": c,
                    "std_effect": c * sd[i]})
    write_table(pd.DataFrame(imp), "t37_task2_feature_effects")

    # threshold sweep demo (final model, 2024) -------------------------------
    dense = [round(float(x), 3) for x in np.unique(np.r_[THR_TRAIN, np.geomspace(0.2, 30, 36)])]
    stn = pd.read_csv(ROOT / "outputs" / "tables" / "t16_station_climatology.csv")
    avail = set(r[0] for r in feats.filter(F.col("split") == "test").select("station_id").distinct().collect())
    st = next(s for s in stn.sort_values("mean_annual_mm", ascending=False)["station_id"] if s in avail)
    cand = (feats.filter((F.col("split") == "test") & (F.col("station_id") == st) & F.col("hour").between(14, 16)))
    cases = {"dry (no rain in last hour)": (F.col("acc60") == 0) & (F.col("nbr_acc30_max") == 0),
             "light (0.4-1.2 mm in last 30 min)": F.col("acc30").between(0.4, 1.2),
             "moderate (3-8 mm in last 30 min)": F.col("acc30").between(3, 8),
             "heavy (>= 15 mm in last 30 min)": F.col("acc30") >= 15}
    picks = []
    for lab, cond in cases.items():
        r = (cand.filter(cond).withColumn("hk", hash_frac("ts", seed=11)).orderBy("hk").select(*RAW_COLS).limit(1))
        picks.append(r.withColumn("case", F.lit(lab)))
    pk = picks[0]
    for q in picks[1:]:
        pk = pk.unionByName(q)
    pk = pk.crossJoin(F.broadcast(thr_frame(spark, dense)))
    dp = prob_col(fm.transform(add_model_inputs(with_label(pk)))).select(
        "case", "station_id", "pred_time", "acc30", "acc60", "nbr_acc30_max", "threshold_mm", "p").toPandas()
    rows = []
    for (cs,), g in dp.groupby(["case"]):
        g = g.sort_values("threshold_mm").copy()
        g["p_raw"] = g["p"]
        g["p_final"] = g["p"].cummin() if use_mono else g["p"]
        rows.append(g.drop(columns="p"))
    write_table(pd.concat(rows), "t38_task2_threshold_sweep")

    # save the selected model ---------------------------------------------------
    shutil.rmtree(MODEL_DIR, ignore_errors=True)
    MODEL_DIR.mkdir(parents=True)
    fm.write().overwrite().save(str(MODEL_DIR / "pipeline_model"))
    fcut = mt[(mt["model"] == (f"{final_name}+cummin" if use_mono else final_name)) & (mt["split"] == "validation")]
    meta = {
        "selected_model": final_name, "spec": CANDIDATES[final_name], "monotone_postprocess_cummin": use_mono,
        "threshold_grid": THR_TRAIN, "headline_thresholds": THR_EVAL,
        "features": feature_names(CANDIDATES[final_name]["spatial"], CANDIDATES[final_name].get("interactions", False)),
        "split": {"train": "prediction time + 30 min <= 2023-01-01", "validation": "2023", "test": "2024"},
        "split_rows_prediction_times": split_rows,
        "operating_cutoff_p_F1_on_2023": {str(r.threshold_mm): r.cutoff_p for r in fcut.itertuples()},
        "timestamp_convention": "row stamp = START of 5-min interval; prediction time t = last observed stamp + 5 min",
        "neighbours": {"k": 4, "max_km": 10.0}, "trained": datetime.now().isoformat(timespec="seconds"),
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    log(f"saved {final_name} -> {MODEL_DIR}")
    log(f"total {time.time() - t0:.0f}s")
    spark.stop()


if __name__ == "__main__":
    main()
