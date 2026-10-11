"""
Stage 2 - Task 1 analysis (Spark).

Consumes the cleaned Parquet store and emits a set of small result tables under
outputs/tables/. Every figure in the report is drawn from one of these tables,
so the plotting stage never touches the 48 M-row fact table.

Analytical conventions
----------------------
Areal ("island") rainfall is NOT a naive average of all rows: the station
network grows from ~50 gauges (2017-19) to ~76 (2020-22) and back to ~62 (2024),
and individual gauges drop out for days at a time. Averaging rows would let the
changing network masquerade as a change in climate. Instead:

  station-day  is usable if it carries >= MIN_SLOT_FRAC of the 288 daily slots
  island-day   is usable if >= MIN_STATIONS usable station-days exist
  island daily rainfall = unweighted mean of usable station-day totals

Annual totals are then reported twice: the raw sum over usable days, and a
gap-filled estimate where unusable days take the day-of-year climatology from
the remaining years. August 2018 is ~90 % absent from the source export, so the
distinction matters.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.window import Window

from common import (DATA_PARQUET, EXPECTED_INTERVALS_PER_DAY, WET_5MIN_MM,
                    WET_DAY_MM, YEARS, spark_session, write_table)
from task2_lib import add_future_target, add_time_features

MIN_SLOT_FRAC = 0.80          # >= 80 % of 288 slots for a usable station-day
MIN_STATIONS = 20             # >= 20 usable gauges for a usable island-day
MIN_SLOTS = math.ceil(MIN_SLOT_FRAC * EXPECTED_INTERVALS_PER_DAY)   # 231 (230/288 = 79.9 %)
EVENT_GAP_MIN = 30            # dry gap that separates two rainfall events

from task2_lib import season_col


def island_series(station_day, min_stations, stations=None):
    """Unweighted mean of usable station-day totals -> one row per island-day."""
    sd = station_day.filter(F.col("usable"))
    if stations is not None:
        sd = sd.filter(F.col("station_id").isin(list(stations)))
    out = (
        sd.groupBy("date", "year", "month", "doy", "season")
        .agg(
            F.count(F.lit(1)).alias("n_stations"),
            F.avg("day_mm").alias("island_mm"),
            F.max("day_mm").alias("wettest_station_mm"),
            F.avg((F.col("day_mm") >= WET_DAY_MM).cast("double")).alias("frac_stations_wet"),
            F.max("max_5min_mm").alias("peak_5min_mm"),
        )
        .filter(F.col("n_stations") >= min_stations)
        .toPandas()
    )
    out["date"] = pd.to_datetime(out["date"])
    out = out.sort_values("date").reset_index(drop=True)
    out["is_wet_day"] = (out["island_mm"] >= WET_DAY_MM).astype(int)
    return out


def gapfill(island_day, calendar):
    """Calendar-complete series; missing days take the (month, day) climatology."""
    full = calendar.merge(island_day[["date", "island_mm"]], on="date", how="left")
    full["year"] = full["date"].dt.year
    full["doy"] = full["date"].dt.dayofyear
    full["md"] = full["date"].dt.month * 100 + full["date"].dt.day
    full["observed"] = full["island_mm"].notna().astype(int)
    clim = full.groupby("md")["island_mm"].mean()
    clim = clim.reindex(sorted(full["md"].unique())).ffill().bfill()   # 29 Feb with no donor
    full["filled_mm"] = full["island_mm"].fillna(full["md"].map(clim))
    return full


def annual_table(full, cov):
    obs = full[full["observed"] == 1]
    annual = pd.DataFrame({"year": sorted(full["year"].unique())})
    annual = annual.merge(
        obs.groupby("year").agg(
            days_observed=("island_mm", "size"),
            observed_total_mm=("island_mm", "sum"),
            mean_daily_mm=("island_mm", "mean"),
            wet_days=("island_mm", lambda s: int((s >= WET_DAY_MM).sum())),
            p95_daily_mm=("island_mm", lambda s: float(np.percentile(s, 95))),
            max_daily_mm=("island_mm", "max"),
        ).reset_index(), on="year", how="left")
    annual = annual.merge(
        full.groupby("year").agg(
            days_in_year=("island_mm", "size"),
            gapfilled_total_mm=("filled_mm", "sum"),
        ).reset_index(), on="year", how="left")
    annual["days_missing"] = annual["days_in_year"] - annual["days_observed"]
    # scale wet-day count to a full year so the count is comparable across years
    annual["wet_days_scaled"] = (
        annual["wet_days"] * annual["days_in_year"] / annual["days_observed"]
    ).round(1)
    annual = annual.merge(
        cov.groupby("year").agg(stations_active=("station_id", "size")).reset_index(),
        on="year", how="left")
    return annual


def monthly_table(full):
    mon = full.copy()
    mon["month"] = mon["date"].dt.month
    monthly = mon.groupby(["year", "month"]).agg(
        total_mm=("filled_mm", "sum"),
        observed_total_mm=("island_mm", "sum"),
        days_observed=("observed", "sum"),
        days=("observed", "size"),
    ).reset_index()
    monthly["complete"] = (monthly["days_observed"] >= 0.9 * monthly["days"]).astype(int)
    clim_m = monthly.groupby("month")["total_mm"].mean().rename("clim_mm")
    monthly = monthly.merge(clim_m, on="month")
    monthly["anomaly_mm"] = monthly["total_mm"] - monthly["clim_mm"]
    monthly["anomaly_pct"] = 100 * monthly["anomaly_mm"] / monthly["clim_mm"]
    return monthly


def spearman(a, b):
    """Rank correlation without a SciPy dependency."""
    return a.rank().corr(b.rank())


def longest_below_run(monthly):
    """Longest unbroken run of below-average months (anomaly < 0)."""
    best = run = 0
    for v in monthly.sort_values(["year", "month"])["anomaly_mm"]:
        run = run + 1 if v < 0 else 0
        best = max(best, run)
    return best


def sensitivity_annual(full_a, fx_a, n_cohort):
    """Side-by-side annual results. Absolute levels differ between networks
    (different gauges), so shape is compared via an index to each network's own
    8-year mean."""
    out = pd.DataFrame({"year": full_a["year"]})
    for tag, a in [("full", full_a), ("fixed", fx_a)]:
        out[f"total_mm_{tag}"] = a["gapfilled_total_mm"].round(1)
        out[f"index_{tag}"] = (100 * a["gapfilled_total_mm"] / a["gapfilled_total_mm"].mean()).round(1)
        out[f"rank_{tag}"] = a["gapfilled_total_mm"].rank(ascending=False).astype(int)
        out[f"wet_days_{tag}"] = a["wet_days_scaled"]
        out[f"mm_per_wet_day_{tag}"] = (a["gapfilled_total_mm"] / a["wet_days_scaled"]).round(2)
        out[f"days_missing_{tag}"] = a["days_missing"]
    out["index_diff_pp"] = (out["index_fixed"] - out["index_full"]).round(1)
    out["n_stations_fixed"] = n_cohort
    return out


def sensitivity_monthly(full_m, fx_m):
    out = pd.DataFrame({"month": range(1, 13)})
    for tag, m in [("full", full_m), ("fixed", fx_m)]:
        c = m.groupby("month")["total_mm"].mean()
        out[f"clim_mm_{tag}"] = out["month"].map(c).round(1)
        out[f"share_pct_{tag}"] = (100 * out[f"clim_mm_{tag}"] / out[f"clim_mm_{tag}"].sum()).round(2)
        out[f"rank_{tag}"] = out[f"clim_mm_{tag}"].rank(ascending=False).astype(int)
    out["share_diff_pp"] = (out["share_pct_fixed"] - out["share_pct_full"]).round(2)
    return out


def sensitivity_summary(full_a, fx_a, full_m, fx_m, cohort, min_fixed):
    rows = []

    def add(metric, value, note=""):
        rows.append({"metric": metric, "value": value, "note": note})

    ta, tf = full_a["gapfilled_total_mm"], fx_a["gapfilled_total_mm"]
    ia, i_f = ta / ta.mean(), tf / tf.mean()
    add("fixed_network_stations", len(cohort), ",".join(cohort))
    add("fixed_island_day_min_stations", min_fixed, "80% of the cohort")
    add("annual_total_spearman", round(float(spearman(ta, tf)), 3), "rank agreement")
    add("annual_total_pearson", round(float(ta.corr(tf)), 3), "")
    add("wettest_year_full/fixed", f"{int(full_a.loc[ta.idxmax(), 'year'])}/{int(fx_a.loc[tf.idxmax(), 'year'])}", "")
    add("driest_year_full/fixed", f"{int(full_a.loc[ta.idxmin(), 'year'])}/{int(fx_a.loc[tf.idxmin(), 'year'])}", "")
    add("max_abs_annual_index_diff_pp", round(float((100 * (i_f - ia)).abs().max()), 1),
        "index to own 8-yr mean, percentage points")
    add("annual_range_pct_full/fixed",
        f"{100 * (ta.max() / ta.min() - 1):.0f}/{100 * (tf.max() / tf.min() - 1):.0f}", "(max/min - 1)")
    wa, wf = full_a["wet_days_scaled"], fx_a["wet_days_scaled"]
    add("wet_days_spearman", round(float(spearman(wa, wf)), 3), "")
    inta, intf = ta / wa, tf / wf
    pre = full_a["year"] <= 2019
    add("intensity_every_year_2020-24_above_2017-19_mean_full/fixed",
        f"{bool((inta[~pre] > inta[pre].mean()).all())}/{bool((intf[~pre] > intf[pre].mean()).all())}",
        "mm per wet day")
    add("intensity_2020-24_mean_vs_2017-19_mean_pct_full/fixed",
        f"{100 * (inta[~pre].mean() / inta[pre].mean() - 1):.0f}/{100 * (intf[~pre].mean() / intf[pre].mean() - 1):.0f}", "")
    ca = full_m.groupby("month")["total_mm"].mean()
    cf = fx_m.groupby("month")["total_mm"].mean()
    add("monthly_climatology_spearman", round(float(spearman(ca, cf)), 3), "")
    add("wettest_month_full/fixed", f"{int(ca.idxmax())}/{int(cf.idxmax())}", "")
    add("driest_month_full/fixed", f"{int(ca.idxmin())}/{int(cf.idxmin())}", "")
    fm = full_m.sort_values(["year", "month"]).reset_index(drop=True)
    xm = fx_m.sort_values(["year", "month"]).reset_index(drop=True)
    add("monthly_anomaly_pct_pearson", round(float(fm["anomaly_pct"].corr(xm["anomaly_pct"])), 3), "96 months")
    add("monthly_anomaly_sign_agreement_pct",
        round(float(100 * (np.sign(fm["anomaly_mm"]) == np.sign(xm["anomaly_mm"])).mean()), 1), "")
    add("longest_below_average_run_months_full/fixed",
        f"{longest_below_run(full_m)}/{longest_below_run(fx_m)}", "")
    # how many months of the full network's longest run are also below average
    # in the fixed network? (distinguishes a flipped marginal month from a
    # genuinely different drought)
    neg = (fm["anomaly_mm"] < 0).to_numpy()
    best = run = end = 0
    for i, flag in enumerate(neg):
        run = run + 1 if flag else 0
        if run > best:
            best, end = run, i
    win = slice(end - best + 1, end + 1)
    add("below_average_months_in_full_longest_run_full/fixed",
        f"{int(neg[win].sum())}/{int((xm['anomaly_mm'].to_numpy()[win] < 0).sum())}",
        f"{fm['year'][end - best + 1]}-{fm['month'][end - best + 1]:02d} to {fm['year'][end]}-{fm['month'][end]:02d}")
    changed = [f"{int(y)} (full #{rf_}, fixed #{rx_}, index {a_:.1f}/{b_:.1f})"
               for y, rf_, rx_, a_, b_ in zip(
                   full_a["year"], ta.rank(ascending=False), tf.rank(ascending=False),
                   100 * ia, 100 * i_f) if rf_ != rx_]
    add("years_whose_rank_changes", "; ".join(changed) or "none",
        "all swaps are between years whose index differs by only a few percentage points")
    # pre-declared headline checks (not tuned after seeing the result)
    checks = {
        "driest year identical": full_a.loc[ta.idxmin(), "year"] == fx_a.loc[tf.idxmin(), "year"],
        "annual totals Pearson >= 0.95": ta.corr(tf) >= 0.95,
        "max annual index difference <= 5 pp": float((100 * (i_f - ia)).abs().max()) <= 5.0,
        "annual range within 5 pp": abs(100 * ((ta.max() / ta.min()) - (tf.max() / tf.min()))) <= 5.0,
        "monthly climatology Spearman >= 0.9, wettest & driest month identical":
            spearman(ca, cf) >= 0.9 and ca.idxmax() == cf.idxmax() and ca.idxmin() == cf.idxmin(),
        "monthly anomaly r >= 0.9": fm["anomaly_pct"].corr(xm["anomaly_pct"]) >= 0.9,
        "post-2019 intensity above 2017-19 level in both":
            bool((inta[~pre] > inta[pre].mean()).all() and (intf[~pre] > intf[pre].mean()).all()),
    }
    for k, v in checks.items():
        add(f"check: {k}", "PASS" if v else "FAIL")
    add("verdict", "HEADLINE CONCLUSIONS ROBUST" if all(checks.values()) else "MATERIAL CHANGE - REVIEW",
        "see years_whose_rank_changes and longest_below_average_run for the fragile details")
    return pd.DataFrame(rows)


def main() -> None:
    t0 = time.time()
    spark = spark_session("dsa5208-task1", shuffle_partitions=96)
    df = spark.read.parquet(str(DATA_PARQUET / "readings"))
    stations = spark.read.parquet(str(DATA_PARQUET / "stations")).toPandas()
    write_table(stations.sort_values("station_id"), "t02_stations")

    df = df.withColumn("season", season_col(F.col("month")))
    df.cache()
    n_rows = df.count()
    print(f"[task1] fact table: {n_rows:,} readings")

    # =====================================================================
    # A. Completeness / data foundation
    # =====================================================================
    station_day = (
        df.groupBy("station_id", "date", "year", "month", "doy", "season")
        .agg(
            F.count(F.lit(1)).alias("slots"),
            F.sum("rain_mm").alias("day_mm"),
            F.max("rain_mm").alias("max_5min_mm"),
            F.sum((F.col("rain_mm") >= WET_5MIN_MM).cast("int")).alias("wet_slots"),
        )
        .withColumn("usable", F.col("slots") >= F.lit(MIN_SLOTS))
    )
    station_day.cache()

    # A1. station x year coverage matrix
    cov = (
        df.groupBy("station_id", "year")
        .agg(F.count(F.lit(1)).alias("slots"))
        .withColumn(
            "expected",
            F.when(F.col("year") % 4 == 0, F.lit(366 * EXPECTED_INTERVALS_PER_DAY))
            .otherwise(F.lit(365 * EXPECTED_INTERVALS_PER_DAY)),
        )
        .withColumn("coverage", F.col("slots") / F.col("expected"))
        .toPandas()
    )
    write_table(cov.sort_values(["station_id", "year"]), "t03_station_year_coverage")

    # A2. daily network size (for the availability timeline)
    daily_net = (
        station_day.groupBy("date")
        .agg(
            F.count(F.lit(1)).alias("stations_reporting"),
            F.sum(F.col("usable").cast("int")).alias("stations_usable"),
        )
        .toPandas()
    )
    daily_net["date"] = pd.to_datetime(daily_net["date"])
    calendar = pd.DataFrame({"date": pd.date_range("2017-01-01", "2024-12-31", freq="D")})
    daily_net = calendar.merge(daily_net, on="date", how="left").fillna(0)
    daily_net[["stations_reporting", "stations_usable"]] = \
        daily_net[["stations_reporting", "stations_usable"]].astype(int)
    write_table(daily_net, "t04_daily_network")

    # =====================================================================
    # B. Island areal daily series  (the backbone of every temporal figure)
    # =====================================================================
    island_day = island_series(station_day, MIN_STATIONS)
    write_table(island_day, "t05_island_daily")

    full = gapfill(island_day, calendar)
    write_table(full, "t06_island_daily_gapfilled")

    # =====================================================================
    # C. Annual summary
    # =====================================================================
    annual = annual_table(full, cov)
    write_table(annual, "t07_annual_summary")
    print(annual.to_string(index=False))

    # =====================================================================
    # D. Intra-annual structure
    # =====================================================================
    # D1. month x year totals (built from the gap-filled daily series so that
    #     an incomplete month is not silently reported as a dry month)
    monthly = monthly_table(full)
    write_table(monthly.sort_values(["year", "month"]), "t08_monthly")

    # =====================================================================
    # C2. Sensitivity: fixed gauge network
    # =====================================================================
    # The island mean above averages whichever gauges pass QC each day, so the
    # spatial sample changes through time. With a strong west-east gradient
    # (section G) that can leak into the temporal results. Re-derive the same
    # products from the stations that reach >= 80 % coverage in EVERY year.
    ok_years = (cov.assign(ok=cov["coverage"] >= MIN_SLOT_FRAC)
                .groupby("station_id").agg(n=("year", "size"), ok=("ok", "sum")))
    cohort = sorted(ok_years[(ok_years["n"] == len(YEARS)) & (ok_years["ok"] == len(YEARS))].index)
    min_fixed = math.ceil(MIN_SLOT_FRAC * len(cohort))
    print(f"[task1] fixed-network cohort: {len(cohort)} stations, island-day needs >= {min_fixed}")
    fx_day = island_series(station_day, min_fixed, cohort)
    fx_full = gapfill(fx_day, calendar)
    fx_annual = annual_table(fx_full, cov[cov["station_id"].isin(cohort)])
    fx_monthly = monthly_table(fx_full)
    write_table(sensitivity_annual(annual, fx_annual, len(cohort)), "t22_sensitivity_annual")
    write_table(sensitivity_monthly(monthly, fx_monthly), "t23_sensitivity_monthly")
    summ = sensitivity_summary(annual, fx_annual, monthly, fx_monthly, cohort, min_fixed)
    write_table(summ, "t24_sensitivity_summary")
    print(summ.to_string(index=False))


    # D2. seasonal totals per year
    seasonal = (
        island_day.groupby(["year", "season"])
        .agg(observed_mm=("island_mm", "sum"), days=("island_mm", "size"),
             mean_daily_mm=("island_mm", "mean"))
        .reset_index()
    )
    write_table(seasonal, "t09_seasonal")

    # =====================================================================
    # E. Diurnal cycle  (half-hour bin x month x year), as a rate in mm/h
    # =====================================================================
    halfhour = (F.floor(F.col("minute_of_day") / F.lit(30)) * F.lit(30)).alias("hh_min")
    diurnal = (
        df.withColumn("hh_min", halfhour)
        .groupBy("year", "month", "season", "hh_min")
        .agg(
            F.avg("rain_mm").alias("mean_mm_per_5min"),
            F.avg((F.col("rain_mm") >= WET_5MIN_MM).cast("double")).alias("wet_frac"),
            F.count(F.lit(1)).alias("n"),
        )
        .withColumn("mm_per_hour", F.col("mean_mm_per_5min") * F.lit(12))
        .toPandas()
    )
    write_table(diurnal, "t10_diurnal_month_year")

    # E2. diurnal by season and station-region (for the map/diurnal link)
    diurnal_station = (
        df.withColumn("hh_min", halfhour)
        .groupBy("station_id", "season", "hh_min")
        .agg(F.avg("rain_mm").alias("mean_mm_per_5min"))
        .withColumn("mm_per_hour", F.col("mean_mm_per_5min") * F.lit(12))
        .toPandas()
    )
    write_table(diurnal_station, "t11_diurnal_station_season")

    # =====================================================================
    # F. Intensity distribution, concentration, extremes
    # =====================================================================
    # F1. exceedance of 5-minute depths, per year (wet slots only)
    wet = df.filter(F.col("rain_mm") >= WET_5MIN_MM)
    grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0,
            6.0, 7.0, 8.0, 10.0, 12.0, 14.0, 16.0, 20.0]
    totals = {r["year"]: r["n"] for r in
              df.groupBy("year").agg(F.count(F.lit(1)).alias("n")).collect()}
    agg = [F.sum((F.col("rain_mm") >= g).cast("long")).alias(f"ge_{i}")
           for i, g in enumerate(grid)]
    exc_raw = wet.groupBy("year").agg(*agg).toPandas()
    rows = []
    for _, r in exc_raw.iterrows():
        for i, g in enumerate(grid):
            rows.append({"year": int(r["year"]), "threshold_mm": g,
                         "n_exceed": int(r[f"ge_{i}"]),
                         "n_slots": totals[int(r["year"])],
                         "exceed_prob": r[f"ge_{i}"] / totals[int(r["year"])]})
    write_table(pd.DataFrame(rows), "t12_intensity_exceedance")

    # F2. concentration: what share of total rain falls in the wettest x% of
    #     wet intervals (a Lorenz curve for rainfall)
    qs = [i / 200 for i in range(201)]
    lorenz_rows = []
    for yr in sorted(totals):
        vals = (wet.filter(F.col("year") == yr).select("rain_mm")
                .toPandas()["rain_mm"].to_numpy())
        vals = np.sort(np.asarray(vals, dtype=float))
        cum = np.cumsum(vals)
        cum = cum / cum[-1]
        idx = (np.array(qs) * (len(vals) - 1)).astype(int)
        for q, c in zip(qs, cum[idx]):
            lorenz_rows.append({"year": yr, "frac_of_wet_intervals": q,
                                "frac_of_rainfall": float(c)})
    write_table(pd.DataFrame(lorenz_rows), "t13_rain_concentration")

    # F3. rolling 30 / 60-minute accumulations -> annual maxima & exceedances
    w30 = (Window.partitionBy("station_id").orderBy(F.col("ts").cast("long"))
           .rangeBetween(-1500, 0))          # 6 x 5-min slots  = 30 min
    w60 = (Window.partitionBy("station_id").orderBy(F.col("ts").cast("long"))
           .rangeBetween(-3300, 0))          # 12 x 5-min slots = 60 min
    roll = (
        df.select("station_id", "ts", "year", "rain_mm", "season")
        .withColumn("acc30", F.sum("rain_mm").over(w30))
        .withColumn("acc60", F.sum("rain_mm").over(w60))
    )
    roll.cache()

    extremes = (
        roll.groupBy("year")
        .agg(
            F.max("acc30").alias("max_30min_mm"),
            F.max("acc60").alias("max_60min_mm"),
            F.sum((F.col("acc30") >= 5).cast("long")).alias("n_acc30_ge5"),
            F.sum((F.col("acc30") >= 10).cast("long")).alias("n_acc30_ge10"),
            F.sum((F.col("acc30") >= 20).cast("long")).alias("n_acc30_ge20"),
            F.count(F.lit(1)).alias("n_slots"),
            F.expr("percentile_approx(acc30, 0.9999, 1000000)").alias("p9999_30min_mm"),
        )
        .toPandas()
        .sort_values("year")
    )
    for c in ["n_acc30_ge5", "n_acc30_ge10", "n_acc30_ge20"]:
        extremes[c.replace("n_", "rate_")] = extremes[c] / extremes["n_slots"]
    write_table(extremes, "t14_extremes_annual")

    # F4. the quantity Task 2 has to predict: P(rain in the NEXT 30 min > h)
    #     as a base rate, by season and hour of the PREDICTION time. Forward
    #     window, strict ">", complete six-slot windows only (task2_lib).
    fwd = add_time_features(add_future_target(
        df.select("station_id", "ts", "rain_mm")))
    n_all = fwd.count()
    fwd = fwd.filter(F.col("future_30min_mm").isNotNull())
    print(f"[task1] forward-window rows: {fwd.count():,} of {n_all:,} have all six future slots")
    base = (
        fwd.groupBy("season", "hour")
        .agg(
            F.avg((F.col("future_30min_mm") > 0.2).cast("double")).alias("p_gt_0p2"),
            F.avg((F.col("future_30min_mm") > 1.0).cast("double")).alias("p_gt_1"),
            F.avg((F.col("future_30min_mm") > 5.0).cast("double")).alias("p_gt_5"),
            F.avg((F.col("future_30min_mm") > 10.0).cast("double")).alias("p_gt_10"),
            F.avg((F.col("future_30min_mm") > 20.0).cast("double")).alias("p_gt_20"),
            F.count(F.lit(1)).alias("n"),
        )
        .toPandas()
    )
    write_table(base, "t15_base_rates_30min")

    # =====================================================================
    # G. Spatial climatology
    # =====================================================================
    st_year = (
        station_day.filter(F.col("usable"))
        .groupBy("station_id", "year")
        .agg(F.count(F.lit(1)).alias("days"), F.sum("day_mm").alias("total_mm"),
             F.avg("day_mm").alias("mean_daily_mm"))
        .filter(F.col("days") >= 300)
        .toPandas()
    )
    st_clim = (
        st_year.groupby("station_id")
        .agg(n_years=("year", "size"),
             mean_daily_mm=("mean_daily_mm", "mean"))
        .reset_index()
    )
    st_clim["mean_annual_mm"] = st_clim["mean_daily_mm"] * 365.25
    st_clim = st_clim.merge(stations, on="station_id", how="left")
    write_table(st_clim.sort_values("mean_annual_mm", ascending=False),
                "t16_station_climatology")
    write_table(st_year, "t17_station_year_totals")

    # =====================================================================
    # H. Spatial coherence: correlation of 30-min rainfall vs distance
    # =====================================================================
    hh = (
        df.withColumn("slot", (F.unix_timestamp("ts") / F.lit(1800)).cast("long"))
        .groupBy("slot", "station_id")
        .agg(F.sum("rain_mm").alias("mm30"))
    )
    keep = st_clim[st_clim["n_years"] >= 4]["station_id"].tolist()
    wide = (
        hh.filter(F.col("station_id").isin(keep))
        .groupBy("slot").pivot("station_id", keep).agg(F.first("mm30"))
        .toPandas()
        .set_index("slot").sort_index()
    )
    print(f"[task1] coherence matrix: {wide.shape}")
    corr = wide.corr(min_periods=5000)
    meta = st_clim.set_index("station_id")
    pairs = []
    ids = list(corr.columns)
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if a not in meta.index or b not in meta.index:
                continue
            # equirectangular approximation is fine at Singapore's scale
            dlat = (meta.loc[b, "lat"] - meta.loc[a, "lat"]) * 110.574
            dlon = (meta.loc[b, "lon"] - meta.loc[a, "lon"]) * 111.320 * \
                np.cos(np.radians(meta.loc[a, "lat"]))
            pairs.append({"a": a, "b": b,
                          "distance_km": float(np.hypot(dlat, dlon)),
                          "corr": float(corr.loc[a, b])})
    pairs_df = pd.DataFrame(pairs).dropna()
    write_table(pairs_df, "t18_pair_correlation")

    # =====================================================================
    # I. Rainfall events (>= EVENT_GAP_MIN dry gap separates events)
    # =====================================================================
    wev = df.filter(F.col("rain_mm") >= WET_5MIN_MM).select(
        "station_id", "ts", "year", "season", "rain_mm")
    wv = Window.partitionBy("station_id").orderBy(F.col("ts").cast("long"))
    ev = (
        wev.withColumn("prev_ts", F.lag("ts").over(wv))
        .withColumn("gap_min",
                    (F.unix_timestamp("ts") - F.unix_timestamp("prev_ts")) / 60)
        .withColumn("new_event",
                    ((F.col("gap_min") > EVENT_GAP_MIN) | F.col("gap_min").isNull())
                    .cast("int"))
        .withColumn("event_id", F.sum("new_event").over(
            wv.rowsBetween(Window.unboundedPreceding, Window.currentRow)))
    )
    events = (
        ev.groupBy("station_id", "event_id")
        .agg(
            F.min("ts").alias("start_ts"),
            F.max("ts").alias("end_ts"),
            F.sum("rain_mm").alias("depth_mm"),
            F.max("rain_mm").alias("peak_5min_mm"),
            F.count(F.lit(1)).alias("wet_slots"),
            F.first("year").alias("year"),
            F.first("season").alias("season"),
        )
        .withColumn("duration_min",
                    (F.unix_timestamp("end_ts") - F.unix_timestamp("start_ts")) / 60 + 5)
        .withColumn("mean_rate_mm_h", F.col("depth_mm") / (F.col("duration_min") / 60))
        .withColumn("start_hour", F.hour("start_ts"))
    )
    events.cache()

    ev_summary = (
        events.groupBy("year")
        .agg(
            F.count(F.lit(1)).alias("n_events"),
            F.countDistinct("station_id").alias("stations"),
            F.avg("depth_mm").alias("mean_depth_mm"),
            F.expr("percentile_approx(depth_mm, 0.5)").alias("median_depth_mm"),
            F.avg("duration_min").alias("mean_duration_min"),
            F.expr("percentile_approx(duration_min, 0.5)").alias("median_duration_min"),
            F.avg("mean_rate_mm_h").alias("mean_rate_mm_h"),
            F.expr("percentile_approx(depth_mm, 0.99)").alias("p99_depth_mm"),
        )
        .toPandas().sort_values("year")
    )
    ev_summary["events_per_station"] = ev_summary["n_events"] / ev_summary["stations"]
    write_table(ev_summary, "t19_event_summary_annual")

    ev_sample = (
        events.select("duration_min", "depth_mm", "peak_5min_mm", "season",
                      "start_hour", "year")
        .sample(False, 0.12, seed=5208).toPandas()
    )
    write_table(ev_sample, "t20_event_sample")

    ev_hour = (
        events.groupBy("season", "start_hour")
        .agg(F.count(F.lit(1)).alias("n_events"),
             F.avg("depth_mm").alias("mean_depth_mm"),
             F.avg("duration_min").alias("mean_duration_min"))
        .toPandas()
    )
    write_table(ev_hour, "t21_event_by_hour_season")

    print(f"[task1] done in {time.time() - t0:0.1f}s")
    spark.stop()


if __name__ == "__main__":
    main()
