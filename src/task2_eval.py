"""
Probability-quality metrics computed from compact Spark aggregates.

A prediction frame is reduced, in Spark, to counts per (group, probability bin)
with 600 bins uniform in logit(p). Everything else (PR curve, ROC-AUC, average
precision, reliability diagram, F1-optimal operating point, confusion matrix,
Brier score, log loss) is derived from that small table in pandas, so one pass
over the 10 M-row evaluation set serves all metrics and all subgroup cuts.
Headline AP is cross-checked against MLlib's areaUnderPR in the train script.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
from pyspark.ml.functions import vector_to_array

NB = 600
ZMAX = 16.2
EPS = 1e-7


def prob_col(df, col="probability", out="p"):
    return df.withColumn(out, vector_to_array(F.col(col))[1])


def bin_aggregate(pred, keys):
    """pred needs p, label, plus the key columns. Returns a pandas frame."""
    p = F.least(F.greatest(F.col("p"), F.lit(EPS)), F.lit(1 - EPS))
    z = F.log(p / (1 - p))
    b = F.least(F.greatest(F.floor((z + ZMAX) / (2 * ZMAX) * NB), F.lit(0)), F.lit(NB - 1)).cast("int")
    y = F.col("label")
    out = (pred.withColumn("bin", b).withColumn("pc", p)
           .groupBy(*keys, "bin")
           .agg(F.count(F.lit(1)).alias("n"), F.sum(y).alias("pos"), F.sum("pc").alias("sum_p"),
                F.sum((F.col("pc") - y) ** 2).alias("sum_se"),
                F.sum(-(y * F.log("pc") + (1 - y) * F.log(1 - F.col("pc")))).alias("sum_ll"))
           .toPandas())
    return out


def bin_to_p(b):
    return 1.0 / (1.0 + np.exp(-(-ZMAX + (np.asarray(b) + 0.0) * 2 * ZMAX / NB)))


def _curve(g):
    """PR / ROC points, one per bin, scanning from the highest probability."""
    g = g.sort_values("bin", ascending=False)
    n, pos = g["n"].to_numpy(float), g["pos"].to_numpy(float)
    tp, fp = np.cumsum(pos), np.cumsum(n - pos)
    P, N = pos.sum(), (n - pos).sum()
    return g["bin"].to_numpy(), tp, fp, P, N


def metrics(g, cutoff_bin=None, fixed_f1_floor=1):
    """Metrics for one group (a frame of bin aggregates)."""
    n, pos = g["n"].sum(), g["pos"].sum()
    out = {"n": int(n), "n_pos": int(pos), "prevalence": pos / n,
           "mean_pred": g["sum_p"].sum() / n, "brier": g["sum_se"].sum() / n,
           "logloss": g["sum_ll"].sum() / n}
    brier_ref = (pos / n) * (1 - pos / n)
    out["brier_skill_vs_prevalence"] = 1 - out["brier"] / brier_ref if brier_ref > 0 else np.nan
    if pos == 0 or pos == n:
        return {**out, "pr_auc": np.nan, "roc_auc": np.nan}
    bins, tp, fp, P, N = _curve(g)
    rec, prec = tp / P, tp / (tp + fp)
    out["pr_auc"] = float(np.sum(np.diff(np.r_[0.0, rec]) * prec))          # average precision
    out["roc_auc"] = float(np.trapz(np.r_[0, tp / P], np.r_[0, fp / N]))
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    if cutoff_bin is None:
        cutoff_bin = int(bins[np.argmax(np.where(tp >= fixed_f1_floor, f1, -1))])
    k = int(np.sum(bins >= cutoff_bin))           # bins predicted positive (p >= cutoff)
    tp_k, fp_k = (tp[k - 1], fp[k - 1]) if k > 0 else (0.0, 0.0)
    pr_k = tp_k / (tp_k + fp_k) if tp_k + fp_k > 0 else 0.0
    rc_k = tp_k / P
    out.update(cutoff_bin=int(cutoff_bin), cutoff_p=float(bin_to_p(cutoff_bin)),
               precision=float(pr_k), recall=float(rc_k),
               f1=float(2 * pr_k * rc_k / (pr_k + rc_k)) if pr_k + rc_k > 0 else 0.0,
               tp=int(tp_k), fp=int(fp_k), fn=int(P - tp_k), tn=int(N - fp_k))
    return out


def group_metrics(binned, keys, cutoffs=None):
    """cutoffs: dict {key tuple -> cutoff bin} carried over from validation."""
    rows = []
    for k, g in binned.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        cb = None if cutoffs is None else cutoffs.get(k)
        r = dict(zip(keys, k))
        r.update(metrics(g, cutoff_bin=cb))
        rows.append(r)
    return pd.DataFrame(rows)


def pr_points(g, max_pts=120):
    bins, tp, fp, P, N = _curve(g)
    rec, prec = tp / P, tp / (tp + fp)
    idx = np.unique(np.linspace(0, len(bins) - 1, max_pts).astype(int))
    return pd.DataFrame({"recall": rec[idx], "precision": prec[idx], "cutoff_p": bin_to_p(bins[idx])})


def reliability(g, edges=None):
    """Reliability points: bins merged until each holds >= 1 % of the rows or 30
    positives, so tail points are not noise."""
    g = g.sort_values("bin")
    rows, cn, cp, cs = [], 0, 0, 0
    tot = g["n"].sum()
    for r in g.itertuples():
        cn += r.n
        cp += r.pos
        cs += r.sum_p
        if cp >= 30 and cn >= 0.005 * tot:
            rows.append((cs / cn, cp / cn, cn, cp))
            cn = cp = cs = 0
    if cn > 0:
        if rows:
            m, o, n0, p0 = rows[-1]
            rows[-1] = ((m * n0 + cs) / (n0 + cn), (p0 + cp) / (n0 + cn), n0 + cn, p0 + cp)
        else:
            rows.append((cs / cn, cp / cn, cn, cp))
    return pd.DataFrame(rows, columns=["mean_pred", "obs_freq", "n", "n_pos"])
