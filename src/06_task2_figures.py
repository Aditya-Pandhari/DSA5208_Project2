"""
Stage 6 - Task 2 figures.

Draws five report figures from the small tables written by 05_task2_train.py
(no Spark needed). Every number plotted comes from those tables; the 2024 test
curves use operating points / selection fixed on 2023.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import viz_style as V
from common import FIGURES, TABLES

V.use_style()
ROOT = Path(__file__).resolve().parent.parent
META = json.loads((ROOT / "models" / "task2_exceedance" / "metadata.json").read_text())
SEL = META["selected_model"] + ("+cummin" if META["monotone_postprocess_cummin"] else "")
HEAD = META["headline_thresholds"]
SRC = ("Source: data.gov.sg rainfall, 2017–2024; Spark MLlib. Target: rain accumulated in the 30 min AFTER the "
       "prediction time exceeds the threshold (strict >).\nTrain 2017–2022 · model choice on 2023 · "
       "2024 scored once after selection.")


def T(name):
    return pd.read_csv(TABLES / f"{name}.csv")


def colour_for(models):
    return {m: V.CAT[i % len(V.CAT)] for i, m in enumerate(models)}


def fig11_metrics_by_threshold():
    m = T("t30_task2_final_metrics")
    m = m[m["split"] == "test"]
    models = list(dict.fromkeys(m["model"]))
    col = colour_for(models)
    panels = [("pr_auc", "PR-AUC (average precision)", False), ("roc_auc", "ROC-AUC", False),
              ("brier", "Brier score (lower is better)", True), ("logloss", "Log loss (lower is better)", True)]
    fig, axs = plt.subplots(1, 4, figsize=(14, 4.6))
    for ax, (c, lab, logy) in zip(axs, panels):
        for mod in models:
            g = m[m["model"] == mod].sort_values("threshold_mm")
            ax.plot(g["threshold_mm"], g[c], marker="o", ms=4, lw=2.4 if mod == SEL.replace("+cummin", "") else 1.4,
                    color=col[mod], label=mod)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
        ax.set_xticks(HEAD)
        ax.set_xticklabels([f"{h:g}" for h in HEAD])
        ax.set_xlabel("threshold (mm in next 30 min)")
        ax.set_title(lab, fontsize=9.5, loc="left")
        V.despine(ax)
    axs[0].legend(frameon=False, fontsize=8)
    V.layout(fig, header=1.55, footer=1.0, left=0.7)
    V.titleblock(fig, "Skill is highest for the common, light-rain thresholds and falls as the event gets rarer",
                 "2024 test set, one model per colour. PR-AUC must be read against prevalence, which drops from ≈5% (0.2 mm) to ≈0.1% (20 mm).")
    V.sourcenote(fig, SRC)
    V.save(fig, FIGURES / "fig11_task2_metrics_by_threshold.png")


def fig12_calibration():
    r = T("t31_task2_reliability")
    r = r[(r["split"] == "test") & (r["model"] == SEL)]
    fig, axs = plt.subplots(1, len(HEAD), figsize=(14, 3.6), sharex=False, sharey=False)
    for ax, h in zip(np.atleast_1d(axs), HEAD):
        g = r[(r["threshold_mm"] == h) & (r["n_pos"] > 0)]
        hi = max(g["mean_pred"].max(), g["obs_freq"].max()) * 1.1 if len(g) else 1
        ax.plot([0, hi], [0, hi], color=V.AXIS, lw=1, ls="--")
        ax.plot(g["mean_pred"], g["obs_freq"], marker="o", ms=4, color=V.CAT[0], lw=1.8)
        ax.set_title(f"> {h:g} mm", fontsize=9.5, loc="left")
        ax.set_xlabel("predicted probability")
        V.despine(ax)
    np.atleast_1d(axs)[0].set_ylabel("observed frequency")
    V.layout(fig, header=1.5, footer=1.0, left=0.8)
    V.titleblock(fig, "Predicted probabilities track observed frequencies",
                 f"Reliability on 2024 for the selected model ({SEL}); dashed line = perfect calibration. Equal-count-style bins from the evaluation code.")
    V.sourcenote(fig, SRC)
    V.save(fig, FIGURES / "fig12_task2_calibration.png")


def fig13_pr_curves():
    p = T("t32_task2_pr_curves")
    p = p[(p["split"] == "test") & (p["model"] == SEL)]
    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for i, h in enumerate(HEAD):
        g = p[p["threshold_mm"] == h].sort_values("recall")
        ax.plot(g["recall"], g["precision"], lw=2, color=V.CAT[i % len(V.CAT)], label=f"> {h:g} mm")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, 1)
    ax.legend(frameon=False, title="threshold")
    V.despine(ax)
    V.layout(fig, header=1.45, footer=1.0, left=0.95)
    V.titleblock(fig, "Precision falls quickly as recall is pushed up, most steeply for heavy rain",
                 f"Precision–recall curves on 2024, selected model ({SEL}).")
    V.sourcenote(fig, SRC)
    V.save(fig, FIGURES / "fig13_task2_pr_curves.png")


def fig14_models_and_effects():
    s = T("t34_task2_macro_summary")
    s = s[s["split"] == "test"].set_index("model")
    fe = T("t37_task2_feature_effects")
    fe = fe[fe["coefficient"].notna()].copy()
    fe["a"] = fe["std_effect"].abs()
    fe = fe.sort_values("a").tail(12)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.25]})
    metric = "macro_pr_auc"
    a1.barh(s.index, s[metric], color=[V.CAT[0] if i == SEL.replace("+cummin", "") else V.MUTED for i in s.index])
    a1.set_xlabel("macro-average PR-AUC over thresholds (2024)")
    V.despine(a1)
    a2.barh(fe["feature"], fe["std_effect"], color=[V.CAT[0] if v > 0 else V.RED for v in fe["std_effect"]])
    a2.set_xlabel("logistic effect per 1 SD of the feature (log-odds)")
    V.despine(a2)
    V.layout(fig, header=1.55, footer=1.0, left=2.0)
    V.titleblock(fig, "Recent rain nearby and the threshold itself dominate; model comparison on the left",
                 "Left: macro PR-AUC on the untouched 2024 test set. Right: largest standardized effects of the best logistic model (fit on training years).")
    V.sourcenote(fig, SRC)
    V.save(fig, FIGURES / "fig14_task2_models_and_effects.png")


def fig15_threshold_sweep():
    d = T("t38_task2_threshold_sweep")
    fig, ax = plt.subplots(figsize=(8.6, 5.4))
    for i, (cs, g) in enumerate(d.groupby("case", sort=False)):
        g = g.sort_values("threshold_mm")
        ax.plot(g["threshold_mm"], g["p_final"], lw=2.2, color=V.CAT[i % len(V.CAT)], label=cs)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("rainfall threshold (mm in next 30 min)")
    ax.set_ylabel("P(rain > threshold)")
    ax.legend(frameon=False, fontsize=8, title="recent-rain state at prediction time")
    V.despine(ax)
    st = d["station_id"].iloc[0]
    V.layout(fig, header=1.45, footer=1.0, left=0.95)
    V.titleblock(fig, "The model answers for any threshold, and the probability falls as the threshold rises",
                 f"Station {st}, 2024 afternoon prediction times chosen by recent-rain state. Same station, four conditions, 36+ thresholds.")
    V.sourcenote(fig, SRC)
    V.save(fig, FIGURES / "fig15_task2_threshold_sweep.png")


if __name__ == "__main__":
    V.open_pdf(FIGURES / "task2_all_figures.pdf", title="Task 2 figures")
    fig11_metrics_by_threshold()
    fig12_calibration()
    fig13_pr_curves()
    fig14_models_and_effects()
    fig15_threshold_sweep()
    V.close_pdf()
