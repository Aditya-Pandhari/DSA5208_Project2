"""
Stage 3 - Task 1 figures.

Draws the report figures from the small result tables written by stage 2. No
Spark is needed here, which keeps the plotting loop fast to iterate on.

Every figure follows the same contract: a left-aligned title and deck that state
the finding (not the variable names), recessive chrome, a single encoding channel
per question, and a source note recording the provenance and the QC rule in
force. Sequential magnitude uses one blue hue; polarity uses a red<->grey<->blue
diverging ramp; categorical identity uses a fixed, CVD-validated slot order.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import viz_style as V
from common import FIGURES, TABLES

V.use_style()

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
SRC = ("Source: data.gov.sg, Historical Rainfall across Singapore, 2017–2024 "
       "(48.5 M five-minute gauge readings, 91 stations). Processed with Apache Spark.")


def T(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES / f"{name}.csv")


def hhmm(minutes: int) -> str:
    return f"{int(minutes) // 60:02d}:{int(minutes) % 60:02d}"


def pad_ylim(ax, below: float = 0.13, above: float = 0.09) -> None:
    """Open up room under a trough and over a peak so that a value label sits
    inside the plot instead of running into the tick labels above or below."""
    lo, hi = ax.get_ylim()
    span = hi - lo
    ax.set_ylim(lo - below * span, hi + above * span)


# =========================================================================
# F01 - the data foundation
# =========================================================================
def fig01_coverage() -> None:
    cov = T("t03_station_year_coverage")
    net = T("t04_daily_network")
    net["date"] = pd.to_datetime(net["date"])

    mat = cov.pivot(index="station_id", columns="year", values="coverage")
    # order stations by first year of service, then by how long they served
    first = mat.apply(lambda r: r.first_valid_index(), axis=1)
    mat = mat.loc[sorted(mat.index, key=lambda s: (first[s], -mat.loc[s].count()))]

    fig = plt.figure(figsize=(13.2, 10.3))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 2.25], width_ratios=[2.6, 1.0],
                          hspace=0.30, wspace=0.26)

    # -- (a) daily network size -------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(net["date"], 0, net["stations_reporting"], color=V.BLUE_RAMP[1],
                    lw=0, zorder=2, label="Reporting at all")
    ax.fill_between(net["date"], 0, net["stations_usable"], color=V.BLUE,
                    lw=0, zorder=3, label="Usable ($\\geq$80% of 288 daily slots)")
    gaps = net[net["stations_reporting"] == 0]
    for d in gaps["date"]:
        ax.axvline(d, color=V.RED, lw=1.1, alpha=0.85, zorder=4)
    ax.set_ylim(0, 88)
    ax.set_ylabel("Gauges reporting")
    ax.set_title("a   The observing network is not constant: it grows, and it drops out",
                 pad=34)
    ax.set_xlim(net["date"].min(), net["date"].max())
    ax.grid(axis="x", visible=False)
    V.despine(ax)
    ax.annotate("August 2018\n28 of 31 days absent", xy=(pd.Timestamp("2018-08-15"), 58),
                xytext=(pd.Timestamp("2018-09-25"), 72), fontsize=8.2, color=V.RED,
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.RED, lw=0.9,
                                connectionstyle="angle3,angleA=0,angleB=80"))
    ax.annotate("network expansion\nmid-2020: +25 gauges",
                xy=(pd.Timestamp("2020-07-15"), 77), xytext=(pd.Timestamp("2021-06-01"), 84),
                fontsize=8.2, color=V.INK_2, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.MUTED, lw=0.9,
                                connectionstyle="angle3,angleA=0,angleB=70"))
    handles = ax.get_legend_handles_labels()[0] + [
        Line2D([0], [0], color=V.RED, lw=1.4, label="No data anywhere on the island")]
    ax.legend(handles=handles, loc="lower left", ncol=3,
              bbox_to_anchor=(0, 1.06), fontsize=8.4)

    # -- (b) station x year coverage heatmap -------------------------------
    ax = fig.add_subplot(gs[1, 0])
    im = ax.imshow(mat.values, aspect="auto", cmap=V.CMAP_RAIN, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks(range(mat.shape[1]), mat.columns)
    ax.set_yticks([])
    ax.set_ylabel(f"{mat.shape[0]} stations  (ordered by year of entry)")
    ax.set_title("b   Per-station annual completeness")
    ax.grid(visible=False)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.03)
    cb.set_label("Completeness", fontsize=8.2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)

    # -- (c) how many station-years survive each threshold -----------------
    ax = fig.add_subplot(gs[1, 1])
    ths = np.arange(0.5, 1.001, 0.05)
    counts = [(cov["coverage"] >= t).sum() for t in ths]
    ax.plot(ths * 100, counts, color=V.BLUE, marker="o", markersize=4,
            markerfacecolor=V.SURFACE, markeredgewidth=1.6, zorder=3)
    ax.axvline(80, color=V.MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.annotate(f"retained: {(cov['coverage'] >= 0.8).sum()} of {len(cov)}\nstation-years",
                xy=(80, (cov["coverage"] >= 0.8).sum()), xytext=(60, 200),
                fontsize=8.2, color=V.INK_2,
                arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.MUTED, lw=0.9))
    ax.set_xlabel("Completeness threshold (%)")
    ax.set_ylabel("Station-years retained")
    ax.set_title("c   Cost of the QC rule")
    V.despine(ax)
    ax.grid(axis="x", visible=False)

    V.layout(fig, header=2.15, footer=1.00, left=1.05)
    V.titleblock(fig,
                 "A changing gauge network, not a changing climate, is the first thing the raw data shows",
                 "Before any weather signal can be read, the record has to be audited. The network triples in places and vanishes in others, so\n"
                 "island-wide figures in this report are areal means over gauges that pass a completeness rule, never raw averages over rows.")
    V.sourcenote(fig, SRC + "\nQC rule: a station-day needs ≥230 of 288 five-minute slots; an island-day needs ≥20 such stations.")
    V.save(fig, FIGURES / "fig01_data_foundation.png")


# =========================================================================
# F02 - annual totals
# =========================================================================
def fig02_annual() -> None:
    a = T("t07_annual_summary")
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 6.7),
                             gridspec_kw=dict(width_ratios=[1.55, 1, 1], wspace=0.28))
    x = np.arange(len(a))

    ax = axes[0]
    ax.bar(x, a["gapfilled_total_mm"], color=V.BLUE_RAMP[2], width=0.68,
           zorder=2, label="Gap-filled estimate")
    ax.bar(x, a["observed_total_mm"], color=V.BLUE, width=0.68, zorder=3,
           label="Observed days only")
    mean = a["gapfilled_total_mm"].mean()
    ax.axhline(mean, color=V.INK_2, lw=1.1, ls=(0, (4, 3)), zorder=4)
    ax.text(len(a) - 0.32, mean, f"8-year mean\n{mean:,.0f} mm", ha="left",
            va="center", fontsize=8.2, color=V.INK_2)
    for i, r in a.iterrows():
        ax.text(i, r["gapfilled_total_mm"] + 60, f"{r['gapfilled_total_mm']:,.0f}",
                ha="center", fontsize=8.2, color=V.INK,
                fontweight="semibold" if r["year"] in (2019, 2022) else "normal")
        if r["days_missing"] > 20:
            ax.text(i, 130, f"{int(r['days_missing'])}\ndays\nmissing", ha="center",
                    va="bottom", fontsize=7.0, color="white", zorder=5,
                    linespacing=1.35)
    ax.set_xticks(x, a["year"])
    ax.set_ylim(0, 3450)
    ax.set_xlim(-0.65, len(a) + 0.45)
    ax.set_ylabel("Island areal rainfall (mm)")
    ax.set_title("a   Annual rainfall varies by 63% across eight years", pad=32)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=8.4)
    V.despine(ax)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    ax.plot(x, a["wet_days_scaled"], color=V.ORANGE, marker="o", markersize=5,
            markerfacecolor=V.SURFACE, markeredgewidth=1.8, zorder=3)
    for i, v in enumerate(a["wet_days_scaled"]):
        if a["year"][i] in (2019, 2022):
            high = a["year"][i] == 2022          # 2022 is a peak, 2019 a trough
            ax.annotate(f"{v:.0f}", (i, v), textcoords="offset points",
                        xytext=(0, 11 if high else -19), ha="center",
                        va="bottom" if high else "top", fontsize=8.2,
                        color=V.INK, fontweight="semibold")
    ax.set_xticks(x, a["year"])
    ax.set_ylabel("Wet days per year ($\\geq$1 mm)")
    ax.set_title("b   Frequency: how often it rains")
    V.despine(ax)
    ax.grid(axis="x", visible=False)
    pad_ylim(ax)

    ax = axes[2]
    intensity = a["gapfilled_total_mm"] / a["wet_days_scaled"]
    ax.plot(x, intensity, color=V.AQUA, marker="o", markersize=5,
            markerfacecolor=V.SURFACE, markeredgewidth=1.8, zorder=3)
    for i, v in enumerate(intensity):
        if a["year"][i] in (2019, 2022):
            # 2019 is a trough; 2022 sits mid-slope, so its label goes out to
            # the right where the line has already passed
            below = a["year"][i] == 2019
            ax.annotate(f"{v:.1f}", (i, v), textcoords="offset points",
                        xytext=(0 if below else 11, -19 if below else 0),
                        ha="center" if below else "left",
                        va="top" if below else "center", fontsize=8.2,
                        color=V.INK, fontweight="semibold")
    ax.set_xticks(x, a["year"])
    ax.set_ylabel("mm per wet day")
    ax.set_title("c   Intensity: how hard it rains")
    V.despine(ax)
    ax.grid(axis="x", visible=False)
    pad_ylim(ax)

    V.layout(fig, header=2.25, footer=1.00, left=1.05)
    V.titleblock(fig,
                 "2019 was dry because it rained less often, not because the rain got lighter",
                 "Splitting the annual total into frequency \u00d7 intensity separates the two ways a year can be dry. The 2019 deficit "
                 "(1,775 mm, a positive-IOD\ndrought year) is a collapse in wet-day count, 172 days against a 225-day norm, while the rain that did fall was no "
                 "lighter than 2017's.\nPanel c carries a separate signal worth noting: per-wet-day intensity climbs steadily after 2019, so the network is seeing "
                 "fewer but heavier\nwet days in recent years. Panels b and c are drawn on their own axes rather than twinned onto panel a.")
    V.sourcenote(fig, SRC + "\nWet-day counts are rescaled to a 365-day year so that years with missing days stay comparable.")
    V.save(fig, FIGURES / "fig02_annual_totals.png")


# =========================================================================
# F03 - the seasonal cycle
# =========================================================================
def fig03_seasonal() -> None:
    m = T("t08_monthly")
    grid = m.pivot(index="month", columns="year", values="total_mm")
    clim = m.groupby("month")["total_mm"].mean()

    fig = plt.figure(figsize=(13.2, 8.1))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1.05], wspace=0.2)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(grid.values, aspect="auto", cmap=V.CMAP_RAIN, vmin=0, vmax=430,
                   interpolation="nearest")
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(12), MONTHS)
    ax.grid(visible=False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(12):
        for j in range(grid.shape[1]):
            v = grid.values[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.9,
                    color="white" if v > 240 else V.INK_2)
    # mark months whose source record is incomplete
    inc = m[m["complete"] == 0]
    for _, r in inc.iterrows():
        ax.add_patch(Rectangle((list(grid.columns).index(r["year"]) - 0.5,
                                r["month"] - 1.5), 1, 1, fill=False,
                               edgecolor=V.RED, lw=1.6, zorder=5))
    ax.set_title("a   Monthly island rainfall, 2017–2024 (mm)")
    cb = fig.colorbar(im, ax=ax, pad=0.012, fraction=0.028)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("mm per month", fontsize=8.2)

    ax = fig.add_subplot(gs[0, 1])
    y = np.arange(12)
    ax.barh(y, clim.values, color=V.BLUE_RAMP[5], height=0.66, zorder=2)
    lo = grid.min(axis=1).values
    hi = grid.max(axis=1).values
    ax.hlines(y, lo, hi, color=V.INK_2, lw=1.2, zorder=3)
    ax.plot(lo, y, "|", color=V.INK_2, markersize=7, markeredgewidth=1.2, zorder=3)
    ax.plot(hi, y, "|", color=V.INK_2, markersize=7, markeredgewidth=1.2, zorder=3)
    # values as a right-hand column so the range whiskers never cross them
    for i, v in enumerate(clim.values):
        ax.text(505, i, f"{v:.0f}", va="center", ha="right", fontsize=8.2,
                color=V.INK_2, zorder=4)
    ax.set_yticks(y, MONTHS)
    ax.invert_yaxis()
    ax.set_xlim(0, 510)
    ax.set_xlabel("mm")
    ax.set_title("b   Climatology (bars) and the 2017\u20132024 range (whiskers)")
    V.despine(ax)
    ax.grid(axis="y", visible=False)

    V.layout(fig, header=2.05, footer=1.00, left=0.8)
    V.titleblock(fig,
                 "A double wet season, and a year-to-year spread that swamps it",
                 "November–January (north-east monsoon) and April–May (inter-monsoon) are the wet peaks; February and July are the dry "
                 "troughs. But the\nsame calendar month ranges over a factor of ten between years. February spans 32 mm (2021) to 352 mm (2023), "
                 "Any monthly\nclimatology for Singapore is therefore a weak predictor of any individual month.")
    V.sourcenote(fig, SRC + "\nRed outline: month whose source record is <90% complete, filled from the day-of-year climatology of the other years.")
    V.save(fig, FIGURES / "fig03_seasonal_cycle.png")


# =========================================================================
# F04 - monthly anomalies through time
# =========================================================================
def fig04_anomaly() -> None:
    m = T("t08_monthly").sort_values(["year", "month"]).reset_index(drop=True)
    m["t"] = pd.to_datetime(dict(year=m["year"], month=m["month"], day=15))
    v = m["anomaly_mm"].to_numpy()
    lim = np.abs(v).max()
    colors = V.CMAP_ANOM((v / (2 * lim)) + 0.5)

    fig, axes = plt.subplots(2, 1, figsize=(13.2, 7.9), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.6, 1], hspace=0.18))

    ax = axes[0]
    ax.bar(m["t"], v, width=24, color=colors, zorder=3)
    ax.axhline(0, color=V.AXIS, lw=1, zorder=4)
    ax.set_ylabel("Departure from the\n2017–2024 monthly mean (mm)")
    ax.set_title("a   Monthly rainfall anomaly")
    V.despine(ax, bottom=True)
    ax.grid(axis="x", visible=False)
    ax.set_xlim(m["t"].min() - pd.Timedelta(days=45),
                m["t"].max() + pd.Timedelta(days=45))

    # measure the longest unbroken run of below-average months rather than
    # asserting a round number in the caption
    neg = (v < 0).astype(int)
    best = run = 0
    best_end = 0
    for i, flag in enumerate(neg):
        run = run + 1 if flag else 0
        if run > best:
            best, best_end = run, i
    run_start, run_end = m["t"].iloc[best_end - best + 1], m["t"].iloc[best_end]
    for lbl, xy, xytext in [
        (f"the 2019 positive-IOD drought:\n{best} consecutive below-average months\n"
         f"({run_start:%b %Y} \u2013 {run_end:%b %Y})",
         (pd.Timestamp("2019-06-15"), -60), (pd.Timestamp("2018-08-01"), 128)),
        ("Aug 2021, the wettest month\nin the record (+231 mm)",
         (pd.Timestamp("2021-08-15"), 228), (pd.Timestamp("2019-12-01"), 196)),
        ("Feb 2023 (+208 mm)",
         (pd.Timestamp("2023-02-15"), 212), (pd.Timestamp("2023-06-10"), 243)),
    ]:
        ax.annotate(lbl, xy=xy, xytext=xytext, fontsize=8.2, color=V.INK_2,
                    ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.MUTED, lw=0.9,
                                    connectionstyle="angle3,angleA=0,angleB=60"))

    ax = axes[1]
    roll = m.set_index("t")["total_mm"].rolling(12, min_periods=9).sum()
    ax.plot(roll.index, roll.values, color=V.BLUE, zorder=3)
    ax.axhline(T("t07_annual_summary")["gapfilled_total_mm"].mean(), color=V.INK_2,
               lw=1.1, ls=(0, (4, 3)), zorder=2)
    ax.fill_between(roll.index, roll.values,
                    T("t07_annual_summary")["gapfilled_total_mm"].mean(),
                    where=roll.values < T("t07_annual_summary")["gapfilled_total_mm"].mean(),
                    color=V.RED, alpha=0.13, zorder=1, interpolate=True)
    ax.set_ylabel("12-month running\ntotal (mm)")
    ax.set_title("b   The same signal as a running annual total")
    V.despine(ax)
    ax.grid(axis="x", visible=False)

    V.layout(fig, header=2.05, footer=1.00, left=1.25)
    V.titleblock(fig,
                 "Wet and dry spells in Singapore run for seasons, not for months",
                 f"Anomalies are strongly autocorrelated: the 2019 drought runs {best} consecutive below-average months, and the 2021–2022 "
                 "La Niña\nperiod delivers a sustained surplus. This persistence is what makes the year-to-year spread in Figure 3 so large. It is "
                 "also a reason\na short-horizon forecast (Task 2) should be given recent-rainfall state, not just a calendar position.")
    V.sourcenote(fig, SRC + "\nAnomalies are computed against each calendar month's own 2017–2024 mean, from the gap-filled island daily series.")
    V.save(fig, FIGURES / "fig04_monthly_anomaly.png")


# =========================================================================
# F05 - the diurnal cycle  (headline scientific figure)
# =========================================================================
def fig05_diurnal() -> None:
    d = T("t10_diurnal_month_year")
    grid = (d.groupby(["hh_min", "month"])["mm_per_hour"].mean()
            .unstack("month").sort_index())

    fig = plt.figure(figsize=(13.2, 8.8))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.45, 1.1], wspace=0.24)

    # -- (a) hour x month heatmap -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    im = ax.pcolormesh(np.arange(13), np.arange(49) * 0.5, grid.values,
                       cmap=V.CMAP_RAIN, vmin=0, vmax=1.5, shading="flat")
    ax.set_xticks(np.arange(12) + 0.5, MONTHS)
    ax.set_yticks(np.arange(0, 25, 3), [f"{h:02d}:00" for h in range(0, 25, 3)])
    ax.set_ylim(0, 24)
    ax.invert_yaxis()
    ax.grid(visible=False)
    for s in ax.spines.values():
        s.set_visible(False)
    # trace the daily maximum
    peak = grid.idxmax(axis=0) / 60.0
    ax.plot(np.arange(12) + 0.5, peak.values, color=V.ORANGE, lw=2.2,
            marker="o", markersize=5, markerfacecolor=V.SURFACE,
            markeredgewidth=1.8, zorder=5)
    ax.annotate("hour of the\ndaily maximum", xy=(1.5, peak.values[1]),
                xytext=(0.30, 21.0), fontsize=8.2, color=V.ORANGE,
                ha="left", va="center", fontweight="semibold", zorder=6,
                arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.ORANGE, lw=0.9,
                                connectionstyle="angle3,angleA=10,angleB=-80"))
    ax.set_title("a   Mean rainfall rate by time of day and month (mm/h)")
    ax.annotate("morning regime\n(Sumatra squalls)", xy=(6.5, 9.5), xytext=(2.4, 6.2),
                fontsize=8.4, color=V.INK, ha="left", va="center", fontweight="semibold",
                arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.INK_2, lw=1.0,
                                connectionstyle="angle3,angleA=0,angleB=60"))
    ax.annotate("afternoon regime\n(local convection)", xy=(10.6, 15.6), xytext=(7.4, 20.4),
                fontsize=8.4, color=V.INK, ha="left", va="center", fontweight="semibold",
                arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.INK_2, lw=1.0,
                                connectionstyle="angle3,angleA=0,angleB=60"))
    cb = fig.colorbar(im, ax=ax, pad=0.014, fraction=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("mm/h", fontsize=8.2)

    # -- (b) seasonal diurnal profiles ------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ds = d.groupby(["season", "hh_min"])["mm_per_hour"].mean().reset_index()
    for name in V.SEASON_ORDER:
        s = ds[ds["season"] == name].sort_values("hh_min")
        pk = s.loc[s["mm_per_hour"].idxmax()]
        ax.plot(s["hh_min"] / 60.0, s["mm_per_hour"], color=V.SEASON_COLOR[name],
                zorder=3, label=f"{name}   peak {hhmm(pk['hh_min'])}")
        ax.plot(pk["hh_min"] / 60, pk["mm_per_hour"], "o", color=V.SEASON_COLOR[name],
                markersize=6, markeredgecolor=V.SURFACE, markeredgewidth=1.6, zorder=4)
    ax.legend(loc="upper left", fontsize=8.2, bbox_to_anchor=(-0.01, 1.01),
              labelcolor="linecolor")
    ax.annotate("the south-west monsoon is the only\nseason with a morning maximum",
                xy=(10.0, 0.56), xytext=(3.9, 0.85), fontsize=8.3, color=V.AQUA,
                ha="left", va="center", fontweight="semibold", zorder=6,
                arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.AQUA, lw=0.9,
                                connectionstyle="angle3,angleA=0,angleB=70"))
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 25, 3), [f"{h:02d}" for h in range(0, 25, 3)])
    ax.set_xlabel("Hour of day (SGT)")
    ax.set_ylabel("Mean rainfall rate (mm/h)")
    ax.set_title("b   The clock changes with the monsoon")
    V.despine(ax)
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.45)

    V.layout(fig, header=2.1, footer=1.00, left=0.95)
    V.titleblock(fig,
                 "Singapore has two rainfall clocks, and the monsoon decides which one runs",
                 "From June to September the south-west monsoon shifts the daily rainfall maximum into the morning. This is the signature of "
                 "Sumatra squall\nlines that form over the Malacca Strait overnight and sweep east across the island at dawn. In every other season "
                 "the maximum is a\nsharp mid-afternoon convective peak near 15:00–16:00. The peak hour swings by roughly six hours between "
                 "the two regimes.")
    V.sourcenote(fig, SRC + "\nRates are means over all stations and all 5-minute slots in each half-hour bin (dry slots included), so panel b is a true expected rate.")
    V.save(fig, FIGURES / "fig05_diurnal_cycle.png")


# =========================================================================
# F06 - intensity distribution and concentration
# =========================================================================
def fig06_intensity() -> None:
    e = T("t12_intensity_exceedance")
    c = T("t13_rain_concentration")
    years = sorted(e["year"].unique())
    ramp = [V.BLUE_RAMP[i] for i in np.linspace(1, 12, len(years)).astype(int)]

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.3),
                             gridspec_kw=dict(wspace=0.24))

    ax = axes[0]
    for yr, col in zip(years, ramp):
        s = e[e["year"] == yr].sort_values("threshold_mm")
        ax.plot(s["threshold_mm"], s["exceed_prob"], color=col, lw=1.9, zorder=3)
    for yr, dy in [(2019, -0.3), (2022, 0.4)]:
        s = e[e["year"] == yr].sort_values("threshold_mm")
        col = ramp[years.index(yr)]
        ax.annotate(str(yr), xy=(s["threshold_mm"].iloc[6], s["exceed_prob"].iloc[6]),
                    xytext=(10, 12 * (1 if dy > 0 else -2)), textcoords="offset points",
                    fontsize=8.6, color=col, fontweight="semibold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("5-minute rainfall depth, $x$ (mm)")
    ax.set_ylabel("P(depth $\\geq x$) over all 5-minute slots")
    ax.set_title("a   Intensity is heavy-tailed, and the whole curve moves with the year")
    V.despine(ax)
    ax.grid(which="both", axis="x", visible=False)
    sm = plt.cm.ScalarMappable(cmap=V.CMAP_RAIN,
                               norm=plt.Normalize(years[0], years[-1]))
    cb = fig.colorbar(sm, ax=ax, pad=0.012, fraction=0.03, ticks=years)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)

    ax = axes[1]
    for yr, col in zip(years, ramp):
        s = c[c["year"] == yr].sort_values("frac_of_wet_intervals")
        ax.plot(100 * (1 - s["frac_of_wet_intervals"]),
                100 * (1 - s["frac_of_rainfall"]), color=col, lw=1.9, zorder=3)
    ax.plot([0, 100], [0, 100], color=V.MUTED, lw=1.1, ls=(0, (3, 3)), zorder=2)
    ax.text(62, 70, "if every wet interval\ndelivered the same depth", fontsize=8,
            color=V.MUTED, rotation=33, ha="center", va="center", rotation_mode="anchor")
    s = c[c["year"] == 2022].set_index("frac_of_wet_intervals")["frac_of_rainfall"]
    # the two call-outs are placed by hand: the 1 % point sits on the diagonal
    # reference line, so its label goes into the empty region beneath it
    for q, lab, tx, ty in [(0.99, "1%", 24, 7), (0.90, "10%", 17, 56)]:
        xx, yy = 100 * (1 - q), 100 * (1 - s.loc[q])
        ax.plot([xx, xx], [0, yy], color=V.AXIS, lw=0.9, zorder=1)
        ax.plot([0, xx], [yy, yy], color=V.AXIS, lw=0.9, zorder=1)
        ax.plot(xx, yy, "o", color=V.ORANGE, markersize=6,
                markeredgecolor=V.SURFACE, markeredgewidth=1.6, zorder=5)
        ax.annotate(f"the wettest {lab} of wet intervals\ncarry {yy:.0f}% of all rainfall",
                    xy=(xx, yy), xytext=(tx, ty), fontsize=8.4, color=V.INK,
                    fontweight="semibold", ha="left", va="center", zorder=6,
                    arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.INK_2, lw=0.9,
                                    connectionstyle="angle3,angleA=0,angleB=75"))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Wettest % of wet 5-minute intervals")
    ax.set_ylabel("% of the year's total rainfall they deliver")
    ax.set_title("b   Rainfall is extremely concentrated in time")
    V.despine(ax)

    V.layout(fig, header=1.95, footer=1.00, left=1.1)
    V.titleblock(fig,
                 "Almost half the rain that falls on Singapore arrives in 10% of the wet minutes",
                 "The concentration curve is near-identical in every year, including the drought: a dry year is not a year of gentler rain, it is "
                 "a year with\nfewer rain events. The heavy tail in panel a is why a rainfall model should be framed as an exceedance probability "
                 "rather than a\nconditional-mean regression, which is exactly how Task 2 is set up.")
    V.sourcenote(fig, SRC + "\n“Wet” = a 5-minute slot recording ≥0.2 mm (one tipping-bucket resolution step).")
    V.save(fig, FIGURES / "fig06_intensity.png")


# =========================================================================
# F07 - spatial climatology
# =========================================================================
COMMON = [2021, 2022, 2023, 2024]


def common_period_stations() -> pd.DataFrame:
    sy = T("t17_station_year_totals")
    st = T("t02_stations")
    sub = sy[sy["year"].isin(COMMON)]
    ok = sub.groupby("station_id").size()
    ok = ok[ok == len(COMMON)].index
    out = (sub[sub["station_id"].isin(ok)]
           .groupby("station_id")["mean_daily_mm"].mean()
           .mul(365.25).rename("mean_annual_mm").reset_index()
           .merge(st, on="station_id"))
    return out


def fig07_spatial() -> None:
    s = common_period_stations()
    fig = plt.figure(figsize=(13.2, 7.9))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1], wspace=0.22)

    # -- (a) interpolated rainfall surface + gauges ------------------------
    ax = fig.add_subplot(gs[0, 0])
    lon, lat, val = (s["lon"].to_numpy(), s["lat"].to_numpy(),
                     s["mean_annual_mm"].to_numpy())
    tri = mtri.Triangulation(lon, lat)
    interp = mtri.LinearTriInterpolator(tri, val)
    gx, gy = np.meshgrid(np.linspace(lon.min(), lon.max(), 500),
                         np.linspace(lat.min(), lat.max(), 500))
    gz = interp(gx, gy)
    ax.contourf(gx, gy, gz, levels=np.arange(1600, 3700, 100), cmap=V.CMAP_RAIN,
                extend="both", zorder=1)
    cs = ax.contour(gx, gy, gz, levels=np.arange(2000, 3600, 400),
                    colors=[V.INK_2], linewidths=0.6, alpha=0.45, zorder=2)
    ax.clabel(cs, fmt="%.0f", fontsize=7, colors=V.INK_2)
    sc = ax.scatter(lon, lat, c=val, cmap=V.CMAP_RAIN, vmin=1600, vmax=3600,
                    s=44, edgecolor=V.SURFACE, linewidth=1.3, zorder=4)

    # label the extremes, pushing each label away from the plot edge it sits on
    marks = pd.concat([s.nlargest(2, "mean_annual_mm"), s.nsmallest(2, "mean_annual_mm")])
    for k, (_, r) in enumerate(marks.iterrows()):
        east = r["lon"] > np.median(lon)
        dy = 15 if k % 2 == 0 else -19          # stagger so adjacent gauges never overlap
        ax.annotate(f"{r['station_name']}  {r['mean_annual_mm']:,.0f} mm",
                    xy=(r["lon"], r["lat"]),
                    xytext=(-13 if east else 13, dy), textcoords="offset points",
                    ha="right" if east else "left", fontsize=8,
                    color=V.INK, fontweight="semibold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.3", fc=V.SURFACE, ec="none",
                              alpha=0.88),
                    arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.INK_2, lw=0.8))
    padx, pady = 0.012, 0.008
    ax.set_xlim(lon.min() - padx, lon.max() + padx)
    ax.set_ylim(lat.min() - pady, lat.max() + pady)
    ax.set_aspect(1 / np.cos(np.radians(1.35)))
    ax.set_xlabel("Longitude (\u00b0E)")
    ax.set_ylabel("Latitude (\u00b0N)")
    ax.set_title(f"a   Mean annual rainfall, {COMMON[0]}\u2013{COMMON[-1]} "
                 f"({len(s)} gauges with complete records)")
    ax.grid(color=V.GRID, lw=0.5)
    cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.115,
                      fraction=0.038, aspect=44)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("Mean annual rainfall (mm)", fontsize=8.2)

    # -- (b) the west-east gradient ---------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(lon, val, s=42, color=V.BLUE, edgecolor=V.SURFACE, linewidth=1.2,
               zorder=3)
    slope, intercept = np.polyfit(lon, val, 1)
    xs = np.linspace(lon.min(), lon.max(), 50)
    ax.plot(xs, intercept + slope * xs, color=V.ORANGE, lw=2, zorder=4)
    r = np.corrcoef(lon, val)[0, 1]
    km_per_deg = 111.32 * np.cos(np.radians(float(np.mean(lat))))
    per_10km = slope * 10.0 / km_per_deg          # mm per 10 km of easting
    ax.annotate(f"$r$ = {r:.2f}\n{-per_10km:,.0f} mm of annual rainfall lost\n"
                f"for every 10 km travelled east",
                xy=(0.97, 0.96), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.6, color=V.INK, fontweight="semibold")
    ax.set_xlabel("Longitude (\u00b0E),  west $\\rightarrow$ east")
    ax.set_ylabel("Mean annual rainfall (mm)")
    ax.set_title("b   A steep west\u2013east gradient")
    V.despine(ax)

    V.layout(fig, header=2.10, footer=1.42, left=0.95)
    V.titleblock(fig,
                 "Rainfall more than doubles across an island 50 km wide",
                 f"Over the common {COMMON[0]}\u2013{COMMON[-1]} period the wettest gauge records {val.max():,.0f} mm a year "
                 f"and the driest {val.min():,.0f} mm, a factor of {val.max() / val.min():.1f}.\n"
                 "The wet core sits over the western and central catchments, where sea-breeze convergence and higher terrain favour convection;\n"
                 "the east and the offshore islands are markedly drier. Station identity therefore carries real predictive information, "
                 "and Task 2 keeps it as a feature.")
    V.sourcenote(fig, SRC + "\nA fixed four-year window is used so that gauges commissioned at different times are never compared over different periods. "
                            "The field in panel a is a linear\ninterpolation across the gauge network and is not extrapolated beyond the convex hull of the gauges.")
    V.save(fig, FIGURES / "fig07_spatial_climatology.png")


# =========================================================================
# F08 - spatial coherence
# =========================================================================
def fig08_coherence() -> None:
    p = T("t18_pair_correlation").dropna()
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 7.1),
                             gridspec_kw=dict(width_ratios=[1.35, 1], wspace=0.24))

    bins = np.arange(0, 42, 2)
    p["bin"] = pd.cut(p["distance_km"], bins)
    binned = p.groupby("bin", observed=True)["corr"].mean()
    centres = np.array([iv.mid for iv in binned.index])

    # A plain exponential fits this correlogram badly (it is dominated by the
    # many long-range pairs and sits above the data from 3 to 30 km). A
    # stretched exponential, the standard form for convective rainfall fields,
    # reduces the RMSE against the binned means by a factor of five.
    d, r = p["distance_km"].to_numpy(), p["corr"].to_numpy()
    m = (r > 0.01) & (r < 0.999) & (d > 0)
    shape, c = np.polyfit(np.log(d[m]), np.log(-np.log(r[m])), 1)
    d0 = np.exp(-c / shape)
    # model-free decorrelation distance: where the binned mean crosses 1/e
    d_emp = float(np.interp(np.exp(-1), binned.values[::-1], centres[::-1]))

    ax = axes[0]
    ax.scatter(d, r, s=9, color=V.BLUE, alpha=0.22, linewidth=0, zorder=2)
    ax.plot(centres, binned.values, color=V.INK, lw=2.2, marker="o", markersize=5,
            markerfacecolor=V.SURFACE, markeredgewidth=1.7, zorder=4,
            label="Binned mean")
    xs = np.linspace(0.2, 40, 300)
    ax.plot(xs, np.exp(-(xs / d0) ** shape), color=V.ORANGE, lw=2, ls=(0, (5, 2)),
            zorder=5,
            label=f"$\\rho = e^{{-(d/d_0)^s}}$,  $d_0$ = {d0:.1f} km, $s$ = {shape:.2f}")
    ax.axhline(np.exp(-1), color=V.MUTED, lw=1, ls=(0, (2, 3)), zorder=1)
    ax.text(39.5, np.exp(-1) + 0.022, "$1/e$", ha="right", fontsize=8, color=V.MUTED)
    ax.plot([d_emp, d_emp], [0, np.exp(-1)], color=V.MUTED, lw=1, ls=(0, (2, 3)),
            zorder=1)
    ax.annotate(f"correlation falls to $1/e$\nat {d_emp:.1f} km",
                xy=(d_emp, np.exp(-1)), xytext=(d_emp + 4.5, 0.60),
                fontsize=8.4, color=V.INK, fontweight="semibold", zorder=6,
                arrowprops=dict(arrowstyle="-", shrinkA=4, shrinkB=9, color=V.INK_2, lw=0.9,
                                connectionstyle="angle3,angleA=0,angleB=70"))
    ax.set_xlim(0, 40)
    ax.set_ylim(-0.03, 1.0)
    ax.set_xlabel("Distance between gauges (km)")
    ax.set_ylabel("Correlation of 30-minute rainfall totals")
    ax.set_title("a   Rainfall decorrelates within a few kilometres")
    ax.legend(loc="upper right", fontsize=8.4)
    V.despine(ax)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    rows = []
    for lo, hi in [(0, 2), (2, 5), (5, 10), (10, 20), (20, 40)]:
        sel = p[(p["distance_km"] >= lo) & (p["distance_km"] < hi)]
        rows.append((f"{lo}\u2013{hi} km", sel["corr"].mean(), len(sel)))
    ypos = np.arange(len(rows))
    val = [x[1] for x in rows]
    ax.barh(ypos, val, color=V.BLUE_RAMP[6], height=0.62, zorder=2)
    for i, (v, n) in enumerate(zip(val, [x[2] for x in rows])):
        ax.text(v + 0.015, i, f"{v:.2f}", va="center", fontsize=8.6,
                color=V.INK, fontweight="semibold")
        ax.text(0.012, i, f"n = {n:,}", va="center", fontsize=7.6, color="white")
    ax.set_yticks(ypos, [x[0] for x in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Mean pairwise correlation")
    ax.set_title("b   Mean correlation by separation")
    V.despine(ax)
    ax.grid(axis="y", visible=False)

    V.layout(fig, header=2.10, footer=1.32, left=1.05)
    V.titleblock(fig,
                 f"Singapore rainfall loses coherence over about {d_emp:.0f} km, a fraction of the island's width",
                 "Gauges less than 2 km apart agree closely on their 30-minute totals ($\\rho$ = 0.84); beyond 20 km they are nearly independent "
                 "($\\rho$ = 0.15).\nThat is the fingerprint of convective, cell-scale rainfall rather than broad frontal systems, and it carries a "
                 "concrete modelling consequence:\na single island-wide forecast cannot be right everywhere at once, so Task 2 predicts per station "
                 "and draws on neighbouring gauges only at short range.")
    V.sourcenote(fig, SRC + f"\n{len(p):,} station pairs, Pearson correlation of concurrent 30-minute totals over 2017\u20132024 "
                            f"(66 gauges with \u22654 complete years; pairwise-complete observations, \u22655,000 overlapping\nhalf-hours per pair). "
                            f"The {d_emp:.1f} km figure is read off the binned means and assumes no functional form.")
    V.save(fig, FIGURES / "fig08_spatial_coherence.png")


# =========================================================================
# F09 - rainfall events
# =========================================================================
def fig09_events() -> None:
    ev = T("t20_event_sample")
    eh = T("t21_event_by_hour_season")
    ea = T("t19_event_summary_annual")

    fig = plt.figure(figsize=(13.2, 7.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])
    hb = ax.hexbin(ev["duration_min"], ev["depth_mm"], gridsize=42, bins="log",
                   xscale="log", yscale="log", cmap=V.CMAP_RAIN, mincnt=1,
                   linewidths=0.2, edgecolors=V.SURFACE)
    xs = np.array([5, 1200])
    for rate in (1, 10, 50):
        ax.plot(xs, rate * xs / 60, color=V.INK_2, lw=0.9, ls=(0, (5, 2)), zorder=4)
        # anchor each label where its line is still inside the visible box
        x_lab = min(900.0, 120.0 * 60.0 / rate)
        ax.text(x_lab, rate * x_lab / 60, f"{rate} mm/h", fontsize=7.4,
                color=V.INK_2, va="center", ha="center", zorder=5, rotation=0,
                bbox=dict(boxstyle="round,pad=0.22", fc=V.SURFACE, ec="none", alpha=0.9))
    ax.set_xlim(4, 1300)
    ax.set_ylim(0.15, 200)
    ax.set_xlabel("Event duration (minutes)")
    ax.set_ylabel("Event depth (mm)")
    ax.set_title("a   Event duration vs depth")
    V.despine(ax)
    ax.grid(which="both", visible=False)
    cb = fig.colorbar(hb, ax=ax, pad=0.014, fraction=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8)
    cb.set_label("events", fontsize=8.2)

    ax = fig.add_subplot(gs[0, 1])
    for name in V.SEASON_ORDER:
        s = eh[eh["season"] == name].sort_values("start_hour")
        share = 100 * s["n_events"] / s["n_events"].sum()
        ax.plot(s["start_hour"], share, color=V.SEASON_COLOR[name], zorder=3,
                label=name.split(" (")[0] + name[name.find(" ("):])
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 4), [f"{h:02d}" for h in range(0, 24, 4)])
    ax.set_xlabel("Hour the event begins (SGT)")
    ax.set_ylabel("% of the season's events")
    ax.set_title("b   When events start")
    ax.legend(loc="upper left", fontsize=7.8, ncol=1, labelcolor="linecolor")
    V.despine(ax)
    ax.grid(axis="x", visible=False)

    ax = fig.add_subplot(gs[0, 2])
    ax.plot(ea["year"], ea["events_per_station"], color=V.BLUE, marker="o",
            markersize=5, markerfacecolor=V.SURFACE, markeredgewidth=1.8, zorder=3)
    for _, r in ea.iterrows():
        if r["year"] in (2019, 2022):
            high = r["year"] == 2022
            ax.annotate(f"{r['events_per_station']:.0f}",
                        (r["year"], r["events_per_station"]),
                        textcoords="offset points",
                        xytext=(0, 12 if high else -20), ha="center",
                        va="bottom" if high else "top",
                        fontsize=8.4, color=V.INK, fontweight="semibold")
    ax.set_ylabel("Rainfall events per gauge per year")
    ax.set_xticks(ea["year"])
    ax.set_title("c   Event frequency drives the dry years")
    V.despine(ax)
    ax.grid(axis="x", visible=False)
    pad_ylim(ax)

    med_d = ea["median_duration_min"].median()
    V.layout(fig, header=2.25, footer=1.00, left=1.0)
    V.titleblock(fig,
                 "The typical rainfall event is short, light, and begins when the season says it should",
                 f"Splitting each gauge's record into events separated by ≥30 minutes of dry weather gives a median duration of "
                 f"{med_d:.0f} minutes and a median\ndepth near 1 mm, with a long tail of hour-scale storms along the 10–50 mm/h lines in panel a. "
                 "Panel b recovers the same two regimes as\nFigure 5 from a completely different construction, and panel c shows the 2019 drought as a "
                 "deficit of events, not of event size.")
    V.sourcenote(fig, SRC + "\nAn event is a maximal run of wet 5-minute slots at one gauge with no dry gap exceeding 30 minutes. Panel a plots a 12% random sample for legibility.")
    V.save(fig, FIGURES / "fig09_events.png")


# =========================================================================
# F10 - extremes and the Task-2 target
# =========================================================================
def fig10_extremes() -> None:
    x = T("t14_extremes_annual").sort_values("year")
    b = T("t15_base_rates_30min")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 7.0),
                             gridspec_kw=dict(wspace=0.3))

    ax = axes[0]
    w = 0.38
    idx = np.arange(len(x))
    ax.bar(idx - w / 2, x["max_30min_mm"], width=w, color=V.BLUE, zorder=3,
           label="30 minutes")
    ax.bar(idx + w / 2, x["max_60min_mm"], width=w, color=V.BLUE_RAMP[3], zorder=3,
           label="60 minutes")
    for i, r in enumerate(x.itertuples()):
        if r.year == 2022:
            ax.text(i + w / 2, r.max_60min_mm + 3, f"{r.max_60min_mm:.0f}",
                    ha="center", fontsize=8.4, color=V.INK, fontweight="semibold")
    ax.set_xticks(idx, x["year"])
    ax.set_ylabel("Greatest depth at any gauge (mm)")
    ax.set_title("a   Annual maximum short-duration depth", pad=32)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, fontsize=8.4)
    V.despine(ax)
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    # each series is labelled at its own year and on its own side, so no label
    # ever lands on a neighbouring line
    label_at = {"rate_acc30_ge5": (5, -12, "top"),
                "rate_acc30_ge10": (5, 11, "bottom"),
                "rate_acc30_ge20": (4, 11, "bottom")}
    for col, lab, colr in [("rate_acc30_ge5", "$\\geq$5 mm", V.BLUE),
                           ("rate_acc30_ge10", "$\\geq$10 mm", V.ORANGE),
                           ("rate_acc30_ge20", "$\\geq$20 mm", V.AQUA)]:
        v = 100 * x[col] / x[col].iloc[0]
        ax.plot(x["year"], v, color=colr, marker="o", markersize=4.5,
                markerfacecolor=V.SURFACE, markeredgewidth=1.6, zorder=3)
        # label at 2021, where the three series are furthest apart, rather than
        # at the right edge where they converge
        idx, dy, va = label_at[col]
        ax.annotate(lab, (x["year"].iloc[idx], v.iloc[idx]), xytext=(0, dy),
                    textcoords="offset points", fontsize=8.4, color=colr,
                    fontweight="semibold", va=va, ha="center")
    ax.axhline(100, color=V.MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
    ax.set_xticks(x["year"])
    ax.set_xlim(2016.7, 2024.4)
    ax.set_ylabel("Frequency, indexed to 2017 = 100")
    ax.set_title("b   How often 30-min totals cross a threshold")
    V.despine(ax)
    ax.grid(axis="x", visible=False)

    ax = axes[2]
    for name in V.SEASON_ORDER:
        s = b[b["season"] == name].sort_values("hour")
        ax.plot(s["hour"], 100 * s["p_gt_5"], color=V.SEASON_COLOR[name], zorder=3,
                label=name)
    ax.set_xlim(0, 23)
    ax.set_xticks(range(0, 24, 4), [f"{h:02d}" for h in range(0, 24, 4)])
    ax.set_xlabel("Hour of day (SGT)")
    ax.set_ylabel("P(30-min total $\\geq$ 5 mm)  (%)")
    ax.set_title("c   The Task-2 target, as a base rate")
    ax.set_ylim(0, 4.3)
    ax.legend(loc="upper left", fontsize=7.6, labelcolor="linecolor")
    V.despine(ax)
    ax.grid(axis="x", visible=False)
    ax.annotate(f"overall base rate {100 * b['p_gt_5'].mean():.2f}%\n(a severely imbalanced\nclassification problem)",
                xy=(0.98, 0.99), xycoords="axes fraction", ha="right", va="top",
                fontsize=8.2, color=V.INK_2)

    V.layout(fig, header=2.15, footer=1.00, left=1.1)
    V.titleblock(fig,
                 "Heavy half-hours became about 20% more frequent after 2020, and they keep the same daily rhythm",
                 "Panel b indexes the frequency of 5, 10 and 20 mm half-hours to 2017. All three rise together into the wet 2021–2022 period and "
                 "stay\nelevated, so the shift is in how often heavy rain occurs rather than in how heavy the heaviest event gets. The annual "
                 "maxima in panel a\nshow no trend. Panel c is the quantity Task 2 must estimate, and it makes the class imbalance and the "
                 "hour × season structure explicit.")
    V.sourcenote(fig, SRC + "\n30- and 60-minute accumulations are trailing rolling sums computed per gauge over the 5-minute series.")
    V.save(fig, FIGURES / "fig10_extremes.png")


if __name__ == "__main__":
    FIGURES.mkdir(parents=True, exist_ok=True)
    combined = FIGURES / "task1_all_figures.pdf"
    V.open_pdf(combined,
               title="DSA5208 Project 2, Task 1: rainfall patterns across Singapore 2017-2024",
               subject="Figures 1-10, generated from outputs/tables by src/03_task1_figures.py")
    for fn in [fig01_coverage, fig02_annual, fig03_seasonal, fig04_anomaly,
               fig05_diurnal, fig06_intensity, fig07_spatial, fig08_coherence,
               fig09_events, fig10_extremes]:
        fn()
    V.close_pdf()
    print(f"\nPNGs written to {FIGURES}")
    print(f"All 10 figures collected into {combined.name}")
