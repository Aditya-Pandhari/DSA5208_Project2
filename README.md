# DSA5208 Project 2: Historical Rainfall across Singapore (2017–2024)

Spark pipeline for **Task 1: exploring weather patterns** in the data.gov.sg
five-minute rainfall record. Task 2 (the 30-minute exceedance model) is built on
the same cleaned Parquet store produced by stage 1.

---

## 1. What the pipeline does

| Stage | Script | Input | Output | Runtime |
|---|---|---|---|---|
| 1. Ingest & clean | `src/01_ingest.py` | 8 annual CSVs (7.5 GB) | `data_parquet/` (338 MB) + `t01_cleaning_audit.csv` | ~4 min |
| 2. Analysis | `src/02_task1_analysis.py` | `data_parquet/` | 20 result tables (`t02`–`t21`) in `outputs/tables/` | ~1.5 min |
| 3. Figures | `src/03_task1_figures.py` | `outputs/tables/` | 10 PNGs (300 dpi) + one 10-page PDF | ~40 s |

Stage 3 uses no Spark, so figures can be re-rendered in seconds while iterating
on presentation.

## 2. Requirements

Java 17 is required, because PySpark 3.5 is not reliable on the Java 22 that ships by
default on recent macOS installs, so the JDK is pinned inside the conda
environment.

```bash
conda create -y -n dsa5208 python=3.11 openjdk=17 pyspark=3.5 pandas numpy matplotlib pyarrow
```

Verified versions: Python 3.11.16, PySpark 3.5.5, OpenJDK 17.0.18, pandas 3.0.6,
NumPy 1.26.4, Matplotlib 3.11.2.

## 3. Running it

Place the eight raw files in `data/`, named exactly as downloaded:

```
data/Historical Rainfall across Singapore (2017).csv
...
data/Historical Rainfall across Singapore (2024).csv
```

Then:

```bash
./run_task1.sh                 # full pipeline
./run_task1.sh --skip-ingest   # re-run analysis + figures only
```

Or step by step:

```bash
export JAVA_HOME=/opt/homebrew/Caskroom/miniforge/base/envs/dsa5208/lib/jvm
export SPARK_LOCAL_CORES=12
$CONDA_PREFIX/bin/python src/01_ingest.py
$CONDA_PREFIX/bin/python src/02_task1_analysis.py
$CONDA_PREFIX/bin/python src/03_task1_figures.py
```

`SPARK_LOCAL_CORES` sets the `local[N]` master; the driver heap is configured in
`src/common.py` (`driver_mem`, default 12 g).

## 4. Computing environment

Single-node local Spark, no cluster:

| | |
|---|---|
| Machine | Apple Silicon Mac, macOS 15 (Darwin 25.6.0) |
| CPU / RAM | 15 logical cores, 24 GB |
| Spark | 3.5.5, `local[12]`, driver heap 12 g, 96 shuffle partitions |
| Session timezone | `Asia/Singapore` (all timestamps are SGT) |
| Storage | Snappy-compressed Parquet, partitioned by `year` |

The 7.5 GB CSV input compresses to 338 MB of Parquet, a 22× reduction, which is
what makes stages 2 and 3 fast enough to iterate on.

## 5. Data cleaning

Record accounting per year is written to `outputs/tables/t01_cleaning_audit.csv`.
The source turned out to be unusually clean at the *value* level: no nulls, no
negative readings, and exactly one implausible value (>30 mm in five minutes)
across 48.5 M records. The real data-quality problem is **completeness**.

1. **Explicit schema**: no inference pass over 7.5 GB.
2. **Single reading type**: only `TB1 Rainfall 5 Minute Total F` in mm.
3. **Timestamp snapping**: observation times are parsed as Asia/Singapore and
   rounded to the nearest 5-minute grid point. The 2017 export is offset by a
   few seconds and drifts between polls, so without this the years do not align.
4. **De-duplication**: a station/5-minute slot re-published under a later
   `update_timestamp` is collapsed to its most recent reading, via a `max()`
   over a struct (one shuffle, rather than a sorted window over 48.5 M rows).
   86 duplicates were found, all in 2017.
5. **Range filter**: readings outside `[0, 30] mm` per 5 minutes are dropped.
6. **Station reconciliation**: a station can appear with slightly different
   coordinates across years, so the modal coordinate becomes the canonical one.
   91 distinct stations resolve from the eight files.

### Completeness rules used downstream

Because the network grows from ~50 gauges (2017–19) to ~76 (2020–22) and falls
back to ~62 (2024), a raw average over rows would let the changing network
masquerade as a change in climate. Instead:

* a **station-day** is usable with ≥ 230 of 288 five-minute slots (80 %);
* an **island-day** is usable with ≥ 20 usable station-days;
* **island rainfall** is the unweighted mean of usable station-day totals;
* annual totals are reported both as observed sums and as gap-filled estimates
  (unusable days take the day-of-year climatology of the remaining years).

63 calendar days have no data at all, 28 of them in **August 2018**, which is
~90 % absent from the source export. 2018 annual totals should not be quoted
without this caveat.

## 6. Outputs

**Figures** (`outputs/figures/`): one PNG per figure at 300 dpi, plus all ten
collected as vector pages in `task1_all_figures.pdf`.

| Figure | Finding |
|---|---|
| `fig01_data_foundation` | Network coverage audit and the cost of the QC rule |
| `fig02_annual_totals` | Annual rainfall split into frequency × intensity |
| `fig03_seasonal_cycle` | Month × year matrix; the double wet season |
| `fig04_monthly_anomaly` | Wet/dry spells persist for seasons, not months |
| `fig05_diurnal_cycle` | **Two rainfall clocks, and the monsoon selects which runs** |
| `fig06_intensity` | Heavy-tailed intensity; rainfall concentration in time |
| `fig07_spatial_climatology` | West–east gradient; rainfall doubles across 50 km |
| `fig08_spatial_coherence` | Correlation vs distance; 8.5 km decorrelation length |
| `fig09_events` | Storm events: duration, depth, start hour |
| `fig10_extremes` | Annual maxima, threshold frequency, Task-2 base rates |

**Tables** (`outputs/tables/`, 21 CSVs: `t01` from stage 1, `t02`–`t21` from stage 2).
Every figure is drawn from one of these, so each chart has a machine-readable
table behind it.

## 7. Headline results

* **Interannual range is large.** 1,775 mm (2019) to 2,904 mm (2022), a 63 %
  spread around a 2,610 mm mean. 2019 was a positive-IOD drought year with 12
  consecutive below-average months; 2021–22 were La Niña-wet.
* **A dry year is a year with fewer rain events, not gentler rain.** 2019's
  deficit is a collapse in wet days (172 vs a 225-day norm) and in events per
  gauge (266 vs ~400); its per-wet-day intensity matches 2017's.
* **Singapore runs two diurnal regimes.** June–September (south-west monsoon)
  peaks at **10:00**, the Sumatra squall signature, while every other season
  peaks sharply at **15:00–16:00** from local convection. The peak hour swings
  about six hours across the year.
* **Rainfall is extremely concentrated.** The wettest 10 % of wet 5-minute
  intervals deliver 49 % of the annual total; the wettest 1 % deliver 10 %. The
  curve is near-identical in every year, drought included.
* **A steep west–east gradient.** Over the common 2021–24 window the wettest
  gauge records 3,548 mm/yr and the driest 1,669 mm/yr, or 257 mm lost per 10 km
  travelled east (r = −0.67).
* **Rainfall decorrelates in ~8.5 km**, shorter than the island. Gauges < 2 km
  apart correlate at ρ = 0.84 on 30-minute totals; beyond 20 km, ρ = 0.15. A
  stretched exponential ρ = exp(−(d/9.3)^0.76) fits the correlogram five times
  better than a plain exponential.
* **Heavy half-hours became ~20 % more frequent after 2020** while annual maxima
  show no trend. This is a change in how often, not in how hard.

## 8. Implications for Task 2

Three Task-1 results constrain the model directly:

1. **Predict per station.** The 8.5 km decorrelation length means one island-wide
   forecast cannot be right everywhere at once; neighbouring-gauge features are
   informative only at short range.
2. **Frame it as exceedance, not conditional mean.** The intensity distribution
   is heavy-tailed, so a regression on the mean would be dominated by the 90 % of
   wet intervals that carry half the rain.
3. **Expect severe class imbalance with strong structure.** P(30-min total ≥ 5 mm)
   is 0.86 % overall but varies by a factor of ten with hour × season
   (`t15_base_rates_30min.csv`), so hour, season and recent-rainfall state all
   belong in the feature set.

## 9. AI usage

Claude (Anthropic) was used as a coding assistant for the Spark pipeline,
figure code and this README. All analytical decisions (QC thresholds, the areal
averaging scheme, choice of figures and interpretation of results) were reviewed
against the data; numerical results come from the pipeline in this repository and
are reproducible with `./run_task1.sh`.

## 10. Repository layout

```
├── data/                     raw CSVs (not in version control, 7.5 GB)
├── data_parquet/             cleaned fact table + station dimension
│   ├── readings/             partitioned by year
│   └── stations/
├── src/
│   ├── common.py             Spark session factory, paths, QC constants
│   ├── viz_style.py          figure design system (palette, layout, styling)
│   ├── 01_ingest.py          stage 1
│   ├── 02_task1_analysis.py  stage 2
│   └── 03_task1_figures.py   stage 3
├── outputs/
│   ├── figures/              10 PNGs + task1_all_figures.pdf (10 pages)
│   └── tables/               21 result tables (t01–t21)
├── logs/
├── run_task1.sh
└── README.md
```
