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

* a **station-day** is usable with ≥ 231 of 288 five-minute slots (≥ 80 %; 231 = ceil(0.80 × 288));
* an **island-day** is usable with ≥ 20 usable station-days;
* **island rainfall** is the unweighted mean of usable station-day totals;
* annual totals are reported both as observed sums and as gap-filled estimates
  (unusable days take the (month, day) climatology of the remaining years, which stays calendar-aligned across leap years).

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

**Tables** (`outputs/tables/`: `t01` from stage 1, `t02`–`t24` Task 1, `t25`–`t38` Task 2).
Every figure is drawn from one of these, so each chart has a machine-readable
table behind it.

## 7. Headline results

* **Interannual range is large.** 1,778 mm (2019, gap-filled) to 2,904 mm (2022), a 63 %
  spread around a 2,610 mm mean. 2019 had 12 consecutive below-average months (Dec 2018–Nov 2019); 2021–22 were the wettest years. Published
  climate-variability explanations (e.g. a positive Indian Ocean Dipole in 2019, La Niña in 2021–22) are
  external context to be cited, not something this dataset demonstrates.
* **A dry year is a year with fewer rain events, not gentler rain.** 2019's
  deficit is a collapse in wet days (172 vs a 225-day norm) and in events per
  gauge (266 vs ~400); its per-wet-day intensity matches 2017's.
* **Singapore runs two diurnal regimes.** June–September (south-west monsoon)
  peaks at **10:00**, a timing consistent with Sumatra-squall activity reported in the literature, while every
  other season peaks sharply at **15:00–16:00**, consistent with afternoon local convection (mechanisms are
  external interpretation, not tested here). The peak hour swings
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

## 9. Task 2: probability of > h mm of rain in the next 30 minutes

**Model input**: station, prediction time, threshold h (mm). **Output**: P(rain in the next 30 min > h).

```bash
./run_task2.sh                  # features -> leakage check -> train/select/test -> figures
./run_task2.sh --skip-features  # re-use the feature table
./run_task2.sh --refit          # ignore cached candidate models in models/_candidates/
python src/predict_task2.py --station S77 --time "2024-10-01 15:00:00" --threshold 5
```

(Use the `dsa5208` environment's Python. Needs `data_parquet/` from `./run_task1.sh`.)
Last full run on the machine above: features about 4.5 min, training/selection/test about 16 min with cached
candidate fits (fits themselves: LR about 1 min each, GBT about 3 min each), figures under 1 min.

**Target.** Row timestamps mark the *start* of a 5-minute interval (the publish stamp trails by ≥ 284 s), so the
prediction time is t = ts + 5 min, the instant the latest slot completed. `future_30min_mm` is the sum of the six
slots stamped ts+5 … ts+30 min, taken with a time-based window and valid only if all six slots exist.
Examples with an incomplete future (or history) are dropped, never labelled dry. Label = `future_30min_mm > h`
(strict). `src/verify_task2_features.py` recomputes labels and features independently in pandas.

**Information boundary.** Features use static coordinates, the clock at t, and rain stamped ≤ ts (own gauge and
up to 4 neighbours within 10 km). Nothing is learned from rainfall outside the training years.

**Features** (`models/task2_exceedance/metadata.json` lists the exact vector): lat, lon; sin/cos of time of day and
of day of year; latest 5-min rain, trailing 15/30/60/180-min totals, 60-min max and wet-slot count, wet flag;
neighbour trailing 30/60-min mean, 30-min max and latest-5-min max, neighbour-availability flag; log1p(threshold);
threshold × recent-rain interaction terms (the `ix_` features).

**Split (temporal, no random split).** Train: t + 30 min ≤ 2023-01-01 (2017–2022). Validation: 2023. Test: 2024,
scored once after selection on 2023.

| split | prediction times | stations | P(>0.2) | P(>1) | P(>5) | P(>10) | P(>20) |
|---|---|---|---|---|---|---|---|
| train | 33,109,182 | 88 | 4.41 % | 2.34 % | 0.82 % | 0.39 % | 0.11 % |
| validation | 6,627,634 | 73 | 5.07 % | 2.63 % | 0.87 % | 0.41 % | 0.11 % |
| test | 6,280,030 | 70 | 4.86 % | 2.57 % | 0.84 % | 0.38 % | 0.10 % |

**Threshold training.** Grid 0.2, 0.5, 1, 2, 3, 5, 7.5, 10, 15, 20, 30 mm. Each training prediction time is
replicated across thresholds by a deterministic hash, sampled so that thresholds up to 7.5 mm contribute about 8,000
positives each and heavier thresholds use a 6 % sample (a threshold-dependent but label-independent sampling rate; `t27_task2_training_design.csv`). Sampling
does not depend on the label, so the per-threshold positive rate is unchanged and no recalibration is needed.
Evaluation uses all common 2023 / 2024 prediction times at 0.2, 1, 2, 5, 10, 20 mm.

**Models.** Climatology and per-threshold base rate (training only); Spark ML `LogisticRegression` pipelines
(local, + neighbours, + interactions, + stronger L2); `GBTClassifier` (depth 5 and 7). Pipelines use
StringIndexer/OneHot/VectorAssembler where applicable. Selection: lowest macro log loss on 2023.

| model (2023) | macro PR-AUC | macro ROC-AUC | macro log-loss skill vs climatology | threshold-violating pairs |
|---|---|---|---|---|
| Climatology | 0.043 | 0.720 | 0 | n/a |
| LR local | 0.403 | 0.899 | 0.325 | 0 % |
| LR + neighbours | 0.445 | 0.929 | 0.373 | 0 % |
| **LR + neighbours + interactions (selected)** | 0.458 | 0.929 | **0.393** | 0 % |
| LR + interactions, stronger L2 | 0.463 | 0.930 | 0.383 | 0 % |
| GBT depth 5 | 0.454 | 0.925 | 0.029 | 0.01 % |
| GBT depth 7 | 0.452 | 0.932 | 0.031 | 0.19 % |

GBT ranks about as well as LR but its probabilities are poorly calibrated (log-loss skill about 0.03) and it
is not monotone in the threshold (up to 0.54 probability violation), so it was not chosen. The selected logistic
model is monotone in h by construction (no post-processing needed).

**Untouched 2024 test (selected model; F1 cutoff fixed on 2023)**

| h (mm) | prevalence | PR-AUC | ROC-AUC | Brier | log loss | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 4.89 % | 0.638 | 0.910 | 0.0276 | 0.1125 | 0.656 | 0.571 | 0.611 |
| 1 | 2.59 % | 0.585 | 0.918 | 0.0152 | 0.0661 | 0.649 | 0.538 | 0.588 |
| 2 | 1.70 % | 0.524 | 0.915 | 0.0109 | 0.0489 | 0.586 | 0.504 | 0.542 |
| 5 | 0.83 % | 0.418 | 0.912 | 0.0061 | 0.0287 | 0.498 | 0.440 | 0.467 |
| 10 | 0.38 % | 0.327 | 0.917 | 0.0030 | 0.0150 | 0.395 | 0.404 | 0.400 |
| 20 | 0.11 % | 0.186 | 0.941 | 0.0009 | 0.0050 | 0.244 | 0.318 | 0.276 |

Macro-averaged over thresholds on 2024: PR-AUC 0.446, ROC-AUC 0.919, log-loss skill 0.392 (climatology: 0.034 and
0.690). Accuracy is deliberately not reported. Full tables: `t28`–`t38` (reliability `t31`, PR curves `t32`,
season / hour / region subgroups `t33`, monotonicity `t35`, feature effects `t37`, threshold sweep `t38`).
Subgroups with fewer than 100 positives are flagged `small_sample`.

**Figures**: `fig11`–`fig15` (+ `task2_all_figures.pdf`): metrics by threshold, calibration, PR curves,
model comparison and logistic effects, and a probability-vs-threshold sweep for four recent-rain states.

**Saved model**: `models/task2_exceedance/pipeline_model` (Spark `PipelineModel`, about 128 KB) with
`metadata.json` (features, thresholds, cutoffs, conventions). `src/predict_task2.py` rebuilds the features from
`data_parquet/` and rejects unknown stations, times outside the history (3 h of history needed) and negative thresholds.

**Limitations.** One chronological split (a single test year); thresholds above 30 mm and unusually heavy
events are poorly constrained; a logistic model with hand-built interactions cannot capture storm dynamics
(no radar or satellite data); thresholds above the 30 mm grid edge are extrapolated (the sweep figure's 30 mm end is already thin); 2023/2024 station sets differ
slightly from training (unseen stations are handled with `handleInvalid="keep"`).

## 10. AI usage

Claude (Anthropic) was used as a coding assistant for the Spark pipeline, figure code and this README, and,
for Task 2, for code, debugging, modelling design (target construction, leakage checks, threshold-training
strategy, evaluation) and documentation. All analytical decisions were reviewed against the data; every
number in this README was generated by the pipeline in this repository (`./run_task1.sh`, `./run_task2.sh`).

## 11. Repository layout

```
├── data/                     raw CSVs (not in version control, 7.5 GB)
├── data_parquet/             cleaned fact table, stations, Task 2 features/sets (not committed)
├── src/
│   ├── common.py, viz_style.py
│   ├── 01_ingest.py, 02_task1_analysis.py, 03_task1_figures.py
│   ├── task2_lib.py, task2_eval.py            shared Task 2 definitions / metrics
│   ├── 04_task2_features.py, verify_task2_features.py
│   ├── 05_task2_train.py, 06_task2_figures.py
│   └── predict_task2.py
├── models/task2_exceedance/  selected PipelineModel + metadata.json
├── outputs/figures/          fig01-fig10 (Task 1), fig11-fig15 (Task 2)
├── outputs/tables/           t01-t38
├── logs/
├── run_task1.sh, run_task2.sh
└── README.md
```
