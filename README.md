# DSA5208 Project 2: Historical Rainfall across Singapore (2017–2024)

Apache Spark analysis of the data.gov.sg five-minute rainfall record:

* **Task 1** explores how rainfall patterns vary within a year and across 2017–2024.
* **Task 2** predicts the probability that a station records **more than *h* mm of rain in the next 30 minutes**,
  given the station, the prediction time and the threshold (Spark MLlib).

Everything (ingestion, cleaning, pre-processing, the Task 1 analysis, feature engineering, baseline and candidate
models, evaluation and the prediction function) is in **one notebook, `DSA5208_Project2.ipynb`**, saved with the
output of every cell, so the whole process can be read top to bottom without running it.

---

## 1. Repository layout

```
├── DSA5208_Project2.ipynb    the complete pipeline, with outputs (sections 0-13, see §3)
├── environment.yml           conda environment (Python 3.11, PySpark 3.5, Java 17)
├── DSA5208_Project2_Report.docx  the project report (Word; export to PDF for submission)
├── reference/                Singapore coastline for the Figure 7 map + SOURCES.md (ODbL)
├── models/task2_exceedance/  trained Spark PipelineModel + metadata.json (the submitted model)
├── outputs/
│   ├── figures/              fig01-fig10 (Task 1), fig11-fig15 (Task 2), 300 dpi PNG
│   │                         + task1_all_figures.pdf, task2_all_figures.pdf
│   └── tables/               every result table behind the figures and the text (CSV, t00-t40)
├── data/                     the eight raw CSVs (not in git, 7.5 GB; see §2)
└── data_parquet/             Spark data layers written by the notebook (not in git):
    ├── bronze/               typed copy of the raw CSVs, partitioned by year
    ├── silver/               cleaned readings + station table
    └── gold/                 Task 2 feature table and training / evaluation sets
```

No other code is needed: the notebook does not import any project modules. Its only input besides the data is
`reference/singapore_boundary.geojson`, used to draw the coastline on the rainfall map.

## 2. How to run

1. **Environment** (Java 17 is required by PySpark 3.5 and is installed by conda):
   ```bash
   conda env create -f environment.yml
   conda activate dsa5208
   ```
2. **Data**: download the annual files 2017–2024 from
   <https://data.gov.sg/collections/2279/view> into `data/`. Either name data.gov.sg has used is accepted:
   `HistoricalRainfallacrossSingapore2017.csv` or `Historical Rainfall across Singapore (2017).csv`.
3. **Run**: open `DSA5208_Project2.ipynb` in Jupyter or VS Code with the `dsa5208` environment's kernel and run
   all cells, or headless:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 DSA5208_Project2.ipynb
   ```
   A full run takes about 15–20 minutes on the machine in §4 and writes `data_parquet/`, `outputs/` and `models/`.
   Optional settings: `SPARK_LOCAL_CORES` (default: all cores) and `SPARK_DRIVER_MEMORY` (default `12g`).

**Using the trained model without retraining.** Run Section 0, then every cell whose first line is `# definitions`
(Sections 4, 4.3, 6, 7 and 10), then Section 12. It needs `data_parquet/silver/` (Sections 1 and 3) to rebuild the
features for the requested time:

```python
predict("S77", "2024-10-01 15:00:00", 5)      # -> {"probability": 0.004, ...}
predict("S77", "2024-10-01 15:00:00", 5, curve=True)["curve"]   # P(> h) for every h on the grid
```

## 3. What the notebook does

| Part | Section | Content |
|---|---|---|
| A. Data engineering | 0. Setup | paths, Spark session, computing environment, plotting style |
| | 1. Ingestion | CSV → **bronze** Parquet with an explicit schema |
| | 2. Data-quality audit | nulls, ranges, duplicates, timestamp grid, publication delay (`t00_raw_profile`) |
| | 3. Cleaning | bronze → **silver** (rules below; `t01`, `t01a`, `t01b`) |
| | 4. Pre-processing | station-days, completeness rules, island-wide daily series, gap filling, fixed gauge network, time-window functions |
| B. Task 1 | 5. Weather patterns | annual totals, seasonal cycle, robustness to the changing network, diurnal cycle, intensity, spatial pattern, spatial coherence, rain events, extremes, trend tests (`fig01`–`fig10`) |
| C. Task 2 | 6. Problem definition | target, inputs, timestamp convention, chronological split, metrics |
| | 7. Feature engineering | trailing rain, neighbouring gauges, calendar → **gold** feature table; independent leakage check |
| | 8. Training / evaluation sets | threshold as an input, label-independent sampling |
| | 9. Baselines | climatology and base rate |
| | 10. Candidate models | 4 logistic regressions, 2 gradient-boosted tree classifiers; selection on 2023 |
| | 11. Final evaluation | selected model vs baselines on the untouched 2024 test year; save model; `fig11`–`fig15` |
| | 12. Using the model | reload the saved model; `predict(station, time, threshold)` |
| | 13. Summary | results and stage timings (`t40`) |

### Data cleaning (Section 3)

The raw values are clean (no nulls or negative readings); the problems are the timestamp grid, completeness and a
few faulty gauges. 48,338,938 of 48,538,713 records (99.6 %) are kept.

1. Keep the single reading type `TB1 Rainfall 5 Minute Total F` (mm).
2. **Snap timestamps** to the nearest 5-minute grid point (the 2017 export stamps `hh:m4:59`).
3. **De-duplicate** station-slots, keeping the latest update (86 duplicates, all 2017).
4. **Range check**: 0–30 mm per 5 minutes (removes one 37.2 mm fault).
5. **Station table**: modal coordinate per station (91 stations).
6. **Dead gauges**: drop a station-month below 10 % of the network median in a month whose median is ≥ 50 mm
   (67 station-months, 198,544 readings).
7. **Spike days**: drop a station-day of ≥ 100 mm that is more than 4× the wettest gauge within 5 km (4 days,
   e.g. S107 with 749 mm on 2022-08-21 while its neighbours recorded ≤ 32 mm).

Missing readings are never filled with zeros. Island-wide quantities use usable station-days (≥ 80 % of the day's
288 slots) and usable island-days (≥ 20 such stations); totals are also reported gap-filled from the (month, day)
climatology of the other years.

## 4. Computing environment

Single-node Spark in local mode (the same code runs on a cluster by changing the master):

| | |
|---|---|
| Machine | Apple MacBook Pro, Apple M4 Max, macOS 26.6.2 |
| CPU / RAM | 16 cores (12 performance + 4 efficiency), 128 GB |
| Spark | 3.5.5, `local[14]`, driver heap 12 g, 96 shuffle partitions (64 for model training) |
| Software | Python 3.11.17, OpenJDK 17.0.18, pandas 3.0.6, NumPy 1.26.4, Matplotlib 3.11.2 |
| Time zone | `Asia/Singapore` for every timestamp |
| Storage | Snappy-compressed Parquet, partitioned by year |

The pipeline was developed on a second Apple Silicon Mac (24 GB RAM) with the same 12 g driver heap, so it does not
need the larger machine. The notebook records the environment in `t00_environment.csv` and the time of every stage in
`t40_stage_timings.csv`.

## 5. Results

### Task 1: weather patterns

* **Large year-to-year variability.** Annual island rainfall ranges from 1,787 mm (2019) to 2,900 mm (2021), a 62 %
  spread around a 2,613 mm mean (2022: 2,893 mm). 2019 was dry because it rained less often (172 wet days against a
  233-day average), not because the rain was lighter.
* **Two wet seasons** (November–January, April–May) and **two diurnal regimes**: a 10:00 peak in the south-west
  monsoon (June–September), a 15:00–16:00 peak in every other season.
* **Concentrated, heavy-tailed rain**: the wettest 10 % of wet 5-minute slots deliver 49 % of the rain.
* **Strong spatial structure**: 3,548 mm/yr at the wettest gauge against 1,700 mm/yr at the driest (2021–24), about
  266 mm less per 10 km travelled east; 30-minute rainfall decorrelates within about 8.5 km.
* **Trends**: heavy half-hours were about 23 % more frequent in 2020–24 than in 2017–19 (20 % on a fixed 21-gauge
  network), but no annual series has a statistically significant trend over the eight years (exact Mann–Kendall;
  rain per wet day comes closest, p = 0.06).
* **Robustness**: every headline result holds on a fixed network of the 21 gauges complete in all eight years.

### Task 2: P(next-30-minute rain > *h*)

One Spark MLlib logistic-regression pipeline with the threshold as an input (selected on 2023 by log loss from four
logistic regressions and two gradient-boosted tree classifiers). Untouched 2024 test year, cut-offs fixed on 2023:

| *h* (mm) | event rate | PR-AUC | ROC-AUC | Brier | log loss | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| 0.2 | 4.89 % | 0.638 | 0.910 | 0.0276 | 0.1124 | 0.672 | 0.562 | 0.612 |
| 1 | 2.59 % | 0.585 | 0.918 | 0.0152 | 0.0661 | 0.648 | 0.539 | 0.588 |
| 2 | 1.70 % | 0.524 | 0.915 | 0.0109 | 0.0489 | 0.577 | 0.511 | 0.542 |
| 5 | 0.83 % | 0.418 | 0.912 | 0.0061 | 0.0287 | 0.497 | 0.441 | 0.467 |
| 10 | 0.38 % | 0.327 | 0.917 | 0.0030 | 0.0150 | 0.363 | 0.435 | 0.395 |
| 20 | 0.11 % | 0.186 | 0.941 | 0.0009 | 0.0050 | 0.243 | 0.320 | 0.276 |

| Model (2024, macro average over thresholds) | PR-AUC | ROC-AUC | log-loss skill vs climatology |
|---|---|---|---|
| Base rate per threshold | 0.017 | 0.500 | −0.04 |
| Climatology (season × hour × threshold) | 0.034 | 0.690 | 0 |
| **Logistic regression + neighbours + interactions (selected)** | **0.446** | **0.919** | **0.392** |
| Gradient-boosted trees, depth 7 | 0.446 | 0.923 | 0.019 |

The selected model is well calibrated from 1 mm upwards and its probability always falls as the threshold rises (0
violations in 25.8 M threshold pairs). The GBT ranks about as well (equal PR-AUC, higher ROC-AUC) but its probabilities are badly calibrated
(Spark's GBT starts every prediction at 0.5, so 80 trees cannot reach 0.1 % probabilities). Limitations: one test
year, thin data above 20 mm, no radar or satellite inputs, and the GBT compared without recalibration.

## 6. Provenance and AI usage

The team built two pipelines: a script pipeline (this repository) and a notebook pipeline (branch
`vanshika/task1-task2`). The script pipeline's Task 2 model was the only one trained on the full data, so it was
selected; its gauge-fault QC rules and trend tests were taken from the notebook pipeline. The final code was then
consolidated into the single notebook above, replacing the earlier `src/*.py` scripts and `run_task*.sh` wrappers
(still available in the git history).

Claude (Anthropic) was used as a coding assistant for the Spark pipeline, figure code and this README; for Task 2 for
code, debugging, modelling design (target construction, leakage checks, threshold-training strategy, evaluation) and
documentation; and to review and compare the two pipelines, port the QC rules and trend tests, and consolidate the
code into the notebook. All analytical decisions were reviewed against the data, and every number reported here is
produced by the notebook.
