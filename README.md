# DSA5208 Project 2: Historical Rainfall across Singapore (2017–2024)

Apache Spark analysis of the data.gov.sg five-minute rainfall record:

* **Task 1** explores how rainfall patterns vary within a year and across 2017–2024.
* **Task 2** predicts the probability that a station records **more than *h* mm of rain in the next 30 minutes**,
  given the station, the prediction time and the threshold (Spark MLlib).

Everything (ingestion, cleaning, pre-processing, the Task 1 analysis, feature engineering, baseline and candidate
models, cross-validation, evaluation and the prediction function) is in **one notebook, `DSA5208_Project2.ipynb`**.
Once executed it keeps the output of every cell, so the whole process can be read top to bottom.

---

## 1. Repository layout

```
├── DSA5208_Project2.ipynb    the complete pipeline (sections 0-20, see §3)
├── environment.yml           conda environment (Python 3.11, PySpark 3.5, Java 17)
├── DSA5208_Project2_Report.pdf   the project report (submitted); DSA5208_Project2_Report.docx is the editable source
├── reference/                Singapore coastline for the Figure 7 map + SOURCES.md (ODbL)
├── cloud/                    Dataproc scalability benchmark: dataproc_benchmark.py + run_dataproc.sh (gcloud commands)
├── models/task2_final_rf/    FINAL Task 2 model: PipelineModel + calibrator + feature_spec.json (submitted)
├── models/task2_baseline_lr/ Stage 1 logistic model (interpretable baseline) + metadata.json
├── outputs/
│   ├── figures/              fig01-fig10 (Task 1), fig11-fig15 (Task 2 stage 1), fig16-fig22 (final model), fig23-fig24 (scalability)
│   │                         + task1_all_figures.pdf, task2_all_figures.pdf, task2_final_model_figures.pdf
│   └── tables/               every result table behind the figures and the text (CSV: t00-t44, t2_*); cloud/ = Dataproc timings
├── data/                     the eight raw CSVs (not in git, 7.5 GB; see §2)
└── data_parquet/             Spark data layers written by the notebook (not in git):
    ├── bronze/               typed copy of the raw CSVs, partitioned by year
    ├── silver/               cleaned readings + station table
    └── gold/                 Task 2 feature tables, training / evaluation sets and 2024 predictions
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
   A full run from scratch takes about 1 h 30 min on the machine in §4: Sections 0–11 about 10 minutes, the
   year-blocked cross-validation in Section 14 about 35–45 minutes with `CV_PARALLELISM=4`, the rest about 30 minutes.
   Keep the machine awake and plugged in for the whole run (on macOS: `caffeinate -is jupyter nbconvert ...`); a
   laptop that sleeps mid-run breaks the Spark session. The run writes `data_parquet/`, `outputs/` and `models/`.
   Optional environment variables:

   | Variable | Default | Meaning |
   |---|---|---|
   | `SPARK_LOCAL_CORES` | all cores | `local[N]` master |
   | `SPARK_DRIVER_MEMORY` | `12g` | driver heap (the submitted run used `16g`; 12 g occasionally hits Spark's execution-memory limit in Section 9 with 14 cores) |
   | `CV_FRACTION` | `0.2` | share of the Stage 2 training sample used for cross-validation (the final refit uses all of it) |
   | `CV_PARALLELISM` | `2` | models fitted in parallel by `CrossValidator` |
   | `RERUN_CV` | `1` | `0` reuses `outputs/tables/t2_cv_results.csv` from a previous run instead of repeating the cross-validation |

   Each finished cross-validation grid is also saved to `outputs/tables/_cv_partial/`, so a run that is interrupted
   during Section 14 resumes with the unfinished grids only; the folder is removed when the cross-validation completes.

**Using the trained model without retraining.** Run Section 0, then every cell whose first line is `# definitions`,
then the `predict()` cell of Section 18. It loads `models/task2_final_rf/` and needs `data_parquet/silver/`
(Sections 1 and 3) to rebuild the features for the requested time:

```python
predict("S77", "2024-10-01 15:00:00", 5)["probability"]          # P(> 5 mm in the next 30 min)
predict("S77", "2024-10-01 15:00:00", 5, curve=True)["curve"]    # P(> h) for h on a grid from 0.2 to 60 mm
```

**Running the cloud benchmark (optional).** `cloud/run_dataproc.sh` lists the `gcloud` commands, block by block:
create a bucket in `asia-southeast1`, upload the eight CSVs and `cloud/dataproc_benchmark.py`, create a Dataproc 2.2
cluster (master and 2 workers `n2-standard-2`, auto-deleted after 30 idle minutes), submit the job, scale to 4 workers,
submit again with `--skip-ingest`, and delete the cluster. The whole experiment costs about US$2. The timings used in
the report are in `outputs/tables/cloud/t43_cloud_scaling.csv` with their Dataproc job IDs; Section 20 of the notebook
reads that file to draw the comparison.

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
| | 10. Stage 1 models | 4 logistic regressions, 2 gradient-boosted tree classifiers; selection on 2023 |
| | 11. Stage 1 evaluation | Stage 1 model vs baselines on 2024; saved as the interpretable baseline; `fig11`–`fig15` |
| | 12. Stage 2 features | richer features → **gold**; independent leakage check |
| | 13. Feature selection | correlation, MLlib `ANOVATest`, selector size tuned in cross-validation (`fig16`) |
| | 14. Cross-validation | year-blocked 4-fold CV + tuning of LR, random forest, GBT; lowest CV log loss = final model (`fig17`) |
| | 15. Ablation, calibration | feature-group ablation (`fig18`); calibrator kept only if it helps on held-out 2023 data |
| | 16. Final evaluation | final model vs baselines on every 2024 row (`fig19`–`fig21`) |
| | 17. Head-to-head | final model vs Stage 1 model on identical 2024 rows (`t41`, `fig22`) |
| | 18. Using the model | save the final model; `predict(station, time, threshold)` |
| | 19. Summary | results and stage timings (`t40`) |
| | 20. Scalability | the cleaning job in fresh Spark sessions with 2, 4, 8 and 16 cores (`t42`, `fig23`); laptop vs Dataproc cluster (`t44`, `fig24`) |

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
| Spark | 3.5.5, `local[14]`, driver heap 16 g, 96 shuffle partitions (64 for model training) |
| Software | Python 3.11.17, OpenJDK 17.0.18, pandas 3.0.6, NumPy 1.26.4, Matplotlib 3.11.2 |
| Time zone | `Asia/Singapore` for every timestamp |
| Storage | Snappy-compressed Parquet, partitioned by year |

The pipeline was developed on a second Apple Silicon Mac (24 GB RAM) with a 12 g driver heap, so it does not
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

The threshold is an input of a single classifier, so one model answers for any station, time and threshold. Task 2
has two stages:

* **Stage 1 (Sections 7–11): an interpretable baseline.** A Spark MLlib logistic regression on the station's own
  recent rain and its four nearest gauges, with threshold × rain interactions, selected on 2023. On 2024 it reaches
  macro PR-AUC 0.446 and ROC-AUC 0.919 (climatology 0.034 and 0.690), is well calibrated from 1 mm upwards and is
  monotone in the threshold by construction. Per threshold (cut-offs fixed on 2023):

  | *h* (mm) | event rate | PR-AUC | ROC-AUC | Brier | log loss | precision | recall | F1 |
  |---|---|---|---|---|---|---|---|---|
  | 0.2 | 4.89 % | 0.638 | 0.910 | 0.0276 | 0.1124 | 0.672 | 0.562 | 0.612 |
  | 1 | 2.59 % | 0.585 | 0.918 | 0.0152 | 0.0661 | 0.648 | 0.539 | 0.588 |
  | 2 | 1.70 % | 0.524 | 0.915 | 0.0109 | 0.0489 | 0.577 | 0.511 | 0.542 |
  | 5 | 0.83 % | 0.418 | 0.912 | 0.0061 | 0.0287 | 0.497 | 0.441 | 0.467 |
  | 10 | 0.38 % | 0.327 | 0.917 | 0.0030 | 0.0150 | 0.363 | 0.435 | 0.395 |
  | 20 | 0.11 % | 0.186 | 0.941 | 0.0009 | 0.0050 | 0.243 | 0.320 | 0.276 |

* **Stage 2 (Sections 12–17): the final model.** Richer features (minutes since the last rain, storm trend, three
  neighbour rings including gauges 5–20 km to the west, island-wide wet share, rain so far today), feature selection
  inside the pipeline, and **4-fold year-blocked cross-validation** with hyper-parameter grids over logistic
  regression, random forest and gradient-boosted trees (25 settings). The model with the lowest cross-validated log
  loss is refitted on 2017–2023, calibrated if a held-out check on 2023 says it helps, and tested once on every 2024
  row. Section 17 scores it and the Stage 1 model on **identical** 2024 rows (`t41_task2_head_to_head.csv`).

  **Result.** The **random forest** (100 trees, depth 12, all 33 candidate features) has the lowest cross-validated
  log loss (0.0320 ± 0.0018, against 0.0392 for logistic regression and 0.0404 for GBT; `t2_cv_results.csv`), and a
  Platt calibrator was kept after the held-out 2023 check. On every 2024 row (`t2_test_metrics.csv`):

  | *h* (mm) | event rate | PR-AUC | ROC-AUC | Brier | log loss | Brier skill | precision | recall | F1 |
  |---|---|---|---|---|---|---|---|---|---|
  | 0.5 | 3.63 % | 0.721 | 0.964 | 0.0174 | 0.0645 | 0.492 | 0.712 | 0.592 | 0.647 |
  | 1 | 2.57 % | 0.667 | 0.965 | 0.0137 | 0.0516 | 0.448 | 0.647 | 0.583 | 0.613 |
  | 2 | 1.69 % | 0.585 | 0.965 | 0.0102 | 0.0397 | 0.383 | 0.589 | 0.521 | 0.553 |
  | 5 | 0.84 % | 0.466 | 0.969 | 0.0058 | 0.0239 | 0.295 | 0.486 | 0.460 | 0.473 |
  | 10 | 0.38 % | 0.349 | 0.973 | 0.0030 | 0.0129 | 0.212 | 0.409 | 0.388 | 0.398 |
  | 20 | 0.10 % | 0.186 | 0.981 | 0.0009 | 0.0046 | 0.098 | 0.277 | 0.280 | 0.278 |

  On identical Stage 1 rows (1,255,003 per threshold, labels matched exactly) the random forest beats the Stage 1
  logistic model on PR-AUC, ROC-AUC and log loss at all five common thresholds (at 5 mm: PR-AUC 0.461 vs 0.418,
  ROC-AUC 0.970 vs 0.912); the PR-AUC gain shrinks as the threshold rises (0.08 at 1 mm, 0.005 at 20 mm). Spark's
  tree models are not bit-for-bit deterministic across runs, so these numbers move in the third decimal.

* **Scalability (Section 20).** The cleaning job (read 48.5 M bronze rows, snap, de-duplicate, aggregate to
  station-days) was timed on the laptop with 2, 4, 8 and 16 cores and on a Google Cloud Dataproc cluster with 2 and 4
  workers (`n2-standard-2`, data in Cloud Storage); both platforms return the same answer.

  | Platform | Configuration | Cores | Wall time (s) | Speed-up | Efficiency |
  |---|---|---|---|---|---|
  | Laptop | `local[2]` / `[4]` / `[8]` / `[16]` | 2 / 4 / 8 / 16 | 26.3 / 13.4 / 8.0 / 5.9 | 1.00 / 1.96 / 3.29 / 4.46× | 100 / 98 / 82 / 56 % |
  | Dataproc | 2 / 4 workers × `n2-standard-2` | 4 / 8 | 74.8 / 37.3 | 1.00 / 2.01× | 100 / 100 % |

  The laptop is faster in absolute terms (fast cores, 128 GB, local SSD), but adding cores to one machine shares one
  memory bus and one disk, so its efficiency falls; adding workers adds CPU, memory and disk together, so doubling the
  cluster halved the time. The cluster was limited to 4 workers by the education project's 12-vCPU quota.

**Leakage checks.** Both stages are verified by an independent pandas recomputation: targets and features of random
rows are recomputed from data truncated at the prediction time and must match the Spark values exactly
(Sections 7.1 and 12.2). Neighbour denominators in Stage 2 count only gauges that have reported by the prediction
time.

**Limitations.** One test year; thin data above 20 mm; no radar or satellite inputs; local-mode Spark only; the GBT
in Stage 1 is uncalibrated (Stage 2 calibrates its candidates).

## 6. Provenance and AI usage

The team built two pipelines: a script pipeline (this repository) and a notebook pipeline (branch
`vanshika/task1-task2`). Both are merged here: the script pipeline provides the data engineering, the Task 1 analysis
and the Stage 1 model; the notebook pipeline provides the gauge-fault QC rules, the trend tests and the Stage 2 final
model (richer features, year-blocked cross-validation, calibration). Changes made while merging Stage 2: it reads
this notebook's silver layer; neighbour denominators no longer count gauges that report only after the prediction
time; the ANOVA ranking uses Spark MLlib's `ANOVATest`; an independent leakage check and an in-notebook head-to-head
comparison were added. The earlier `src/*.py` scripts and `run_task*.sh` wrappers are in the git history.

Claude (Anthropic) was used as a coding assistant for the Spark pipeline, figure code and this README; for Task 2 for
code, debugging, modelling design (target construction, leakage checks, threshold-training strategy, evaluation) and
documentation; and to review and compare the two pipelines, port the QC rules, trend tests and the Stage 2 model,
and consolidate the code into the notebook. All analytical decisions were reviewed against the data, and every number reported here is
produced by the notebook.
