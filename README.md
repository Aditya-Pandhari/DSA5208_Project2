# DSA5208 Project 2: Historical Rainfall across Singapore (2017–2024)

This project uses Apache Spark to analyse Singapore's five-minute rainfall record from data.gov.sg.

* Task 1 examines rainfall patterns within each year and across 2017–2024.
* Task 2 estimates the probability that a station records more than *h* mm of rain in the next 30 minutes,
  given the station, the prediction time and the threshold (Spark MLlib).

The full pipeline is in `DSA5208_Project2.ipynb`: ingestion, cleaning, Task 1 analysis, feature engineering,
model fitting, cross-validation, evaluation and prediction. The executed notebook retains every cell's output,
so readers can follow the analysis from the raw files to the final model.

## 1. Repository layout

```
├── DSA5208_Project2.ipynb    the complete pipeline (sections 0-20, see §3)
├── environment.yml           conda environment (Python 3.11, PySpark 3.5, Java 17)
├── DSA5208_Project2_Report.pdf   PDF report
├── DSA5208_Project2_Report.docx  editable report source
├── reference/                Singapore coastline for the Figure 7 map + SOURCES.md (ODbL)
├── cloud/                    Google Cloud Dataproc: scaling benchmark (dataproc_benchmark.py, run_dataproc.sh) and the
│                             complete notebook run on a cluster (make_dataproc_notebook.py, master_run.sh,
│                             run_full_pipeline_dataproc.sh; executed notebook and outputs in cloud/full_run/)
├── models/task2_final_rf/    final Task 2 model: PipelineModel + calibrator + feature_spec.json
├── models/task2_baseline_lr/ Stage 1 logistic model (interpretable baseline) + metadata.json
├── outputs/
│   ├── figures/              fig01-fig10 (Task 1), fig11-fig15 (Task 2 stage 1), fig16-fig22 (final model), fig23-fig24 (scalability)
│   │                         + task1_all_figures.pdf, task2_all_figures.pdf, task2_final_model_figures.pdf
│   └── tables/               every result table behind the figures and the text (CSV: t00-t47, t2_*; t46 = Spark physical plans); cloud/ = Dataproc timings
├── data/                     the eight raw CSVs (not in git, 7.5 GB; see §2)
└── data_parquet/             Spark data layers written by the notebook (not in git):
    ├── bronze/               typed copy of the raw CSVs, partitioned by year
    ├── silver/               cleaned readings + station table
    └── gold/                 Task 2 feature tables, training / evaluation sets and 2024 predictions
```

The notebook does not import project modules. Besides the rainfall data, it reads
`reference/singapore_boundary.geojson` to draw the coastline on the rainfall map.

## 2. How to run

1. Create the conda environment, which includes Java 17:
   ```bash
   conda env create -f environment.yml
   conda activate dsa5208
   ```
2. Download the annual files for 2017–2024 from
   <https://data.gov.sg/collections/2279/view> into `data/`. The notebook accepts both naming conventions:
   `HistoricalRainfallacrossSingapore2017.csv` or `Historical Rainfall across Singapore (2017).csv`.
3. Open `DSA5208_Project2.ipynb` in Jupyter or VS Code, select the `dsa5208` kernel and run all cells.
   To run it without a notebook interface:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 DSA5208_Project2.ipynb
   ```
   The submitted run took about 1 h 20 min on the machine in §4: 80 minutes wall-clock, including
   73 minutes of timed stages. Sections 0–11 took about 10 minutes, Section 14's year-blocked cross-validation
   took about 37 minutes with `CV_PARALLELISM=4`, and the remaining timed stages took about 25 minutes.
   Keep the machine awake and plugged in throughout (on macOS: `caffeinate -is jupyter nbconvert ...`).
   Sleep can interrupt the Spark session. The notebook writes `data_parquet/`, `outputs/` and `models/`.
   Optional environment variables:

   | Variable | Default | Meaning |
   |---|---|---|
   | `SPARK_LOCAL_CORES` | all cores | `local[N]` master |
   | `SPARK_DRIVER_MEMORY` | `12g` | driver heap (the submitted run used `16g`; 12 g occasionally hits Spark's execution-memory limit in Section 9 with 14 cores) |
   | `CV_FRACTION` | `0.2` | share of the Stage 2 training sample used for cross-validation (the final refit uses all of it) |
   | `CV_PARALLELISM` | `2` | models fitted in parallel by `CrossValidator` |
   | `RERUN_CV` | `1` | `0` reuses `outputs/tables/t2_cv_results.csv` from a previous run instead of repeating the cross-validation |

   Section 14 saves each completed cross-validation grid in `outputs/tables/_cv_partial/`. After an interruption,
   it resumes the unfinished grids and removes the folder when cross-validation completes.

### Use the trained model without retraining

Run Section 0, then every cell whose first line is `# definitions`, followed by the `predict()` cell in Section 18.
The function loads `models/task2_final_rf/` and rebuilds features for the requested time from
`data_parquet/silver/`. If the silver layer is missing, create it with Sections 1 and 3 first.

```python
predict("S77", "2024-10-01 15:00:00", 5)["probability"]          # P(> 5 mm in the next 30 min)
predict("S77", "2024-10-01 15:00:00", 5, curve=True)["curve"]    # P(> h) for h on a grid from 0.2 to 60 mm
```

`predict()` scores the six evaluated thresholds (0.5, 1, 2, 5, 10, 20 mm) and applies the same running minimum
as the evaluation. It returns those probabilities at the evaluated thresholds and interpolates in log *h* between
them. Extensions to 0.2, 30 and 60 mm use model values clipped to prevent increases. Separate calls therefore
cannot return a higher probability for a higher threshold.

Features use readings stamped at or before `t − 5 min`. This sets the measurement-time boundary, but does not
guarantee that those readings were available at clock time `t`: data.gov.sg publishes them 2–10 minutes after
their stamp. Section 4.3 of the notebook explains this distinction.

### Run the complete notebook on Dataproc

`python cloud/make_dataproc_notebook.py` writes the cluster copy to `cloud/DSA5208_Project2_dataproc.ipynb`.
Upload it, `cloud/master_run.sh` and the coastline file to `gs://<bucket>/code/full_run/`, then follow
`cloud/run_full_pipeline_dataproc.sh` in Cloud Shell. The nodes have no public IP, so the cluster needs Cloud NAT
to install Python packages. Allow about 5 h; the original budget estimate was about US$5.

### Run the cloud benchmark

`cloud/run_dataproc.sh` lists the `gcloud` commands in order. Create a bucket in `asia-southeast1` and upload
the eight CSVs and `cloud/dataproc_benchmark.py`. Create a Dataproc 2.2 cluster with a master and 2
`n2-standard-2` workers, configured to delete itself after 30 idle minutes. Submit the job, scale to 4 workers,
submit again with `--skip-ingest`, then delete the cluster. The original budget estimate was about US$2.
The report uses `outputs/tables/cloud/t43_cloud_scaling.csv`, which includes the Dataproc job IDs;
Section 20 reads this file to plot the comparison.

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
| | 8. Training / evaluation sets | threshold as an input, label-independent hash sampling (Stage 1) |
| | 9. Baselines | climatology and base rate |
| | 10. Stage 1 models | 4 logistic regressions, 2 gradient-boosted tree classifiers; selection on 2023 |
| | 11. Stage 1 evaluation | Stage 1 model vs baselines on 2024; saved as the interpretable baseline; `fig11`–`fig15` |
| | 12. Stage 2 features | richer features → **gold**; independent leakage check; 12.4: the physical plans Spark chose (broadcast hash join vs sort-merge join) and adaptive query execution (`t46`) |
| | 13. Feature selection | correlation, MLlib `ANOVATest`, selector size tuned in cross-validation (`fig16`) |
| | 14. Cross-validation | year-blocked 4-fold CV + tuning of LR, random forest, GBT; lowest CV log loss = final model (`fig17`) |
| | 15. Ablation, calibration | feature-group ablation (`fig18`); calibrator kept only if it helps on held-out 2023 data |
| | 16. Final evaluation | final model vs baselines on every 2024 row (`fig19`–`fig21`) |
| | 17. Head-to-head | final model vs Stage 1 model on identical 2024 rows (`t41`, `fig22`) |
| | 18. Using the model | save the final model; `predict(station, time, threshold)` |
| | 19. Summary | results and stage timings (`t40`) |
| | 20. Scalability | strong scaling (8 years on 2–16 cores) with an Amdahl's-law fit, weak scaling (years grow with the cores) (`t42`, `t45`, `t47`, `fig23`); laptop vs Dataproc cluster (`t44`, `fig24`) |

### Data cleaning (Section 3)

The raw data contain no null or negative readings, but timestamp errors and suspected gauge faults still need
correction. The cleaning rules retain 48,338,938 of 48,538,713 records (99.6 %).

1. Keep the single reading type `TB1 Rainfall 5 Minute Total F` (mm).
2. Snap timestamps to the nearest 5-minute grid point (the 2017 export stamps `hh:m4:59`).
3. De-duplicate station-slots, keeping the latest update (86 duplicates, all 2017).
4. Keep readings within 0–30 mm per 5 minutes (removes one suspected 37.2 mm fault).
5. Build the station table from each station's modal coordinate (91 stations).
6. Drop a station-month below 10 % of the network median in a month whose median is ≥ 50 mm
   (67 station-months, 198,544 readings).
7. Drop a station-day of ≥ 100 mm that is more than 4× the wettest gauge within 5 km (4 days,
   e.g. S107 with 749 mm on 2022-08-21 while its neighbours recorded ≤ 32 mm).

Missing readings remain unknown. A usable station-day needs ≥ 80 % of its 288 slots; a usable island-day needs
≥ 20 such stations. The notebook also reports gap-filled totals using the other years' (month, day) climatology.

## 4. Computing environment

The submitted run used Spark in local mode on one machine. `cloud/make_dataproc_notebook.py` contains
the adaptations for cluster execution, including the Spark master, storage and laptop benchmark changes.

| | |
|---|---|
| Machine | Apple MacBook Pro, Apple M4 Max, macOS 26.6.2 |
| CPU / RAM | 16 cores (12 performance + 4 efficiency), 128 GB |
| Spark | 3.5.5, `local[14]`, driver heap 16 g, 96 shuffle partitions (64 for model training) |
| Software | Python 3.11.17, OpenJDK 17.0.18, pandas 3.0.6, NumPy 1.26.4, Matplotlib 3.11.2 |
| Time zone | `Asia/Singapore` for every timestamp |
| Storage | Snappy-compressed Parquet, partitioned by year |

The pipeline was also developed on an Apple Silicon Mac with 24 GB RAM and a 12 g driver heap.
`t00_environment.csv` records the submitted run's environment, and `t40_stage_timings.csv` records its stage timings.

## 5. Results

### Task 1: weather patterns

* Annual island rainfall ranges from 1,787 mm (2019) to 2,900 mm (2021): the wettest year had 62 % more rain
  than the driest. The mean is 2,613 mm (2022: 2,893 mm). The dry year of 2019 had 172 wet days against a
  233-day average; less frequent rain accounts for most of the difference.
* Rainfall has two wet seasons (November–January, April–May) and two daily regimes: a 10:00 peak in the south-west
  monsoon (June–September), a 15:00–16:00 peak in every other season.
* The wettest 10 % of wet 5-minute slots deliver 49 % of the rain.
* Annual rainfall ranges from 3,548 mm/yr at the wettest gauge to 1,700 mm/yr at the driest (2021–24), decreasing by
  about 266 mm per 10 km travelled east. The correlation of 30-minute totals falls to 1/e at about 8.5 km.
* Heavy half-hours were about 23 % more frequent in 2020–24 than in 2017–19 (20 % on a fixed 21-gauge
  network), but no annual series has a statistically significant trend over the eight years (exact Mann–Kendall;
  rain per wet day comes closest, p = 0.06).
* Repeating the annual and monthly analyses on a fixed network of the 21 gauges complete in all eight years
  gives similar results. The fixed-network heavy-rain comparison is included above.

### Task 2: P(next-30-minute rain > *h*)

The threshold is an input to one classifier. The saved predictor accepts an available historical station-time
and a threshold within 0.2–60 mm. Task 2 has two stages:

* **Stage 1 (Sections 7–11): an interpretable baseline.** We selected a Spark MLlib logistic regression on 2023,
  using recent rain at the station and its four nearest gauges, with threshold × rain interactions. On 2024 it reaches
  macro PR-AUC 0.446 and ROC-AUC 0.919 (climatology 0.034 and 0.690), with close average calibration from 1 mm
  upwards. The fitted model is monotone in the threshold for physically valid inputs. Per threshold
  (cut-offs fixed on 2023):

  | *h* (mm) | event rate | PR-AUC | ROC-AUC | Brier | log loss | precision | recall | F1 |
  |---|---|---|---|---|---|---|---|---|
  | 0.2 | 4.89 % | 0.638 | 0.910 | 0.0276 | 0.1124 | 0.672 | 0.562 | 0.612 |
  | 1 | 2.59 % | 0.585 | 0.918 | 0.0152 | 0.0661 | 0.648 | 0.539 | 0.588 |
  | 2 | 1.70 % | 0.524 | 0.915 | 0.0109 | 0.0489 | 0.577 | 0.511 | 0.542 |
  | 5 | 0.83 % | 0.418 | 0.912 | 0.0061 | 0.0287 | 0.497 | 0.441 | 0.467 |
  | 10 | 0.38 % | 0.327 | 0.917 | 0.0030 | 0.0150 | 0.363 | 0.435 | 0.395 |
  | 20 | 0.11 % | 0.186 | 0.941 | 0.0009 | 0.0050 | 0.243 | 0.320 | 0.276 |

* **Stage 2 (Sections 12–17): the final model.** We add minutes since the last rain, storm trend, three
  neighbour rings including gauges 5–20 km to the west, island-wide wet share and rain so far today.
  The pipeline includes feature selection and uses 4-fold year-blocked cross-validation. The grids cover logistic
  regression, random forest and gradient-boosted trees: 22 model settings plus 3 selector sizes, giving
  100 cross-validation fold fits. We select the model with
  the lowest cross-validation log loss, keep calibration only if it helps on held-out 2023 data, refit on
  2017–2023 and test once on every 2024 row. Section 17 compares it with Stage 1 on identical rows
  (`t41_task2_head_to_head.csv`).

  The random forest (100 trees, depth 12, all 33 candidate features) has the lowest cross-validation
  log loss (0.0319 ± 0.0018, against 0.0391 for logistic regression and 0.0403 for GBT; `t2_cv_results.csv`), and a
  Platt calibrator passed the held-out 2023 check. Results on every 2024 row
  (6,371,188 prediction times; `t2_test_metrics.csv`):

  | *h* (mm) | event rate | PR-AUC | ROC-AUC | Brier | log loss | Brier skill | precision | recall | F1 |
  |---|---|---|---|---|---|---|---|---|---|
  | 0.5 | 3.63 % | 0.721 | 0.965 | 0.0174 | 0.0645 | 0.492 | 0.715 | 0.589 | 0.646 |
  | 1 | 2.57 % | 0.667 | 0.966 | 0.0137 | 0.0516 | 0.448 | 0.645 | 0.582 | 0.612 |
  | 2 | 1.69 % | 0.587 | 0.966 | 0.0101 | 0.0395 | 0.386 | 0.567 | 0.543 | 0.555 |
  | 5 | 0.84 % | 0.465 | 0.969 | 0.0059 | 0.0240 | 0.293 | 0.484 | 0.459 | 0.471 |
  | 10 | 0.38 % | 0.351 | 0.973 | 0.0030 | 0.0130 | 0.211 | 0.420 | 0.380 | 0.399 |
  | 20 | 0.10 % | 0.185 | 0.982 | 0.0009 | 0.0045 | 0.101 | 0.263 | 0.284 | 0.273 |

  On the same Stage 1 rows (1,255,003 per threshold, with labels matching exactly), the random forest improves
  PR-AUC, ROC-AUC and log loss at all five common thresholds. At 5 mm, PR-AUC is 0.459 vs 0.418 and ROC-AUC
  is 0.970 vs 0.912. Spark's tree models vary across runs; the observed differences are in the third decimal.

* **Scalability (Section 20).** The cleaning job (read 48.5 M bronze rows, snap, de-duplicate, aggregate to
  station-days) was timed on the laptop with 2, 4, 8 and 16 cores and on a Google Cloud Dataproc cluster with 2 and 4
  workers (`n2-standard-2`, data in Cloud Storage). Both platforms return the same answer.

  | Platform | Configuration | Cores | Wall time (s) | Speed-up | Efficiency |
  |---|---|---|---|---|---|
  | Laptop | `local[2]` / `[4]` / `[8]` / `[16]` | 2 / 4 / 8 / 16 | 27.8 / 13.6 / 8.0 / 5.7 | 1.00 / 2.04 / 3.48 / 4.88× | 100 / 102 / 87 / 61 % |
  | Dataproc | 2 / 4 workers × `n2-standard-2` | 4 / 8 | 74.8 / 37.3 | 1.00 / 2.01× | 100 / 100 % |

  For strong scaling, fitting Amdahl's law, T(n) = a + b/n, to the laptop's median times gives
  T(n) = 1.79 + 51.1/n s, with an estimated serial share of 3.4 %. The implied ceiling of 30× over one core
  extrapolates a two-parameter fit to four points; the experiment does not measure that ceiling.
  For weak scaling (1 year on 2 cores up to 8 years on 16, `t45_weak_scaling.csv`), the data grows 9.2× while
  time rises from 3.3 s to 5.8 s. Scaled speed-up is 5.3× on 16 cores (ideal 8×, efficiency 66 %; Gustafson's law).

  The laptop runs the job faster in absolute terms. It has fast cores, 128 GB RAM and a local SSD, although its
  cores share one memory bus and disk. Adding cluster workers adds CPU, memory and disk together, consistent
  with the observed halving of runtime when workers doubled. The education project's 12-vCPU quota limited
  the cluster to 4 workers.

* **The complete notebook on a Dataproc cluster (`cloud/full_run/`).** An earlier version of the whole notebook ran
  on Google Cloud Dataproc 2.2 (master + 2 workers `n2-highmem-4`, Spark 3.5.3 on YARN, data layers on HDFS).
  `cloud/make_dataproc_notebook.py` generated that copy from the notebook as it stood on 14 October 2026.
  It predates the Spark-plan cell 12.4, the strong/weak-scaling rewrite of Section 20, the Stage 2 train/test boundary
  purge and the fixed-grid `predict()` (see `cloud/full_run/README.md`). It ran without errors in 290 minutes
  of timed work (4.0× the laptop's 73), with the following comparison:

  | Result | Laptop | Cluster |
  |---|---|---|
  | Cleaning, Task 1 tables and Stage 1 data sets (9 tables) | identical | identical (largest difference 1e-12) |
  | Stage 1 logistic model: 2024 macro PR-AUC | 0.4462 | 0.4462 |
  | Cross-validation: best setting of all four grids | — | same settings |
  | Cross-validation: random-forest log loss | 0.0319 | 0.0319 |
  | Final random forest on 2024: PR-AUC at 1 / 5 / 20 mm | 0.667 / 0.465 / 0.185 | 0.666 / 0.465 / 0.189 |
  | Head-to-head at 5 mm: final vs Stage 1 PR-AUC | 0.459 vs 0.418 | 0.460 vs 0.418 |

  The only Task 1 value that differs is the 99th-percentile event depth (0.26 mm), which uses Spark's approximate
  percentile. A quoting bug prevented collection of the cluster-trained Spark model folders before cluster deletion.
  `cloud/master_run.sh` now fixes that collection step; the submitted model is the laptop-trained one.

### Leakage checks

Independent pandas calculations check random rows in both stages (Sections 7.1 and 12.2).
The feature check truncates input history at the anchor; a separate calculation checks future rainfall targets.
The checked rows match Spark exactly. Stage 2 neighbour denominators count only gauges with readings by the
prediction time.

Both stages purge the train/test boundary. Stage 2 drops end-of-2023 rows whose 30-minute targets reach into
2024. Its three internal cross-validation boundaries remain unpurged: about 370 of 13.7 M rows have targets
extending up to 20 minutes into the next fold. This slightly weakens the isolation between folds.

The checks test window computations on retrospectively cleaned data. They do not establish that every input
was published by the prediction time or that the month/day-level gauge-fault rules could run live.

### Sampling

Stage 1 uses a hash of station, time and threshold, independently of the label. Within each threshold, this
preserves prevalence in expectation and needs no inclusion weights. Stage 2 keeps every row with rain in the
last 3 hours or the next 30 minutes, plus 3 % of the remaining quiet rows. Its selection therefore depends on
the outcome. Inverse inclusion weights (1 for always-kept rows, 1 / 0.03 for sampled quiet rows) correct weighted
totals in expectation. The notebook reports weighted and unweighted positive rates, and checks that the
weighted row count, adjusted for threshold expansion, is close to the population row count.

### Limitations

The evaluation has one test year and few examples above 20 mm. Gauge observations cannot show storms before
they reach a gauge; the models have no radar or satellite inputs. The submitted model was trained locally,
and the cloud experiments used only 2–4 workers. Stage 1's GBT is uncalibrated; Stage 2 checks calibration
on held-out data. Publication delays, retrospective cleaning and the internal fold overlaps also limit the
claims above.

## 6. Provenance and AI usage

The team combined a script pipeline from this repository with a notebook pipeline from branch
`vanshika/task1-task2`. The script pipeline contributed data engineering, Task 1 and the Stage 1 model.
The notebook pipeline contributed the gauge-fault QC rules, trend tests and Stage 2, including richer features,
year-blocked cross-validation and calibration.

During the merge, Stage 2 was changed to read this notebook's silver layer. We corrected neighbour denominators
that counted gauges reporting after the prediction time, switched the ANOVA ranking to MLlib's `ANOVATest`,
and added an independent leakage check and a comparison on identical rows. The earlier `src/*.py` scripts and
`run_task*.sh` wrappers remain in the git history.

The team used Claude (Anthropic) to help write and debug the Spark pipeline, figures and documentation.
For Task 2, this included modelling design: target construction, leakage checks, threshold-training strategy
and evaluation. Claude also helped compare the pipelines, port the QC rules, trend tests and Stage 2 model,
and consolidate the code into the notebook. OpenAI Codex, using the humanize skill, edited the notebook's
Markdown cells, report prose and this README. The team reviewed analytical decisions against the data;
the notebook produces the reported results.
