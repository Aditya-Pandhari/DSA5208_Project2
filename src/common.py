"""Shared Spark session factory and project paths (DSA5208 Project 2)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data"
DATA_PARQUET = ROOT / "data_parquet"
OUT = ROOT / "outputs"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"

YEARS = list(range(2017, 2025))

# Physical / QC constants -------------------------------------------------
# NEA TB1 tipping-bucket gauges report a 5-minute total in mm. The Singapore
# record 5-minute depth is ~20 mm; anything beyond PLAUSIBLE_MAX_5MIN is
# treated as an instrument fault rather than rain.
PLAUSIBLE_MAX_5MIN = 30.0
WET_5MIN_MM = 0.2          # a 5-min interval counts as "wet" above this
WET_DAY_MM = 1.0           # WMO-style wet-day threshold (mm/day)
EXPECTED_INTERVALS_PER_DAY = 288


def spark_session(app_name: str, shuffle_partitions: int = 64, driver_mem: str = "12g"):
    """Build a local Spark session tuned for a single multi-core workstation."""
    from pyspark.sql import SparkSession

    # workers must run the same interpreter as the driver (needed whenever Python
    # objects, e.g. createDataFrame from a list, cross into the JVM)
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)

    cores = os.environ.get("SPARK_LOCAL_CORES", "*")
    builder = (
        SparkSession.builder.appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.session.timeZone", "Asia/Singapore")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.parquet.compression.codec", "snappy")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.local.dir", os.environ.get("SPARK_LOCAL_DIR", "/tmp/spark-tmp"))
    )
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark


def write_table(pdf, name: str) -> Path:
    """Persist a small pandas frame as a CSV result table."""
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    pdf.to_csv(path, index=False)
    print(f"  [table] {path.relative_to(ROOT)}  ({len(pdf)} rows)")
    return path
