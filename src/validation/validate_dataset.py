
from pathlib import Path

import pandas as pd
import pandera.pandas as pa

from src.config import load_config
from src.validation.data_schema import schema


# ============================================================
# 1. LOAD DATA
# ============================================================

config = load_config()

DATA_PATH = (
    Path(__file__).resolve().parents[2]
    / config["data"]["synthetic_path"]
    / "land_acquisition_synthetic.csv"
)

df = pd.read_csv(DATA_PATH)

# Parse date columns before schema validation.
date_columns = ["monitoring_date", "planned_start_date"]

for column in date_columns:
    df[column] = pd.to_datetime(df[column], errors="coerce")


# ============================================================
# 2. PANDERA SCHEMA VALIDATION
# ============================================================

print("\n========== PANDERA VALIDATION ==========")

try:
    schema.validate(df, lazy=True)
    print("PASS: All schema checks passed.")

except pa.errors.SchemaErrors as error:
    print("FAIL: Schema validation found errors.")
    print(error.failure_cases.to_string(index=False))
    raise SystemExit("Fix schema errors before continuing.")


# ============================================================
# 3. DOMAIN-SPECIFIC VALIDATION
# ============================================================

print("\n========== DOMAIN VALIDATION ==========")

domain_errors = []


def check(condition, message):
    """Record a failed domain-specific check."""
    if not condition:
        domain_errors.append(message)


# ------------------------------------------------------------
# Basic dataset structure
# ------------------------------------------------------------

check(not df.empty, "Dataset must not be empty.")

check(
    df["project_id"].nunique() == config["data_generation"]["num_projects"],
    "Unexpected number of unique projects.",
)

expected_snapshots = config["data_generation"]["snapshots_per_project"]

snapshot_counts = df.groupby("project_id").size()

check(
    (snapshot_counts == expected_snapshots).all(),
    f"Every project must have exactly {expected_snapshots} snapshots.",
)

check(
    not df.duplicated(["project_id", "monitoring_date"]).any(),
    "Duplicate monitoring dates found within a project.",
)


# ------------------------------------------------------------
# Project-level attributes must remain stable
# ------------------------------------------------------------

stable_columns = [
    "project_type",
    "state",
    "district",
    "land_required_acres",
    "affected_families",
    "compensation_amount",
    "rehabilitation_required",
    "planned_duration_days",
    "planned_start_date",
    "delayed",
]

for column in stable_columns:
    unique_counts = df.groupby("project_id")[column].nunique(
        dropna=False
    )

    check(
        (unique_counts <= 1).all(),
        f"{column} must remain constant within each project.",
    )


# ------------------------------------------------------------
# Temporal consistency
# ------------------------------------------------------------

check(
    (df["monitoring_date"] >= df["planned_start_date"]).all(),
    "Monitoring date cannot precede the planned start date.",
)

ordered = df.sort_values(
    ["project_id", "monitoring_date"]
).copy()

check(
    (
        ordered.groupby("project_id")["elapsed_days"]
        .diff()
        .dropna()
        >= 0
    ).all(),
    "Elapsed days must not decrease across snapshots.",
)

check(
    (
        ordered.groupby("project_id")["monitoring_date"]
        .diff()
        .dropna()
        .dt.days
        > 0
    ).all(),
    "Monitoring dates must increase across snapshots.",
)

check(
    (
        ordered["elapsed_days"]
        == (
            ordered["monitoring_date"]
            - ordered["planned_start_date"]
        ).dt.days
    ).all(),
    "Elapsed days must match the dates in the dataset.",
)


# ------------------------------------------------------------
# Cross-column consistency
# ------------------------------------------------------------

check(
    (
        df["approvals_completed"] <= df["approvals_required"]
    ).all(),
    "Completed approvals cannot exceed required approvals.",
)

check(
    (
        df["delay_days_so_far"]
        == (
            df["elapsed_days"] - df["planned_duration_days"]
        ).clip(lower=0)
    ).all(),
    "Current delay days do not match elapsed and planned duration.",
)

check(
    (
        df.loc[
            df["rehabilitation_required"] == 0,
            "rehabilitation_progress_pct",
        ] == 0
    ).all(),
    "Rehabilitation progress must be zero when rehabilitation is not required.",
)


# ------------------------------------------------------------
# Progress trend checks
# ------------------------------------------------------------

progress_columns = [
    "land_acquired_pct",
    "compensation_paid_pct",
    "documentation_completion_pct",
    "rehabilitation_progress_pct",
]

for column in progress_columns:
    changes = ordered.groupby("project_id")[column].diff().dropna()

    check(
        (changes >= 0).all(),
        f"{column} must not decrease across snapshots.",
    )


# ------------------------------------------------------------
# Report results
# ------------------------------------------------------------

print(f"Rows checked: {len(df)}")
print(f"Projects checked: {df['project_id'].nunique()}")

if domain_errors:
    print("\nFAIL: Domain validation found problems:")

    for number, message in enumerate(domain_errors, start=1):
        print(f"{number}. {message}")

    raise SystemExit("Domain validation failed.")

print("PASS: All domain-specific checks passed.")
print("\nDataset validation completed successfully.")
