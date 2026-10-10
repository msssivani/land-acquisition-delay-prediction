
import pandera.pandas as pa


# ============================================================
# CANONICAL DATA SCHEMA
# ============================================================

schema = pa.DataFrameSchema(
    columns={
        # ----------------------------------------------------
        # Project identity and location
        # ----------------------------------------------------
        "project_id": pa.Column(
            str,
            nullable=False,
        ),
        "project_type": pa.Column(
            str,
            checks=pa.Check.isin([
                "Road",
                "Railway",
                "Bridge",
                "Power",
                "Water",
                "Urban_Infrastructure",
                "Irrigation",
                "Industrial_Infrastructure",
            ]),
            nullable=False,
        ),
        "state": pa.Column(str, nullable=False),
        "district": pa.Column(str, nullable=False),

        # ----------------------------------------------------
        # Land and compensation
        # ----------------------------------------------------
        "land_required_acres": pa.Column(
            float,
            checks=pa.Check.in_range(5, 5000),
            nullable=False,
        ),
        "land_acquired_pct": pa.Column(
            float,
            checks=pa.Check.in_range(0, 100),
            nullable=False,
        ),
        "affected_families": pa.Column(
            int,
            checks=pa.Check.in_range(0, 5000),
            nullable=False,
        ),
        "compensation_amount": pa.Column(
            float,
            checks=pa.Check.in_range(100_000, 5_000_000_000),
            nullable=False,
        ),
        "compensation_paid_pct": pa.Column(
            float,
            checks=pa.Check.in_range(0, 100),
            nullable=False,
        ),

        # ----------------------------------------------------
        # Administrative approvals
        # ----------------------------------------------------
        "approvals_required": pa.Column(
            int,
            checks=pa.Check.in_range(1, 15),
            nullable=False,
        ),
        "approvals_completed": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "approval_delay_days": pa.Column(
            int,
            checks=pa.Check.in_range(0, 365),
            nullable=False,
        ),

        # ----------------------------------------------------
        # Legal and documentation status
        # ----------------------------------------------------
        "legal_cases": pa.Column(
            int,
            checks=pa.Check.in_range(0, 50),
            nullable=False,
        ),
        "disputed_land_pct": pa.Column(
            float,
            checks=pa.Check.in_range(0, 100),
            nullable=False,
        ),
        "documentation_completion_pct": pa.Column(
            float,
            checks=pa.Check.in_range(0, 100),
            nullable=False,
        ),

        # ----------------------------------------------------
        # Rehabilitation and resources
        # ----------------------------------------------------
        "rehabilitation_required": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
        ),
        "rehabilitation_progress_pct": pa.Column(
            float,
            checks=pa.Check.in_range(0, 100),
            nullable=False,
        ),
        "resource_availability_pct": pa.Column(
            float,
            checks=pa.Check.in_range(30, 100),
            nullable=False,
        ),

        # ----------------------------------------------------
        # Cost and physical progress
        # ----------------------------------------------------
        "cost_variance_pct": pa.Column(
            float,
            checks=pa.Check.in_range(-20, 50),
            nullable=False,
        ),
        "physical_progress_pct": pa.Column(
            float,
            checks=pa.Check.in_range(0, 100),
            nullable=False,
        ),
        "planned_progress_pct": pa.Column(
            float,
            checks=pa.Check.in_range(0, 100),
            nullable=False,
        ),

        # ----------------------------------------------------
        # Schedule information
        # ----------------------------------------------------
        "planned_duration_days": pa.Column(
            int,
            checks=pa.Check.ge(1),
            nullable=False,
        ),
        "elapsed_days": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False,
        ),
        "monitoring_date": pa.Column(
            "datetime64[ns]",
            nullable=False,
        ),
        "planned_start_date": pa.Column(
            "datetime64[ns]",
            nullable=False,
        ),
        "delay_days_so_far": pa.Column(
            int,
            checks=pa.Check.ge(0),
            nullable=False,
        ),

        # ----------------------------------------------------
        # Prediction target
        # ----------------------------------------------------
        "delayed": pa.Column(
            int,
            checks=pa.Check.isin([0, 1]),
            nullable=False,
        ),
    },

    # Catch missing or unexpected columns.
    strict=True,

    # Convert compatible values to the declared data types.
    coerce=True,

    # Cross-column consistency rule.
    checks=[
        pa.Check(
            lambda df: (
                df["approvals_completed"]
                <= df["approvals_required"]
            ).all(),
            error="Completed approvals cannot exceed required approvals.",
        ),
    ],
)
