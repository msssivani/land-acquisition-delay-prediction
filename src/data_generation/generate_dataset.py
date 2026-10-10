from pathlib import Path

import numpy as np
import pandas as pd

from src.config import load_config


# ============================================================
# 1. LOAD CONFIGURATION
# ============================================================

config = load_config()

num_projects = config["data_generation"]["num_projects"]
snapshots_per_project = config["data_generation"]["snapshots_per_project"]
random_seed = config["data_generation"]["random_seed"]

synthetic_path = Path(config["data"]["synthetic_path"])

rng = np.random.default_rng(random_seed)


# ============================================================
# 2. REFERENCE DATA
# ============================================================

project_types = [
    "Road",
    "Railway",
    "Bridge",
    "Power",
    "Water",
    "Urban_Infrastructure",
    "Irrigation",
    "Industrial_Infrastructure",
]

states_and_districts = {
    "Andhra_Pradesh": ["Guntur", "Krishna", "Prakasam", "NTR"],
    "Telangana": ["Hyderabad", "Warangal", "Nalgonda"],
    "Karnataka": ["Bengaluru", "Mysuru", "Belagavi"],
    "Tamil_Nadu": ["Chennai", "Madurai", "Coimbatore"],
    "Maharashtra": ["Pune", "Nashik", "Nagpur"],
    "Gujarat": ["Ahmedabad", "Surat", "Rajkot"],
    "Madhya_Pradesh": ["Bhopal", "Indore", "Jabalpur"],
    "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur"],
    "Uttar_Pradesh": ["Lucknow", "Agra", "Varanasi"],
    "West_Bengal": ["Kolkata", "Howrah", "Siliguri"],
    "Odisha": ["Bhubaneswar", "Cuttack", "Sambalpur"],
    "Kerala": ["Ernakulam", "Kozhikode", "Thiruvananthapuram"],
    "Bihar": ["Patna", "Gaya", "Muzaffarpur"],
    "Jharkhand": ["Ranchi", "Dhanbad", "Jamshedpur"],
    "Chhattisgarh": ["Raipur", "Bilaspur", "Durg"],
    "Punjab": ["Ludhiana", "Amritsar", "Patiala"],
    "Haryana": ["Gurugram", "Hisar", "Karnal"],
    "Assam": ["Guwahati", "Dibrugarh", "Jorhat"],
    "Uttarakhand": ["Dehradun", "Haridwar", "Nainital"],
    "Himachal_Pradesh": ["Shimla", "Kangra", "Mandi"],
}


# Different infrastructure types can have different typical
# project durations. These are only used for synthetic variation,
# NOT to directly determine whether a project is delayed.
typical_duration_days = {
    "Road": (365, 1800),
    "Railway": (900, 3650),
    "Bridge": (365, 1800),
    "Power": (500, 2500),
    "Water": (400, 2200),
    "Urban_Infrastructure": (300, 1500),
    "Irrigation": (600, 3000),
    "Industrial_Infrastructure": (400, 2200),
}


# ============================================================
# 3. GENERATE PROJECTS AND SNAPSHOTS
# ============================================================

projects = []

for project_number in range(num_projects):

    project_id = f"PRJ{project_number + 1:05d}"

    # --------------------------------------------------------
    # Project-level attributes
    # These remain constant across all snapshots.
    # --------------------------------------------------------

    project_type = rng.choice(project_types)

    state = rng.choice(list(states_and_districts.keys()))
    district = rng.choice(states_and_districts[state])

    duration_min, duration_max = typical_duration_days[project_type]

    planned_duration_days = int(
        rng.integers(duration_min, duration_max + 1)
    )

    land_required_acres = float(
        rng.lognormal(mean=np.log(300), sigma=0.9)
    )

    land_required_acres = np.clip(
        land_required_acres,
        5,
        5000
    )

    # Family density is intentionally variable.
    # It is NOT a fixed relationship with land area.
    family_density = rng.uniform(0.03, 0.25)

    # Project type changes the distribution slightly,
    # but does not directly determine risk.
    if project_type in ["Urban_Infrastructure", "Railway"]:
        family_density *= rng.uniform(1.1, 1.5)

    elif project_type in ["Irrigation", "Power"]:
        family_density *= rng.uniform(0.7, 1.1)

    affected_families = max(
        0,
        int(
            rng.normal(
                land_required_acres * family_density,
                50
            )
        )
    )

    affected_families = min(
        affected_families,
        5000
    )

    # --------------------------------------------------------
    # Compensation
    # --------------------------------------------------------

    compensation_per_acre = rng.lognormal(
        mean=np.log(8_000_000),
        sigma=0.8
    )

    compensation_per_family = rng.lognormal(
        mean=np.log(800_000),
        sigma=0.7
    )

    compensation_amount = (
        land_required_acres * compensation_per_acre
        + affected_families * compensation_per_family
    )

    compensation_amount = np.clip(
        compensation_amount,
        100_000,
        5_000_000_000
    )

    # --------------------------------------------------------
    # Administrative / dispute propensities
    #
    # These are hidden variables used only to create realistic
    # relationships in the synthetic data.
    #
    # They will NOT be included in the final dataset.
    # --------------------------------------------------------

    administrative_complexity = rng.beta(2.5, 4.0)

    dispute_propensity = rng.beta(2.0, 5.0)

    documentation_difficulty = rng.beta(2.5, 4.5)

    coordination_difficulty = rng.beta(2.0, 4.0)

    resource_stability = rng.beta(5.0, 2.0)

    # --------------------------------------------------------
    # Rehabilitation requirement
    # --------------------------------------------------------

    rehabilitation_required = int(
        affected_families >= 100
        or rng.random() < 0.20
    )

    # --------------------------------------------------------
    # Hidden project difficulty
    #
    # This is NOT saved.
    # It creates realistic relationships between the variables
    # and the eventual delayed/not-delayed outcome.
    # --------------------------------------------------------

    project_difficulty = (
        0.20 * administrative_complexity
        + 0.20 * dispute_propensity
        + 0.15 * documentation_difficulty
        + 0.15 * coordination_difficulty
        + 0.10 * (1 - resource_stability)
        + 0.10 * min(land_required_acres / 3000, 1)
        + 0.10 * min(affected_families / 3000, 1)
    )

    project_difficulty = np.clip(
        project_difficulty,
        0,
        1
    )

    # --------------------------------------------------------
    # Generate eventual outcome.
    #
    # This represents what ultimately happened to the project.
    # The model will later try to predict this outcome from
    # information available at earlier snapshots.
    # --------------------------------------------------------

    delay_probability = 1 / (
        1 + np.exp(
            -(
                -1.2
                + 4.0 * project_difficulty
                + rng.normal(0, 0.35)
            )
        )
    )

    delayed = int(
        rng.random() < delay_probability
    )

    # --------------------------------------------------------
    # Actual eventual duration
    #
    # Used internally to create realistic snapshots and target.
    # It will NOT be used as a model input.
    # --------------------------------------------------------

    if delayed:
        additional_delay = int(
            rng.gamma(
                shape=2.5,
                scale=90
            )
            + 60 * project_difficulty
        )

        actual_duration_days = (
            planned_duration_days
            + additional_delay
        )

    else:
        early_completion = int(
            rng.uniform(0, 0.10)
            * planned_duration_days
        )

        actual_duration_days = max(
            1,
            planned_duration_days - early_completion
        )

    # --------------------------------------------------------
    # Planned start date
    # --------------------------------------------------------

    planned_start_date = pd.Timestamp("2022-01-01") + pd.Timedelta(
        days=int(rng.integers(0, 365))
    )

    # ========================================================
    # SNAPSHOTS
    # ========================================================

    for snapshot_number in range(snapshots_per_project):

        # ----------------------------------------------------
        # Snapshot timing
        #
        # Most snapshots happen before the planned deadline.
        # This is important because the project is supposed to
        # support EARLY risk prediction.
        # ----------------------------------------------------

        snapshot_position = (
            snapshot_number + 1
        ) / (
            snapshots_per_project + 1
        )

        elapsed_days = int(
            actual_duration_days
            * snapshot_position
        )

        monitoring_date = (
            planned_start_date
            + pd.Timedelta(days=elapsed_days)
        )

        # ----------------------------------------------------
        # Expected progress based on planned schedule
        # ----------------------------------------------------

        planned_progress_pct = (
            elapsed_days
            / planned_duration_days
        ) * 100

        planned_progress_pct = np.clip(
            planned_progress_pct,
            0,
            100
        )

        # ----------------------------------------------------
        # Land acquisition progress
        #
        # It generally increases over time but is affected by
        # project difficulty.
        # ----------------------------------------------------

        expected_land_progress = (
            15
            + snapshot_position * 75
            - project_difficulty * 20
        )

        land_acquired_pct = (
            expected_land_progress
            + rng.normal(0, 5)
        )

        land_acquired_pct = np.clip(
            land_acquired_pct,
            0,
            100
        )

        # ----------------------------------------------------
        # Approval progress
        # ----------------------------------------------------

        approvals_required = int(
            np.clip(
                round(
                    rng.normal(
                        6 + administrative_complexity * 5,
                        2
                    )
                ),
                1,
                15
            )
        )

        approval_rate = (
            snapshot_position
            - administrative_complexity * 0.25
            + rng.normal(0, 0.04)
        )

        approval_rate = np.clip(
            approval_rate,
            0,
            1
        )

        approvals_completed = int(
            np.floor(
                approvals_required
                * approval_rate
            )
        )

        approvals_completed = min(
            approvals_completed,
            approvals_required
        )

        # ----------------------------------------------------
        # Approval delay
        # ----------------------------------------------------

        approval_delay_days = int(
            max(
                0,
                administrative_complexity * 180
                + snapshot_position * administrative_complexity * 100
                + rng.normal(0, 20)
            )
        )

        approval_delay_days = min(
            approval_delay_days,
            365
        )

        # ----------------------------------------------------
        # Legal cases
        # ----------------------------------------------------

        expected_legal_cases = (
            dispute_propensity * 12
            + affected_families / 500
            + land_required_acres / 1000
        )

        legal_cases = int(
            max(
                0,
                rng.poisson(
                    max(expected_legal_cases, 0.1)
                )
            )
        )

        legal_cases = min(
            legal_cases,
            50
        )

        # ----------------------------------------------------
        # Disputed land
        # ----------------------------------------------------

        disputed_land_pct = (
            dispute_propensity * 50
            + rng.normal(0, 5)
        )

        disputed_land_pct = np.clip(
            disputed_land_pct,
            0,
            100
        )

        # ----------------------------------------------------
        # Documentation progress
        # ----------------------------------------------------

        documentation_completion_pct = (
            snapshot_position * 85
            - documentation_difficulty * 25
            + rng.normal(0, 5)
        )

        documentation_completion_pct = np.clip(
            documentation_completion_pct,
            0,
            100
        )

        # ----------------------------------------------------
        # Compensation progress
        # ----------------------------------------------------

        compensation_paid_pct = (
            snapshot_position * 85
            - dispute_propensity * 20
            - documentation_difficulty * 10
            + rng.normal(0, 5)
        )

        compensation_paid_pct = np.clip(
            compensation_paid_pct,
            0,
            100
        )

        # ----------------------------------------------------
        # Rehabilitation progress
        # ----------------------------------------------------

        if rehabilitation_required:

            rehabilitation_progress_pct = (
                snapshot_position * 80
                - coordination_difficulty * 20
                + rng.normal(0, 5)
            )

            rehabilitation_progress_pct = np.clip(
                rehabilitation_progress_pct,
                0,
                100
            )

        else:
            rehabilitation_progress_pct = 0

        # ----------------------------------------------------
        # Resource availability
        # ----------------------------------------------------

        resource_availability_pct = (
            65
            + resource_stability * 30
            - project_difficulty * 15
            + rng.normal(0, 5)
        )

        resource_availability_pct = np.clip(
            resource_availability_pct,
            30,
            100
        )

        # ----------------------------------------------------
        # Cost variance
        # ----------------------------------------------------

        cost_variance_pct = (
            project_difficulty * 15
            + rng.normal(0, 8)
        )

        cost_variance_pct = np.clip(
            cost_variance_pct,
            -20,
            50
        )

        # ----------------------------------------------------
        # Physical progress
        #
        # Physical progress can lag behind planned progress
        # when acquisition/approval issues exist.
        # ----------------------------------------------------

        physical_progress_pct = (
            planned_progress_pct
            - project_difficulty * 25
            + rng.normal(0, 5)
        )

        physical_progress_pct = np.clip(
            physical_progress_pct,
            0,
            100
        )

        # ----------------------------------------------------
        # Delay accumulated so far
        #
        # IMPORTANT:
        # This is descriptive information about the current
        # state and will later be treated carefully for ML.
        # ----------------------------------------------------

        delay_days_so_far = max(
            0,
            elapsed_days - planned_duration_days
        )

        # ----------------------------------------------------
        # Store snapshot
        # ----------------------------------------------------

        row = {
            "project_id": project_id,
            "project_type": project_type,
            "state": state,
            "district": district,

            "land_required_acres": round(
                land_required_acres,
                2
            ),

            "land_acquired_pct": round(
                land_acquired_pct,
                2
            ),

            "affected_families": affected_families,

            "compensation_amount": round(
                compensation_amount,
                2
            ),

            "compensation_paid_pct": round(
                compensation_paid_pct,
                2
            ),

            "approvals_required": approvals_required,

            "approvals_completed": approvals_completed,

            "approval_delay_days": approval_delay_days,

            "legal_cases": legal_cases,

            "disputed_land_pct": round(
                disputed_land_pct,
                2
            ),

            "documentation_completion_pct": round(
                documentation_completion_pct,
                2
            ),

            "rehabilitation_required": rehabilitation_required,

            "rehabilitation_progress_pct": round(
                rehabilitation_progress_pct,
                2
            ),

            "resource_availability_pct": round(
                resource_availability_pct,
                2
            ),

            "cost_variance_pct": round(
                cost_variance_pct,
                2
            ),

            "physical_progress_pct": round(
                physical_progress_pct,
                2
            ),

            "planned_progress_pct": round(
                planned_progress_pct,
                2
            ),

            "planned_duration_days": planned_duration_days,

            "elapsed_days": elapsed_days,

            "monitoring_date": monitoring_date,

            "planned_start_date": planned_start_date,

            "delay_days_so_far": delay_days_so_far,

            "delayed": delayed,
        }

        projects.append(row)


# ============================================================
# 4. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(projects)


# ============================================================
# 5. SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    ["project_id", "monitoring_date"]
).reset_index(drop=True)


# ============================================================
# 6. BASIC VALIDATION
# ============================================================

assert len(df) == num_projects * snapshots_per_project

assert df["project_id"].nunique() == num_projects

assert (
    df.groupby("project_id").size()
    == snapshots_per_project
).all()

assert (
    df["land_acquired_pct"].between(0, 100)
).all()

assert (
    df["compensation_paid_pct"].between(0, 100)
).all()

assert (
    df["documentation_completion_pct"].between(0, 100)
).all()

assert (
    df["rehabilitation_progress_pct"].between(0, 100)
).all()

assert (
    df["approvals_completed"]
    <= df["approvals_required"]
).all()

assert (
    df["delay_days_so_far"] >= 0
).all()

assert set(df["delayed"].unique()).issubset({0, 1})


# ============================================================
# 7. SAVE DATASET
# ============================================================

synthetic_path.mkdir(
    parents=True,
    exist_ok=True
)

output_file = (
    synthetic_path
    / "land_acquisition_synthetic.csv"
)

df.to_csv(
    output_file,
    index=False
)


# ============================================================
# 8. SUMMARY
# ============================================================

print("\nDataset generated successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Projects: {df['project_id'].nunique()}")
print(f"Snapshots per project: {snapshots_per_project}")
print(f"Delayed projects: {df['delayed'].mean():.2%}")
print(f"Output: {output_file}")

print("\nFirst 5 rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)