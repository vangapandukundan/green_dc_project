"""
config.py — Central Configuration (Phase 2)
=============================================
Expanded to include all features, unsupervised model 
configurations, and detailed categorical mappings.
"""

import os

# ── Root Directory ────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))

# ── Data Paths ────────────────────────────────────────────────
DATA_DIR       = os.path.join(ROOT_DIR, "data")
RAW_DATA_PATH  = os.path.join(DATA_DIR, "green_quantum_data_centers_2.csv")
PROC_DATA_PATH = os.path.join(DATA_DIR, "processed_data.csv")

# ── Model Artifact Paths ──────────────────────────────────────
MODELS_DIR     = os.path.join(ROOT_DIR, "models")
RESULTS_PATH   = os.path.join(MODELS_DIR, "model_results.pkl")
OUTPUTS_DIR    = os.path.join(ROOT_DIR, "outputs")

# ── Feature Definitions (Expanded for Phase 2) ────────────────
FEATURE_COLS = [
    "compute_demand_TFlops",
    "storage_demand_TB",
    "network_demand_Gbps",
    "renewable_share_percent",
    "qso_optimization_score",
    "uncertainty_factor",
    "pqc_enabled",
    "energy_efficiency_index",
    "service_quality_index",
    "secure_operations_score",
    "operational_cost_usd",
    "performance_metric",
    # Encoded categorical features:
    "workload_encoded",
    "energy_source_encoded",
    "security_level_encoded",
    "scenario_encoded",
    "strategy_encoded"
]

TARGET_COL = "energy_consumption_kWh"

# ── Categorical Mappings ──────────────────────────────────────
WORKLOAD_MAP = {
    "Cloud Storage"    : 0,
    "Database Queries" : 1,
    "IoT Processing"   : 2,
    "Web Hosting"      : 3,
}

ENERGY_SOURCE_MAP = {
    "Grid"  : 0,
    "Solar" : 1,
    "Wind"  : 2,
    "Hydro" : 3,
}

SECURITY_LEVEL_MAP = {
    "Low"    : 0,
    "Medium" : 1,
    "High"   : 2,
}

SCENARIO_MAP = {
    "Idle"   : 0,
    "Normal" : 1,
    "Peak"   : 2,
}

STRATEGY_MAP = {
    "Conventional"             : 0,
    "Efficient Focus"          : 1,
    "Carbon-Neutral Objective" : 2,
}

# ── City Temperature Map (For What-If) ───────────────────────
CITY_TEMPS = {
    "Chennai (Hot)"    : 35,
    "Mumbai"           : 32,
    "Delhi"            : 28,
    "Hyderabad"        : 30,
    "Bangalore (Cool)" : 22,
}

# ── Constants ─────────────────────────────────────────────────
CO2_KG_PER_KWH          = 0.82
ZOMBIE_ENERGY_QUANTILE  = 0.75
ZOMBIE_COMPUTE_QUANTILE = 0.25

TEST_SIZE    = 0.2
RANDOM_STATE = 42
CV_FOLDS     = 5

OPTUNA_TRIALS  = 30
OPTUNA_TIMEOUT = 120

# ── Model Configs ─────────────────────────────────────────────
MODEL_DEFAULTS = {
    "RandomForest": {
        "n_estimators" : 200,
        "max_depth"    : 10,
        "random_state" : RANDOM_STATE,
    },
    "XGBoost": {
        "n_estimators"      : 150,
        "max_depth"         : 6,
        "learning_rate"     : 0.1,
        "random_state"      : RANDOM_STATE,
    },
    "LightGBM": {
        "n_estimators"  : 150,
        "max_depth"     : 6,
        "learning_rate" : 0.1,
        "random_state"  : RANDOM_STATE,
        "verbose"       : -1,
    },
    "GradientBoosting": {
        "n_estimators"  : 150,
        "max_depth"     : 5,
        "learning_rate" : 0.1,
        "random_state"  : RANDOM_STATE,
    },
}

# Unsupervised ML configs
CLUSTERING_CLUSTERS = 3
ANOMALY_CONTAMINATION = 0.05  # Assume 5% of our servers exhibit anomalous behavior
