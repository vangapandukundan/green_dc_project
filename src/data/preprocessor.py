"""
src/data/preprocessor.py — Feature Engineering Pipeline
=========================================================
Handles:
  1. Deduplication & null removal
  2. PUE calculation
  3. CO2 emissions derivation
  4. Zombie server flagging
  5. Workload type label encoding
  6. Saving processed CSV

All thresholds and constants come from config.py — never hardcoded here.
"""

import os
import sys
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config           import (PROC_DATA_PATH, CO2_KG_PER_KWH,
                               ZOMBIE_ENERGY_QUANTILE,
                               ZOMBIE_COMPUTE_QUANTILE,
                               WORKLOAD_MAP, ENERGY_SOURCE_MAP,
                               SECURITY_LEVEL_MAP, SCENARIO_MAP,
                               STRATEGY_MAP)
from src.utils.logger import get_logger

log = get_logger(__name__)


# ── Step 1: Clean ─────────────────────────────────────────────
def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates().dropna().reset_index(drop=True)
    removed = before - len(df)
    log.info(f"Cleaned data → removed {removed} rows, {len(df):,} rows remaining")
    return df


# ── Step 2: Engineer Features ─────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # PUE — Power Usage Effectiveness
    max_compute      = df["compute_demand_TFlops"].max()
    efficiency       = df["compute_demand_TFlops"] / max_compute
    df["it_energy_kWh"] = df["energy_consumption_kWh"] * (0.50 + 0.20 * efficiency)
    df["PUE"]           = (df["energy_consumption_kWh"] / df["it_energy_kWh"]).round(4)

    # CO2 Emissions
    df["co2_kg"] = (df["energy_consumption_kWh"] * CO2_KG_PER_KWH).round(4)

    # Zombie Server Flag
    e_thresh         = df["energy_consumption_kWh"].quantile(ZOMBIE_ENERGY_QUANTILE)
    c_thresh         = df["compute_demand_TFlops"].quantile(ZOMBIE_COMPUTE_QUANTILE)
    df["is_zombie"]  = (
        (df["energy_consumption_kWh"] > e_thresh) &
        (df["compute_demand_TFlops"]  < c_thresh)
    ).astype(int)

    # Workload & Categorical Encoding (Use explicit maps for consistency across dashboard)
    # .map() replaces string with integer. .fillna(-1) catches any missing map items just in case
    df["workload_encoded"]       = df["workload_type"].map(WORKLOAD_MAP).fillna(-1).astype(int)
    df["energy_source_encoded"]  = df["energy_source"].map(ENERGY_SOURCE_MAP).fillna(-1).astype(int)
    df["security_level_encoded"] = df["security_level"].map(SECURITY_LEVEL_MAP).fillna(-1).astype(int)
    df["scenario_encoded"]       = df["workload_scenario"].map(SCENARIO_MAP).fillna(-1).astype(int)
    df["strategy_encoded"]       = df["scenario_strategy"].map(STRATEGY_MAP).fillna(-1).astype(int)

    # Convert pqc_enabled to int
    if "pqc_enabled" in df.columns:
        df["pqc_enabled"] = df["pqc_enabled"].astype(int)

    # Some data cleaning depending on column format
    if "renewable_share_percent" in df.columns and df["renewable_share_percent"].dtype == object:
        df["renewable_share_percent"] = df["renewable_share_percent"].str.replace('%', '').astype(float)

    log.info(f"PUE added          → avg={df['PUE'].mean():.3f}  "
             f"min={df['PUE'].min():.2f}  max={df['PUE'].max():.2f}")
    log.info(f"CO2 added          → total={df['co2_kg'].sum():,.0f} kg")
    log.info(f"Zombie servers     → {df['is_zombie'].sum()} flagged")
    return df


# ── Step 3: Save ──────────────────────────────────────────────
def save(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(PROC_DATA_PATH), exist_ok=True)
    df.to_csv(PROC_DATA_PATH, index=False)
    log.info(f"Processed data saved → {PROC_DATA_PATH}")


# ── Public API ────────────────────────────────────────────────
def run_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Returns the engineered DataFrame and saves it to disk.
    """
    df = clean(df)
    df = engineer_features(df)
    save(df)
    return df
