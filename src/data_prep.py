import pandas as pd
import numpy as np
import os

def load_data():
    path = os.path.join("data", "green_quantum_data_centers_2.csv")
    df = pd.read_csv(path)
    print(f"✅ Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"\n📋 Columns found:\n{list(df.columns)}\n")
    return df


def clean_data(df):
    before = len(df)
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)
    print(f"🧹 Cleaned: {before - len(df)} rows removed → {len(df)} rows remaining\n")
    return df


def engineer_features(df):
    # --- PUE Calculation (variable, based on compute efficiency) ---
    # Higher compute demand = better server utilization = lower PUE
    # Range: PUE between 1.2 and 2.0 (realistic data center range)
    max_compute = df["compute_demand_TFlops"].max()
    efficiency  = df["compute_demand_TFlops"] / max_compute  # 0 to 1
    df["it_energy_kWh"] = df["energy_consumption_kWh"] * (0.50 + 0.20 * efficiency)
    df["PUE"]           = (df["energy_consumption_kWh"] / df["it_energy_kWh"]).round(4)

    # --- CO2 Emissions (India grid = 0.82 kg per kWh) ---
    df["co2_kg"] = (df["energy_consumption_kWh"] * 0.82).round(4)

    # --- Zombie Server Flag ---
    # High energy but low compute = wasteful server
    energy_threshold  = df["energy_consumption_kWh"].quantile(0.75)
    compute_threshold = df["compute_demand_TFlops"].quantile(0.25)
    df["is_zombie"]   = (
        (df["energy_consumption_kWh"] > energy_threshold) &
        (df["compute_demand_TFlops"]  < compute_threshold)
    ).astype(int)

    # --- Encode workload_type (text to number) ---
    df["workload_encoded"] = df["workload_type"].astype("category").cat.codes

    print(f"⚡ PUE column added     → avg PUE = {df['PUE'].mean():.3f}  "
          f"(min={df['PUE'].min():.2f}, max={df['PUE'].max():.2f})")
    print(f"🌿 CO2 column added     → total CO2 = {df['co2_kg'].sum():,.0f} kg")
    print(f"🧟 Zombie servers found → {df['is_zombie'].sum()} servers flagged\n")

    return df


def save_data(df):
    out_path = os.path.join("data", "processed_data.csv")
    df.to_csv(out_path, index=False)
    print(f"💾 Saved → {out_path}")
    print(f"\n📊 Preview:")
    print(df[["workload_type", "compute_demand_TFlops",
              "energy_consumption_kWh", "PUE",
              "co2_kg", "is_zombie"]].head(5))


if __name__ == "__main__":
    df = load_data()
    df = clean_data(df)
    df = engineer_features(df)
    save_data(df)
    print("\n✅ data_prep.py completed successfully!")