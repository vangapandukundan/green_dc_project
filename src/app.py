import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt

# ── Load model and data ───────────────────────────────────────
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    return pd.read_csv("data/processed_data.csv")

model = load_model()
df    = load_data()

# ── Page config ───────────────────────────────────────────────
st.set_page_config(page_title="Green DC Dashboard", layout="wide")
st.title("🌿 Green Computing Data Center Dashboard")
st.markdown("Predict energy usage, track CO₂, and identify zombie servers.")

# ── SECTION 1: What-If Energy Predictor ──────────────────────
st.header("⚡ What-If Energy Predictor")
st.markdown("Move the sliders to simulate different workloads.")

col1, col2 = st.columns(2)

with col1:
    compute  = st.slider("Compute Demand (TFlops)", 10.0, 500.0, 200.0)
    storage  = st.slider("Storage Demand (TB)",      1.0, 200.0,  50.0)

with col2:
    network  = st.slider("Network Demand (Gbps)",    1.0, 100.0,  30.0)
    workload = st.selectbox("Workload Type", [
        "Cloud Storage", "Database Queries",
        "IoT Processing", "Web Hosting"
    ])

# workload encoding must match data_prep.py category order
workload_map = {
    "Cloud Storage"    : 0,
    "Database Queries" : 1,
    "IoT Processing"   : 2,
    "Web Hosting"      : 3
}
workload_enc = workload_map[workload]

# ── Predict using exactly 4 features (same as model.py) ──────
X_input       = pd.DataFrame([[compute, storage, network, workload_enc]],
                  columns=["compute_demand_TFlops", "storage_demand_TB",
                            "network_demand_Gbps",  "workload_encoded"])
predicted_kwh = model.predict(X_input)[0]
co2_kg        = predicted_kwh * 0.82
pue           = 1.0 + (1.0 - compute / 500.0) * 0.8   # estimated PUE

st.subheader("🔮 Prediction Results")
m1, m2, m3 = st.columns(3)
m1.metric("Predicted Energy", f"{predicted_kwh:,.0f} kWh")
m2.metric("CO₂ Emitted",      f"{co2_kg:,.0f} kg")
m3.metric("Estimated PUE",    f"{pue:.2f}")

# ── SECTION 2: City Location What-If ─────────────────────────
st.header("🌍 City Location What-If")
st.markdown("How much CO₂ do we save by moving to a cooler city?")

city_temps = {
    "Chennai (Hot)"    : 35,
    "Mumbai"           : 32,
    "Delhi"            : 28,
    "Hyderabad"        : 30,
    "Bangalore (Cool)" : 22,
}

city      = st.selectbox("Select City", list(city_temps.keys()))
temp      = city_temps[city]

# Cooler city → servers run more efficiently → ~1% less energy per degree cooler
temp_factor = 1.0 - ((35 - temp) * 0.01)
adj_input   = pd.DataFrame([[compute, storage, network, workload_enc]],
                columns=["compute_demand_TFlops", "storage_demand_TB",
                          "network_demand_Gbps",  "workload_encoded"])
adj_kwh     = model.predict(adj_input)[0] * max(0.7, temp_factor)
adj_co2     = adj_kwh * 0.82
co2_saved   = co2_kg - adj_co2

c1, c2, c3 = st.columns(3)
c1.metric("City Temp",       f"{temp}°C")
c2.metric("Adjusted Energy", f"{adj_kwh:,.0f} kWh")
c3.metric("CO₂ Saved",       f"{max(0, co2_saved):,.0f} kg",
          delta=f"-{max(0, co2_saved):,.0f} kg")

# ── SECTION 3: Zombie Server Analysis ────────────────────────
st.header("🧟 Zombie Server Analysis")
st.markdown("Servers with **high energy** but **low compute demand** — pure waste.")

zombies = df[df["is_zombie"] == 1][
    ["workload_type", "compute_demand_TFlops",
     "energy_consumption_kWh", "co2_kg"]
].head(20)

st.dataframe(zombies, width="stretch")
total_zombie_co2 = df[df["is_zombie"] == 1]["co2_kg"].sum()
st.warning(
    f"⚠️ {len(df[df['is_zombie']==1])} zombie servers found — "
    f"wasting {total_zombie_co2:,.0f} kg CO₂!"
)

# ── SECTION 4: Average Energy by Workload Type ───────────────
st.header("📊 Average Energy by Workload Type")

avg_energy = df.groupby("workload_type")["energy_consumption_kWh"].mean().sort_values()
labels = list(avg_energy.index)
values = [float(v) for v in avg_energy.values]
fig, ax = plt.subplots(figsize=(8, 4))
ax.barh(labels, values, color="#27ae60", edgecolor="black", linewidth=0.5)
ax.set_xlabel("Avg Energy (kWh)")
ax.set_title("Which workload consumes the most energy?")
plt.tight_layout()
st.pyplot(fig)

# ── SECTION 5: CO2 Distribution ──────────────────────────────
st.header("🌿 CO₂ Emissions Distribution")

fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.hist(df["co2_kg"], bins=40, color="#3498db", edgecolor="black", linewidth=0.3)
ax2.set_xlabel("CO₂ (kg)")
ax2.set_ylabel("Count")
ax2.set_title("Distribution of CO₂ Emissions Across All Servers")
plt.tight_layout()
st.pyplot(fig2)