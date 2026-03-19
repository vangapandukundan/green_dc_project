"""
src/dashboard/components/predictor.py
What-If Energy Predictor tab
"""

import sys, os
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from config import WORKLOAD_MAP, CITY_TEMPS, CO2_KG_PER_KWH, FEATURE_COLS


def render(best_model, df):
    st.header("⚡ What-If Energy Predictor")
    st.markdown(
        "Simulate different workload scenarios and instantly see predicted "
        "energy usage, CO₂ emissions, and Power Usage Effectiveness (PUE)."
    )

    col1, col2 = st.columns(2)
    with col1:
        compute  = st.slider("Compute Demand (TFlops)", 10.0, 500.0, 200.0, step=5.0)
        storage  = st.slider("Storage Demand (TB)",      1.0, 200.0,  50.0, step=1.0)
    with col2:
        network  = st.slider("Network Demand (Gbps)",    1.0, 100.0,  30.0, step=1.0)
        workload = st.selectbox("Workload Type", list(WORKLOAD_MAP.keys()))

    workload_enc  = WORKLOAD_MAP[workload]
    
    # Phase 2: Start with median values for all 17 features
    medians = df[FEATURE_COLS].median().to_dict()
    
    # Overwrite the 4 features the user is interacting with
    medians["compute_demand_TFlops"] = compute
    medians["storage_demand_TB"]     = storage
    medians["network_demand_Gbps"]   = network
    medians["workload_encoded"]      = workload_enc
    
    X_input = pd.DataFrame([medians], columns=FEATURE_COLS)

    predicted_kwh = best_model.predict(X_input)[0]
    co2_kg        = predicted_kwh * CO2_KG_PER_KWH
    pue           = 1.0 + (1.0 - compute / 500.0) * 0.8

    st.subheader("🔮 Prediction Results")
    m1, m2, m3 = st.columns(3)
    m1.metric("Predicted Energy", f"{predicted_kwh:,.0f} kWh")
    m2.metric("CO₂ Emitted",      f"{co2_kg:,.0f} kg")
    m3.metric("Estimated PUE",    f"{pue:.2f}")

    # Efficiency rating
    st.divider()
    if pue < 1.3:
        st.success("🌟 Excellent PUE — this is a highly efficient configuration!")
    elif pue < 1.6:
        st.info("👍 Good PUE — room for improvement by increasing compute utilization.")
    else:
        st.warning("⚠️ Poor PUE — consider increasing compute demand to improve efficiency.")

    return compute, storage, network, workload_enc, predicted_kwh, co2_kg
