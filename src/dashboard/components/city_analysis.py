"""
src/dashboard/components/city_analysis.py
City CO₂ What-If tab
"""

import sys, os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from config import CITY_TEMPS, CO2_KG_PER_KWH, FEATURE_COLS


def render(best_model, df, compute, storage, network, workload_enc, baseline_co2):
    st.header("🌍 City Location What-If — CO₂ vs. Geography")
    st.markdown(
        "Cooler cities reduce cooling overhead → servers run more efficiently → "
        "less energy wasted → fewer CO₂ emissions. "
        "Use the predictor tab sliders to set the workload."
    )

    city = st.selectbox("Select Data Center Location", list(CITY_TEMPS.keys()),
                        key="city_select")
    temp = CITY_TEMPS[city]

    # Cooler city → ~1% less energy per degree below 35°C
    temp_factor = max(0.70, 1.0 - ((35 - temp) * 0.01))
    # Phase 2: Start with medians of all 17 features, override the 4 what-if parameters
    medians = df[FEATURE_COLS].median().to_dict()
    medians["compute_demand_TFlops"] = compute
    medians["storage_demand_TB"]     = storage
    medians["network_demand_Gbps"]   = network
    medians["workload_encoded"]      = workload_enc
    
    X_adj = pd.DataFrame([medians], columns=FEATURE_COLS)
    adj_kwh  = best_model.predict(X_adj)[0] * temp_factor
    adj_co2  = adj_kwh * CO2_KG_PER_KWH
    co2_saved = baseline_co2 - adj_co2

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("City Temperature", f"{temp}°C")
    c2.metric("Temp Factor",       f"{temp_factor:.2f}×")
    c3.metric("Adjusted Energy",   f"{adj_kwh:,.0f} kWh")
    c4.metric("CO₂ Saved vs. Baseline",
              f"{max(0, co2_saved):,.0f} kg",
              delta=f"-{max(0, co2_saved):,.0f} kg" if co2_saved > 0 else None,
              delta_color="inverse")

    # ── All-city comparison bar chart ────────────────────────
    st.subheader("📊 CO₂ Comparison Across All Cities")
    city_results = []
    for c, t in CITY_TEMPS.items():
        tf  = max(0.70, 1.0 - ((35 - t) * 0.01))
        kwh = best_model.predict(X_adj)[0] * tf
        co2 = kwh * CO2_KG_PER_KWH
        city_results.append({"City": c, "Temp (°C)": t,
                              "Energy (kWh)": round(kwh), "CO₂ (kg)": round(co2)})

    city_df = pd.DataFrame(city_results).sort_values("CO₂ (kg)")

    fig = go.Figure(go.Bar(
        x=city_df["City"],
        y=city_df["CO₂ (kg)"],
        marker=dict(
            color=city_df["Temp (°C)"],
            colorscale="RdYlGn_r",
            showscale=True,
            colorbar=dict(title="Temp (°C)"),
        ),
        text=city_df["CO₂ (kg)"].apply(lambda v: f"{v:,} kg"),
        textposition="outside",
    ))
    fig.update_layout(
        yaxis_title="CO₂ Emissions (kg)",
        xaxis_title="City",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(city_df.set_index("City"), use_container_width=True)
