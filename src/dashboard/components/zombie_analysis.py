"""
src/dashboard/components/zombie_analysis.py
Zombie Server Analysis tab
"""

import sys, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def render(df: pd.DataFrame):
    st.header("🧟 Zombie Server Analysis")
    st.markdown(
        "**Zombie servers** = high energy consumption + low compute utilisation. "
        "They consume power but do little useful work — pure carbon waste."
    )

    zombies    = df[df["is_zombie"] == 1]
    non_zombie = df[df["is_zombie"] == 0]
    total_z    = len(zombies)
    z_co2      = zombies["co2_kg"].sum()
    z_pct      = total_z / len(df) * 100

    # ── KPI row ───────────────────────────────────────────────
    k1, k2, k3 = st.columns(3)
    k1.metric("Zombie Servers",   f"{total_z:,}",         delta=f"{z_pct:.1f}% of fleet",
              delta_color="inverse")
    k2.metric("CO₂ Wasted",       f"{z_co2:,.0f} kg",    delta="Pure waste")
    k3.metric("Energy Wasted",    f"{zombies['energy_consumption_kWh'].sum():,.0f} kWh",
              delta_color="inverse")

    st.error(
        f"⚠️ **{total_z} zombie servers** are wasting **{z_co2:,.0f} kg CO₂** "
        f"({z_pct:.1f}% of your fleet)!"
    )

    # ── Scatter: Energy vs Compute coloured by zombie ─────────
    st.subheader("🔵 Energy vs. Compute — Spot the Zombies")
    scatter_df = df[["compute_demand_TFlops", "energy_consumption_kWh",
                     "workload_type", "is_zombie"]].copy()
    scatter_df["Status"] = scatter_df["is_zombie"].map({0: "Normal", 1: "Zombie 🧟"})

    fig_sc = px.scatter(
        scatter_df,
        x="compute_demand_TFlops",
        y="energy_consumption_kWh",
        color="Status",
        color_discrete_map={"Normal": "#3498db", "Zombie 🧟": "#e74c3c"},
        hover_data=["workload_type"],
        opacity=0.6,
        labels={"compute_demand_TFlops" : "Compute Demand (TFlops)",
                "energy_consumption_kWh": "Energy (kWh)"},
        height=420,
    )
    fig_sc.update_layout(
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", legend=dict(bgcolor="#1c1c2e"),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Zombies by workload ────────────────────────────────────
    st.subheader("📊 Zombie Distribution by Workload Type")
    z_by_wl = (
        zombies.groupby("workload_type")
               .agg(count=("is_zombie", "count"), co2=("co2_kg", "sum"))
               .reset_index()
               .sort_values("count", ascending=False)
    )
    fig_wl = go.Figure(go.Bar(
        x=z_by_wl["workload_type"],
        y=z_by_wl["count"],
        marker_color="#e74c3c",
        text=z_by_wl["count"],
        textposition="outside",
    ))
    fig_wl.update_layout(
        yaxis_title="Number of Zombie Servers",
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", height=350,
    )
    st.plotly_chart(fig_wl, use_container_width=True)

    # ── Raw table ─────────────────────────────────────────────
    st.subheader("📋 Top 20 Zombie Servers")
    display = (
        zombies[["workload_type", "compute_demand_TFlops",
                 "energy_consumption_kWh", "co2_kg", "PUE"]]
        .sort_values("energy_consumption_kWh", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    st.dataframe(display, use_container_width=True)
