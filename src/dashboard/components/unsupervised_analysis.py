"""
src/dashboard/components/unsupervised_analysis.py
Phase 2: Anomaly Detection & Clustering Tabs
"""

import sys, os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def render(df_unsup: pd.DataFrame):
    st.header("🤖 Advanced Unsupervised ML Analysis")
    st.markdown(
        "Phase 2 Engine: **Isolation Forest** (Anomaly Detection) and **K-Means + PCA** (Clustering). "
        "These models find patterns without knowing the specific target variable."
    )

    t1, t2 = st.tabs(["🔴 Anomaly Detection (Isolation Forest)", "🌌 3D Server Clustering (K-Means)"])

    # ── TAB 1: Anomaly Detection ───────────────────────────────────────────
    with t1:
        st.subheader("🔴 Anomaly Detection — Isolation Forest")
        st.markdown(
            "Isolation Forest explicitly searches for data points that are 'few and different'. "
            "These servers exhibit extreme configurations compared to the rest of the fleet."
        )

        anomalies = df_unsup[df_unsup["is_anomaly"] == 1]
        normal    = df_unsup[df_unsup["is_anomaly"] == 0]

        col1, col2 = st.columns(2)
        col1.metric("Anomalous Servers Flagged", f"{len(anomalies)}", delta=f"{len(anomalies)/len(df_unsup)*100:.1f}% of fleet", delta_color="inverse")
        col2.metric("Average Anomaly Score", f"{anomalies['anomaly_score'].mean():.3f}", help="Lower score globally means more anomalous.")

        st.markdown("### 🔍 Energy vs Compute (Highlighting Anomalies)")
        
        # We plot energy vs compute, but color by anomaly
        df_plot = df_unsup.copy()
        df_plot["Status"] = df_plot["is_anomaly"].map({0: "Normal", 1: "🚨 Anomaly"})
        
        fig = px.scatter(
            df_plot,
            x="compute_demand_TFlops",
            y="energy_consumption_kWh",
            color="Status",
            color_discrete_map={"Normal": "#3498db", "🚨 Anomaly": "#e74c3c"},
            opacity=0.7,
            hover_data=["workload_type", "energy_source", "anomaly_score"],
            height=450
        )
        fig.update_layout(
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            legend=dict(title="Server State", bgcolor="#1c1c2e")
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📋 The Anomalous Servers (Raw Data)")
        st.dataframe(
            anomalies[["record_id", "workload_type", "compute_demand_TFlops", "energy_consumption_kWh", "anomaly_score", "renewable_share_percent"]]
            .sort_values("anomaly_score")
            .head(20),
            use_container_width=True
        )

    # ── TAB 2: Clustering ──────────────────────────────────────────────────
    with t2:
        st.subheader("🌌 3D Server Clustering — K-Means + PCA")
        st.markdown(
            "Servers have been grouped into **3 distinct behavioral clusters** using K-Means. "
            "Since we have 17 features, we used **PCA (Principal Component Analysis)** to reduce "
            "dimensions down to 3, so we can visualize them in a 3D space."
        )

        cluster_counts = df_unsup["cluster"].value_counts().sort_index()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Cluster 0 Size", f"{cluster_counts.get(0, 0)} servers")
        c2.metric("Cluster 1 Size", f"{cluster_counts.get(1, 0)} servers")
        c3.metric("Cluster 2 Size", f"{cluster_counts.get(2, 0)} servers")

        df_unsup["Cluster_Label"] = df_unsup["cluster"].apply(lambda x: f"Cluster {x}")

        fig3d = px.scatter_3d(
            df_unsup,
            x="pca1", y="pca2", z="pca3",
            color="Cluster_Label",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            opacity=0.8,
            hover_data=["compute_demand_TFlops", "energy_consumption_kWh", "workload_type"],
            height=600
        )
        
        fig3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            scene=dict(
                xaxis=dict(backgroundcolor="#0e1117", gridcolor="#2d3748"),
                yaxis=dict(backgroundcolor="#0e1117", gridcolor="#2d3748"),
                zaxis=dict(backgroundcolor="#0e1117", gridcolor="#2d3748"),
            ),
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
            legend=dict(bgcolor="#1c1c2e")
        )
        st.plotly_chart(fig3d, use_container_width=True)

        st.markdown("### 📊 What defines these clusters?")
        st.markdown("Average physical demands per cluster:")
        
        cluster_summary = df_unsup.groupby("cluster")[["compute_demand_TFlops", "storage_demand_TB", "network_demand_Gbps", "energy_consumption_kWh"]].mean()
        st.dataframe(cluster_summary.style.format("{:.2f}").background_gradient(cmap="viridis"), use_container_width=True)

