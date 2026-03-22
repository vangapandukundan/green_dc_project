"""
src/dashboard/components/model_comparison.py
Model Benchmark tab — shows all 4 models side-by-side
"""

import sys, os
import plotly.graph_objects as go
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def render(results_df: pd.DataFrame, best_model_name: str):
    st.header("📊 Model Comparison — 4 Algorithms Benchmarked")
    st.markdown(
        "All four models are trained on the **same data** and evaluated "
        "with **K-Fold Cross-Validation** for a fair comparison."
    )

    # ── Metrics Table ─────────────────────────────────────────
    st.subheader("📋 Performance Metrics Table")

    # Highlight the best row
    def highlight_best(row):
        style = [""] * len(row)
        if row["Model"] == best_model_name:
            style = ["background-color: #1e4d2b; color: white; font-weight: bold"] * len(row)
        return style

    styled_df = results_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df, use_container_width=True)

    st.caption(f"🏆 Best model: **{best_model_name}** (highlighted in green)")

    # ── R² Bar Chart ──────────────────────────────────────────
    st.subheader("🎯 R² Score Comparison")
    bar_colors = [
        "#27ae60" if m == best_model_name else "#3498db"
        for m in results_df["Model"]
    ]
    fig_r2 = go.Figure(go.Bar(
        x=results_df["Model"],
        y=results_df["R²"],
        marker_color=bar_colors,
        text=results_df["R²"].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
    ))
    fig_r2.update_layout(
        yaxis=dict(range=[0, 1.05], title="R² Score"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=350,
    )
    st.plotly_chart(fig_r2, use_container_width=True)

    # ── MAE & RMSE Grouped Bar ────────────────────────────────
    st.subheader("📉 Error Metrics (lower = better)")
    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(
        name="MAE (kWh)",
        x=results_df["Model"],
        y=results_df["MAE (kWh)"],
        marker_color="#e74c3c",
    ))
    fig_err.add_trace(go.Bar(
        name="RMSE (kWh)",
        x=results_df["Model"],
        y=results_df["RMSE (kWh)"],
        marker_color="#e67e22",
    ))
    fig_err.update_layout(
        barmode="group",
        yaxis_title="Error (kWh)",
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=350,
        legend=dict(bgcolor="#1c1c2e"),
    )
    st.plotly_chart(fig_err, use_container_width=True)

    # ── CV R² with Error Bars ─────────────────────────────────
    st.subheader("🔁 Cross-Validation R² (Mean ± Std Dev)")
    st.markdown(
        "Cross-validation avoids overfitting bias — "
        "a **high mean with low std** is the goal."
    )
    fig_cv = go.Figure(go.Bar(
        x=results_df["Model"],
        y=results_df["CV R² Mean"],
        error_y=dict(type="data", array=results_df["CV R² Std"], visible=True),
        marker_color="#9b59b6",
        text=results_df["CV R² Mean"].apply(lambda v: f"{v:.4f}"),
        textposition="outside",
    ))
    fig_cv.update_layout(
        yaxis=dict(range=[0, 1.05], title="CV R² Mean"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=350,
    )
    st.plotly_chart(fig_cv, use_container_width=True)
