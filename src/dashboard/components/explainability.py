"""
src/dashboard/components/explainability.py
SHAP Explainability tab
"""

import sys, os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def render(shap_values: np.ndarray,
           X_shap: pd.DataFrame,
           shap_importance: pd.DataFrame,
           best_model_name: str):

    st.header("🧠 SHAP Explainability — Why Did the Model Predict That?")
    st.markdown(
        f"SHAP (SHapley Additive exPlanations) reveals **which features drive "
        f"predictions** for the **{best_model_name}** model. "
        f"Unlike feature importance from the model itself, SHAP is model-agnostic "
        f"and accounts for feature interactions."
    )

    # ── 1. Global Feature Importance Bar ─────────────────────
    st.subheader("🏆 Global Feature Importance (Mean |SHAP|)")
    st.markdown(
        "Larger bar = this feature has more influence on predictions on average."
    )
    fig_bar = go.Figure(go.Bar(
        x=shap_importance["Mean |SHAP|"],
        y=shap_importance["Feature"],
        orientation="h",
        marker=dict(
            color=shap_importance["Mean |SHAP|"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="SHAP Value"),
        ),
    ))
    fig_bar.update_layout(
        xaxis_title="Mean |SHAP Value| (average impact on output)",
        yaxis=dict(autorange="reversed"),
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        height=300,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── 2. SHAP Beeswarm (scatter per feature) ────────────────
    st.subheader("🐝 SHAP Beeswarm Plot — Distribution of Impacts")
    st.markdown(
        "Each dot = one data point. "
        "**Red dots (high feature value)** pushed prediction up or down. "
        "**Blue dots (low feature value)** have the opposite effect."
    )

    feature_names = X_shap.columns.tolist()
    rows = []
    for i, feat in enumerate(feature_names):
        sv  = shap_values[:, i]
        fv  = X_shap.iloc[:, i].values
        rows.extend([{
            "Feature"      : feat,
            "SHAP Value"   : s,
            "Feature Value": f,
        } for s, f in zip(sv, fv)])

    beedf = pd.DataFrame(rows)

    fig_bee = px.scatter(
        beedf,
        x="SHAP Value",
        y="Feature",
        color="Feature Value",
        color_continuous_scale="RdBu_r",
        height=400,
        opacity=0.6,
    )
    fig_bee.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
        coloraxis_colorbar=dict(title="Feature<br>Value"),
        yaxis=dict(categoryorder="array",
                   categoryarray=shap_importance["Feature"].tolist()[::-1]),
    )
    st.plotly_chart(fig_bee, use_container_width=True)

    # ── 3. SHAP Dependence for top feature ───────────────────
    top_feature = shap_importance.iloc[0]["Feature"]
    st.subheader(f"🔍 SHAP Dependence Plot — Top Feature: `{top_feature}`")
    st.markdown(
        f"Shows how `{top_feature}` values relate to their SHAP contributions. "
        "A clear trend here confirms this feature is genuinely predictive."
    )

    feat_idx = feature_names.index(top_feature)
    dep_df   = pd.DataFrame({
        "Feature Value": X_shap.iloc[:, feat_idx].values,
        "SHAP Value"   : shap_values[:, feat_idx],
    })

    fig_dep = px.scatter(
        dep_df,
        x="Feature Value",
        y="SHAP Value",
        color="SHAP Value",
        color_continuous_scale="RdYlGn",
        labels={"Feature Value": top_feature},
        height=350,
    )
    fig_dep.update_layout(
        plot_bgcolor="#0e1117",
        paper_bgcolor="#0e1117",
        font_color="white",
    )
    st.plotly_chart(fig_dep, use_container_width=True)

    # ── 4. Key Insights ───────────────────────────────────────
    st.subheader("💡 Key Insights")
    for _, row in shap_importance.iterrows():
        st.markdown(
            f"- **{row['Feature']}** → average impact of "
            f"`{row['Mean |SHAP|']:.2f}` kWh on each prediction"
        )
