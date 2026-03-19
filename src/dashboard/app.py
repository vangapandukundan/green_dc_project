"""
src/dashboard/app.py — Main Streamlit Dashboard
=================================================
Launch with:
    streamlit run src/dashboard/app.py

Requires train.py to have been run first so that
models/model_results.pkl exists.

Tab layout:
    1. ⚡ What-If Predictor
    2. 📊 Model Comparison
    3. 🧠 SHAP Explainability
    4. 🌍 City CO₂ Analysis
    5. 🧟 Zombie Servers
    6. 🤖 Unsupervised Analysis
"""

import os, sys, pickle
import pandas as pd
import streamlit as st

# ── Resolve project root so all imports work regardless of CWD ─
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from config import RESULTS_PATH, PROC_DATA_PATH

# ── Dashboard components ───────────────────────────────────────
from src.dashboard.components import (
    predictor,
    model_comparison,
    explainability,
    city_analysis,
    zombie_analysis,
    unsupervised_analysis,
)

# ══════════════════════════════════════════════════════════════
#  Page Config
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Green DC — Phase 2 Dashboard",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for a dark, premium feel ───────────────────────
st.markdown("""
<style>
    /* Dark gradient background */
    .stApp { background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 100%); }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: #0d1b2a;
        border-radius: 12px;
        padding: 6px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        color: #a0aec0;
        font-weight: 600;
        padding: 8px 20px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #1e4d2b, #27ae60);
        color: white !important;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1c2333;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 16px;
    }

    /* Headers */
    h1 { color: #27ae60 !important; }
    h2, h3 { color: #2ecc71 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  Load Cached Artifacts
# ══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading model artifacts…", ttl=None)
def load_results():
    if not os.path.exists(RESULTS_PATH):
        return None
    with open(RESULTS_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data(show_spinner="Loading processed data…")
def load_data():
    if not os.path.exists(PROC_DATA_PATH):
        return None
    return pd.read_csv(PROC_DATA_PATH)


results = load_results()
df      = load_data()

# ══════════════════════════════════════════════════════════════
#  Header
# ══════════════════════════════════════════════════════════════
st.title("🌿 Green Computing Data Center — Phase 2 Dashboard")
st.markdown(
    "**Multi-model ML** • **K-Fold CV** • "
    "**SHAP Explainability** • **Unsupervised ML (Isolation Forest & K-Means)**"
)

# ── Guard: require train.py to run first ──────────────────────
if results is None or df is None:
    st.error(
        "❌ No trained models found!\n\n"
        "Please run the training pipeline first:\n\n"
        "```bash\n"
        "python train.py\n"
        "```\n\n"
        "Or with Optuna tuning:\n\n"
        "```bash\n"
        "python train.py --tune\n"
        "```"
    )
    st.stop()

# ── Unpack results ────────────────────────────────────────────
fitted_models    = results["fitted_models"]
best_model_name  = results["best_model_name"]
best_model       = results["best_model"]
results_df       = results["results_df"]
shap_values      = results["shap_values"]
X_shap           = results["X_shap"]
shap_importance  = results["shap_importance"]
df_unsup         = results.get("df_unsup", None)

# ── Sidebar — quick stats ─────────────────────────────────────
with st.sidebar:
    st.header("📌 Pipeline Summary")
    st.success(f"🏆 Best Model: **{best_model_name}**")
    best_row = results_df[results_df["Model"] == best_model_name].iloc[0]
    st.metric("Best R²",   f"{best_row['R²']:.4f}")
    st.metric("Best MAE",  f"{best_row['MAE (kWh)']:,.1f} kWh")
    st.metric("CV R²",     f"{best_row['CV R² Mean']:.4f} ± {best_row['CV R² Std']:.4f}")
    st.divider()
    st.markdown(f"📊 Dataset rows: `{len(df):,}`")
    st.markdown(f"🧟 Zombie servers: `{df['is_zombie'].sum():,}`")
    st.markdown(f"🌿 Total CO₂: `{df['co2_kg'].sum():,.0f} kg`")
    st.divider()
    st.caption("Run `python train.py --tune` to enable Optuna hyperparameter tuning.")

# ══════════════════════════════════════════════════════════════
#  Tabs
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "⚡ What-If Predictor",
    "📊 Model Comparison",
    "🧠 SHAP Explainability",
    "🌍 City CO₂ Analysis",
    "🧟 Zombie Servers",
    "🤖 Unsupervised ML Analysis",
])

with tab1:
    compute, storage, network, workload_enc, pred_kwh, pred_co2 = \
        predictor.render(best_model, df)

with tab2:
    model_comparison.render(results_df, best_model_name)

with tab3:
    explainability.render(shap_values, X_shap, shap_importance, best_model_name)

with tab4:
    city_analysis.render(best_model, df, compute, storage, network,
                         workload_enc, pred_co2)

with tab5:
    zombie_analysis.render(df)

with tab6:
    if df_unsup is not None:
        unsupervised_analysis.render(df_unsup)
    else:
        st.error("❌ Unsupervised ML artifacts not found. Please re-run: `python train.py`")
