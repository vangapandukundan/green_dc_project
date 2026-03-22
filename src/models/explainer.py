"""
src/models/explainer.py — SHAP Explainability
===============================================
Computes SHAP values for a fitted model and returns
everything the dashboard needs to render SHAP plots:
  • shap_values  (numpy array)
  • explainer    (shap.Explainer object)
  • feature_names

SHAP tells us WHY a model made a specific prediction —
which features pushed the output up or down.
"""

import os, sys
import numpy  as np
import pandas as pd
import shap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config           import FEATURE_COLS
from src.utils.logger import get_logger

log = get_logger(__name__)


def compute_shap(model, X: pd.DataFrame, sample_size: int = 300):
    """
    Compute SHAP values for the given model and data.

    Uses TreeExplainer for tree-based models (fast),
    falls back to KernelExplainer for others (slow).

    Parameters
    ----------
    model       : fitted sklearn-compatible model
    X           : feature DataFrame (usually X_test)
    sample_size : max rows to compute SHAP (keeps it fast)

    Returns
    -------
    explainer   : shap.Explainer
    shap_values : np.ndarray  shape (n_samples, n_features)
    X_sample    : pd.DataFrame  (the rows used for SHAP)
    """
    # Keep computation fast — sample rows if needed
    if len(X) > sample_size:
        X_sample = X.sample(sample_size, random_state=42).reset_index(drop=True)
    else:
        X_sample = X.reset_index(drop=True)

    log.info(f"Computing SHAP values on {len(X_sample)} samples...")

    try:
        # Fast path — works for RF, XGB, LGBM, GBM
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception:
        # Fallback for any non-tree model
        log.warning("TreeExplainer failed — falling back to KernelExplainer (slower).")
        background  = shap.sample(X_sample, 50)
        explainer   = shap.KernelExplainer(model.predict, background)
        shap_values = explainer.shap_values(X_sample)

    # For multi-output SHAP, take first output
    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    log.info("SHAP computation complete ✅")
    return explainer, shap_values, X_sample


def mean_abs_shap(shap_values: np.ndarray,
                  feature_names: list) -> pd.DataFrame:
    """
    Return a DataFrame of mean |SHAP| per feature, sorted descending.
    Useful for a bar chart of global feature importance.
    """
    importance = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": importance})
        .sort_values("Mean |SHAP|", ascending=False)
        .reset_index(drop=True)
    )
