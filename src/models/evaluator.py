"""
src/models/evaluator.py — Model Evaluation with K-Fold CV
===========================================================
Computes per-model:
  • R² Score
  • Mean Absolute Error (MAE)
  • Root Mean Squared Error (RMSE)
  • K-Fold Cross-Validation R² (mean ± std)

Returns a tidy pandas DataFrame so the dashboard can
display it directly as a comparison table.
"""

import os, sys
import numpy  as np
import pandas as pd

from sklearn.metrics         import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config           import FEATURE_COLS, TARGET_COL, CV_FOLDS
from src.utils.logger import get_logger

log = get_logger(__name__)


def evaluate_all(fitted_models: dict,
                 X_test: pd.DataFrame,
                 y_test:  pd.Series,
                 X_full:  pd.DataFrame,
                 y_full:  pd.Series) -> pd.DataFrame:
    """
    Evaluate every model and return a comparison DataFrame.

    Parameters
    ----------
    fitted_models : {model_name → fitted model}
    X_test, y_test: held-out test set
    X_full, y_full: full dataset for cross-validation

    Returns
    -------
    pd.DataFrame with columns:
        Model | R2 | MAE | RMSE | CV_R2_Mean | CV_R2_Std
    """
    rows = []

    for name, model in fitted_models.items():
        preds = model.predict(X_test)

        r2   = r2_score(y_test, preds)
        mae  = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        # K-Fold cross-validation on full dataset
        cv_scores = cross_val_score(
            model, X_full, y_full,
            cv=CV_FOLDS, scoring="r2", n_jobs=-1
        )

        rows.append({
            "Model"      : name,
            "R²"         : round(r2, 4),
            "MAE (kWh)"  : round(mae, 2),
            "RMSE (kWh)" : round(rmse, 2),
            "CV R² Mean" : round(cv_scores.mean(), 4),
            "CV R² Std"  : round(cv_scores.std(), 4),
        })

        log.info(
            f"{name:20s}  R²={r2:.4f}  MAE={mae:,.1f}  "
            f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}"
        )

    results_df = pd.DataFrame(rows).sort_values("R²", ascending=False)
    log.info(f"\n{results_df.to_string(index=False)}")
    return results_df


def get_best_model(fitted_models: dict, results_df: pd.DataFrame):
    """Return the best-performing fitted model by R² score."""
    best_name = results_df.iloc[0]["Model"]
    log.info(f"Best model → {best_name}")
    return best_name, fitted_models[best_name]
