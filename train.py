"""
train.py — One-Click Training Pipeline
========================================
Run this script ONCE before launching the dashboard:

    python train.py              # train with default params
    python train.py --tune       # run Optuna tuning first (slower, better)

What it does:
  1. Load raw CSV
  2. Preprocess & engineer features
  3. (Optional) Tune hyperparameters with Optuna
  4. Train all 4 models
  5. Evaluate with K-Fold cross-validation
  6. Compute SHAP values for the best model
  7. Save everything to models/model_results.pkl
"""

import os, sys, argparse, pickle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config                   import (FEATURE_COLS, TARGET_COL,
                                       MODELS_DIR, RESULTS_PATH, OUTPUTS_DIR)
from src.data.loader          import load_raw_data
from src.data.preprocessor    import run_pipeline
from src.models.trainer       import train_all, get_splits
from src.models.evaluator     import evaluate_all, get_best_model
from src.models.tuner         import tune_model
from src.models.explainer     import compute_shap, mean_abs_shap
from src.models.unsupervised  import detect_anomalies, run_clustering
from src.utils.logger         import get_logger

import pandas as pd

log = get_logger("train")


def main(tune: bool = False):
    log.info("=" * 60)
    log.info("  GREEN DC — Training Pipeline  ")
    log.info("=" * 60)

    # ── 1. Load & Preprocess ──────────────────────────────────
    log.info("\n[1/6] Loading & preprocessing data...")
    raw_df = load_raw_data()
    df     = run_pipeline(raw_df)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = get_splits(df)

    # ── 2. Optional Optuna Tuning ─────────────────────────────
    tuned_params = None
    if tune:
        log.info("\n[2/6] Hyperparameter tuning with Optuna...")
        # Quick pre-train to find best model first, then tune it
        quick_models = train_all(X_train, y_train)
        quick_eval   = evaluate_all(quick_models, X_test, y_test, X, y)
        best_name, _ = get_best_model(quick_models, quick_eval)

        best_params  = tune_model(best_name, X_train, y_train)
        tuned_params = {best_name: best_params}
        log.info(f"Tuning complete. Best params: {best_params}")
    else:
        log.info("\n[2/6] Skipping tuning (use --tune to enable)")

    # ── 3. Train All Models ───────────────────────────────────
    log.info("\n[3/6] Training all models...")
    fitted_models = train_all(X_train, y_train, tuned_params)

    # ── 4. Evaluate ───────────────────────────────────────────
    log.info("\n[4/6] Evaluating with K-Fold cross-validation...")
    results_df = evaluate_all(fitted_models, X_test, y_test, X, y)
    best_name, best_model = get_best_model(fitted_models, results_df)

    # ── 5. SHAP Explainability ────────────────────────────────
    log.info(f"\n[5/6] Computing SHAP values for best model ({best_name})...")
    explainer, shap_values, X_shap = compute_shap(best_model, X_test)
    shap_importance = mean_abs_shap(shap_values, FEATURE_COLS)
    log.info(f"\nSHAP Feature Importance:\n{shap_importance.to_string(index=False)}")

    # ── 6. Unsupervised Machine Learning (Phase 2) ────────────
    log.info("\n[6/7] Running Unsupervised Anomaly Detection & Clustering...")
    # We pass the full processed dataframe so we have the raw states mapped to clusters
    df_unsup, iso_model, iso_scaler = detect_anomalies(df, FEATURE_COLS)
    df_unsup, kmeans_model, pca_model, kmeans_scaler = run_clustering(df_unsup, FEATURE_COLS)

    # ── 7. Save Everything ────────────────────────────────────
    log.info("\n[7/7] Saving model artifacts...")
    os.makedirs(MODELS_DIR,  exist_ok=True)
    os.makedirs(OUTPUTS_DIR, exist_ok=True)

    payload = {
        # Models
        "fitted_models"   : fitted_models,
        "best_model_name" : best_name,
        "best_model"      : best_model,

        # Evaluation
        "results_df"      : results_df,

        # SHAP
        "shap_values"     : shap_values,
        "X_shap"          : X_shap,
        "shap_importance" : shap_importance,

        # Data splits (for dashboard re-use)
        "X_test"          : X_test,
        "y_test"          : y_test,
        "feature_cols"    : FEATURE_COLS,

        # Unsupervised ML (Phase 2)
        "df_unsup"        : df_unsup,
        "iso_model"       : iso_model,
        "kmeans_model"    : kmeans_model,
        "pca_model"       : pca_model,
    }

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump(payload, f)

    log.info(f"Saved → {RESULTS_PATH}")
    log.info("\n✅ Training pipeline complete! Run the dashboard with:")
    log.info("   streamlit run src/dashboard/app.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Green DC Training Pipeline")
    parser.add_argument("--tune", action="store_true",
                        help="Run Optuna hyperparameter tuning before training")
    args = parser.parse_args()
    main(tune=args.tune)
