"""
src/models/tuner.py — Optuna Hyperparameter Tuning
====================================================
Uses Optuna (Bayesian optimization) to find the best
hyperparameters for the best-performing model (decided
by the evaluator).

Only the best model is tuned to keep training time practical.
Returns a dict of {model_name → best_params}.
"""

import os, sys
import optuna
import pandas as pd

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from xgboost                 import XGBRegressor
from lightgbm                import LGBMRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config           import RANDOM_STATE, CV_FOLDS, OPTUNA_TRIALS, OPTUNA_TIMEOUT
from src.utils.logger import get_logger

# Suppress Optuna verbose output
optuna.logging.set_verbosity(optuna.logging.WARNING)

log = get_logger(__name__)


# ── Objective functions per model ─────────────────────────────

def _rf_objective(trial, X, y):
    params = {
        "n_estimators" : trial.suggest_int("n_estimators", 100, 400),
        "max_depth"    : trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "random_state" : RANDOM_STATE,
    }
    model  = RandomForestRegressor(**params)
    scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="r2", n_jobs=-1)
    return scores.mean()


def _xgb_objective(trial, X, y):
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators", 100, 400),
        "max_depth"        : trial.suggest_int("max_depth", 3, 10),
        "learning_rate"    : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample"        : trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state"     : RANDOM_STATE,
        "verbosity"        : 0,
    }
    model  = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="r2", n_jobs=-1)
    return scores.mean()


def _lgbm_objective(trial, X, y):
    params = {
        "n_estimators"  : trial.suggest_int("n_estimators", 100, 400),
        "max_depth"     : trial.suggest_int("max_depth", 3, 12),
        "learning_rate" : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves"    : trial.suggest_int("num_leaves", 20, 80),
        "random_state"  : RANDOM_STATE,
        "verbose"       : -1,
    }
    model  = LGBMRegressor(**params)
    scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="r2", n_jobs=-1)
    return scores.mean()


def _gb_objective(trial, X, y):
    params = {
        "n_estimators"  : trial.suggest_int("n_estimators", 100, 300),
        "max_depth"     : trial.suggest_int("max_depth", 3, 8),
        "learning_rate" : trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "random_state"  : RANDOM_STATE,
    }
    model  = GradientBoostingRegressor(**params)
    scores = cross_val_score(model, X, y, cv=CV_FOLDS, scoring="r2", n_jobs=-1)
    return scores.mean()


# ── Dispatcher ────────────────────────────────────────────────

_OBJECTIVES = {
    "RandomForest"     : _rf_objective,
    "XGBoost"          : _xgb_objective,
    "LightGBM"         : _lgbm_objective,
    "GradientBoosting" : _gb_objective,
}


def tune_model(model_name: str,
               X_train:    pd.DataFrame,
               y_train:    pd.Series) -> dict:
    """
    Run Optuna study for the given model.

    Returns
    -------
    dict : best hyperparameters found
    """
    if model_name not in _OBJECTIVES:
        log.warning(f"No tuning objective for '{model_name}'. Skipping.")
        return {}

    log.info(f"Starting Optuna tuning for {model_name} "
             f"({OPTUNA_TRIALS} trials / {OPTUNA_TIMEOUT}s timeout)...")

    objective = _OBJECTIVES[model_name]
    study     = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(trial, X_train, y_train),
        n_trials=OPTUNA_TRIALS,
        timeout=OPTUNA_TIMEOUT,
        show_progress_bar=False,
    )

    best = study.best_params
    log.info(f"Best params for {model_name}: {best}")
    log.info(f"Best CV R²: {study.best_value:.4f}")
    return best
