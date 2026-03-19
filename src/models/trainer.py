"""
src/models/trainer.py — Multi-Model Trainer
=============================================
Trains four models on the same dataset:
  • RandomForestRegressor
  • XGBRegressor
  • LGBMRegressor
  • GradientBoostingRegressor

All hyperparameters come from config.MODEL_DEFAULTS.
Returns a dict of {model_name → fitted_model}.
"""

import os, sys
import pandas as pd

from sklearn.ensemble        import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from xgboost                 import XGBRegressor
from lightgbm                import LGBMRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config           import (FEATURE_COLS, TARGET_COL,
                               TEST_SIZE, RANDOM_STATE, MODEL_DEFAULTS)
from src.utils.logger import get_logger

log = get_logger(__name__)


def get_splits(df: pd.DataFrame):
    """Split processed DataFrame into train/test sets."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    log.info(f"Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")
    return X_train, X_test, y_train, y_test


def _build_models(params: dict = None) -> dict:
    """
    Instantiate all four model objects.

    Parameters
    ----------
    params : dict, optional
        If provided, should be {model_name: hyperparameter_dict}.
        Falls back to config.MODEL_DEFAULTS for missing entries.
    """
    cfg = MODEL_DEFAULTS.copy()
    if params:
        for name, p in params.items():
            cfg[name] = p                        # override with tuned params

    return {
        "RandomForest"     : RandomForestRegressor(**cfg["RandomForest"]),
        "XGBoost"          : XGBRegressor(**cfg["XGBoost"]),
        "LightGBM"         : LGBMRegressor(**cfg["LightGBM"]),
        "GradientBoosting" : GradientBoostingRegressor(**cfg["GradientBoosting"]),
    }


def train_all(X_train: pd.DataFrame, y_train: pd.Series,
              tuned_params: dict = None) -> dict:
    """
    Train all models and return a dict of fitted models.

    Parameters
    ----------
    X_train, y_train : train split
    tuned_params     : optional dict of tuned hyperparams from tuner.py

    Returns
    -------
    dict : {model_name → fitted sklearn-compatible model}
    """
    models = _build_models(tuned_params)
    fitted = {}

    for name, model in models.items():
        log.info(f"Training {name}...")
        model.fit(X_train, y_train)
        fitted[name] = model
        log.info(f"  ✅ {name} trained")

    return fitted
