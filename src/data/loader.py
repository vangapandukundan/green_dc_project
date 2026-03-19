"""
src/data/loader.py — Raw Data Loader
======================================
Responsible ONLY for reading the raw CSV from disk
and returning a validated DataFrame.
"""

import pandas as pd
import sys, os

# ── allow imports from project root ──────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config               import RAW_DATA_PATH
from src.utils.logger     import get_logger

log = get_logger(__name__)


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw Green DC dataset from disk.

    Returns
    -------
    pd.DataFrame
        Un-processed raw data.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at the configured path.
    """
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(
            f"Raw data not found at: {RAW_DATA_PATH}\n"
            "Please place the CSV in the data/ directory."
        )

    df = pd.read_csv(RAW_DATA_PATH)
    log.info(f"Loaded raw data → {df.shape[0]:,} rows × {df.shape[1]} columns")
    log.info(f"Columns: {list(df.columns)}")
    return df
