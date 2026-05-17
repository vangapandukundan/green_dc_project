#!/usr/bin/env python
"""Test script to diagnose dashboard issues"""

import os, sys
import traceback

# Add project root to path
ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

print("=" * 60)
print("TESTING DASHBOARD COMPONENTS")
print("=" * 60)

try:
    print("\n[1/6] Testing config import...")
    from config import RESULTS_PATH, PROC_DATA_PATH
    print(f"  ✓ Config loaded")
    print(f"  - RESULTS_PATH: {RESULTS_PATH}")
    print(f"  - PROC_DATA_PATH: {PROC_DATA_PATH}")
    print(f"  - Model file exists: {os.path.exists(RESULTS_PATH)}")
    
    print("\n[2/6] Testing data loading...")
    import pickle
    import pandas as pd
    
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "rb") as f:
            results = pickle.load(f)
        print(f"  ✓ Model results loaded")
        print(f"  - Keys: {list(results.keys())}")
    
    if os.path.exists(PROC_DATA_PATH):
        df = pd.read_csv(PROC_DATA_PATH)
        print(f"  ✓ Data loaded: {df.shape}")
    
    print("\n[3/6] Testing streamlit import...")
    import streamlit as st
    print(f"  ✓ Streamlit {st.__version__} imported")
    
    print("\n[4/6] Testing dashboard components...")
    from src.dashboard.components import (
        predictor,
        model_comparison,
        explainability,
        city_analysis,
        zombie_analysis,
        unsupervised_analysis,
    )
    print(f"  ✓ All dashboard components imported successfully")
    
    print("\n[5/6] Testing app module...")
    from src.dashboard import app
    print(f"  ✓ Dashboard app module imported")
    
    print("\n[6/6] All tests passed! ✓")
    print("\nDashboard should be ready to run.")
    
except Exception as e:
    print(f"\n✗ ERROR: {type(e).__name__}: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)
