"""
src/models/unsupervised.py — Anomaly Detection & Clustering
=============================================================
Phase 2 Addition:
Finds strange server behavior using Isolation Forest.
Groups servers into distinct behavioral clusters using KMeans + PCA.
"""

import os, sys
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster  import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from config import CLUSTERING_CLUSTERS, ANOMALY_CONTAMINATION, RANDOM_STATE
from src.utils.logger import get_logger

log = get_logger(__name__)


def detect_anomalies(df: pd.DataFrame, features: list):
    """
    Train an Isolation Forest to flag weird servers.
    Returns the dataframe with a new 'anomaly_score' and 'is_anomaly' column,
    along with the fitted model.
    """
    log.info(f"Running Isolation Forest Anomaly Detection...")
    X = df[features].copy()
    
    # Scale features for Isolation Forest
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    iso = IsolationForest(
        n_estimators=100,
        contamination=ANOMALY_CONTAMINATION,
        random_state=RANDOM_STATE
    )
    
    # Fit and predict (-1 for anomaly, 1 for normal)
    preds = iso.fit_predict(X_scaled)
    scores = iso.decision_function(X_scaled)
    
    df_out = df.copy()
    df_out["anomaly_score"] = scores
    df_out["is_anomaly"] = (preds == -1).astype(int)
    
    log.info(f"  ✅ Flagged {df_out['is_anomaly'].sum()} anomalous servers.")
    return df_out, iso, scaler


def run_clustering(df: pd.DataFrame, features: list):
    """
    Groups servers into N clusters based on behavior.
    Reduces dimensions to 3D with PCA for easy visualization.
    Returns the dataframe with 'cluster', 'pca1', 'pca2', 'pca3' cols,
    and all fitted models.
    """
    log.info(f"Running KMeans Clustering ({CLUSTERING_CLUSTERS} clusters)...")
    X = df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(
        n_clusters=CLUSTERING_CLUSTERS, 
        random_state=RANDOM_STATE, 
        n_init=10
    )
    clusters = kmeans.fit_predict(X_scaled)
    
    # Reduce to 3 components so we can plot it in 3D in the dashboard
    log.info("Running PCA down to 3 dimensions for 3D visualization...")
    pca = PCA(n_components=3, random_state=RANDOM_STATE)
    pca_comps = pca.fit_transform(X_scaled)
    
    df_out = df.copy()
    df_out["cluster"] = clusters
    df_out["pca1"] = pca_comps[:, 0]
    df_out["pca2"] = pca_comps[:, 1]
    df_out["pca3"] = pca_comps[:, 2]
    
    log.info("  ✅ Clustering & PCA complete.")
    return df_out, kmeans, pca, scaler
