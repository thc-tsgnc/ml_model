# src/model_training/clustering_algorithms.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN

def run_kmeans(df: pd.DataFrame, n_clusters: int) -> np.ndarray:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    return kmeans.fit_predict(df)

def run_dbscan(df: pd.DataFrame, eps: float, min_samples: int) -> np.ndarray:
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    return dbscan.fit_predict(df)