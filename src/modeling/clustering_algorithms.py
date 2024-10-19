# src/model_training/clustering_algorithms.py

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture


def run_kmeans(df: pd.DataFrame, n_clusters: int, return_centers: bool = False, random_state=42):
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = kmeans.fit_predict(df)
    if return_centers:
        return labels, kmeans.cluster_centers_
    else:
        return labels

def run_dbscan(df: pd.DataFrame, eps: float, min_samples: int, return_core_sample_indices: bool = False):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df)
    if return_core_sample_indices:
        return labels, dbscan.core_sample_indices_
    else:
        return labels

def run_gmm(df: pd.DataFrame, n_components: int, return_means: bool = False, random_state=42):
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    labels = gmm.fit_predict(df)
    if return_means:
        return labels, gmm.means_
    else:
        return labels
