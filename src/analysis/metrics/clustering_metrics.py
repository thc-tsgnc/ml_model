# src/data_processing/metrics/clustering_metrics.py

import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from typing import Dict  # Import Dict from typing module


def calculate_internal_metrics(df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    return {
        'silhouette_score': silhouette_score(df, labels),
        'davies_bouldin_score': davies_bouldin_score(df, labels),
        'calinski_harabasz_score': calinski_harabasz_score(df, labels)
    }

def calculate_external_metrics(df: pd.DataFrame, labels: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    return {
        'adjusted_rand_score': adjusted_rand_score(true_labels, labels),
        'normalized_mutual_info_score': normalized_mutual_info_score(true_labels, labels),
        'v_measure_score': v_measure_score(true_labels, labels)
    }