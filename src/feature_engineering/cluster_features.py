# src/feature_engineering/cluster_features.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from typing import Dict, Any

def generate_cluster_features(data: pd.DataFrame, clustering_model: BaseEstimator, params: Dict[str, Any]) -> pd.DataFrame:
    """
    Create cluster features by applying the user-selected clustering algorithm and configuration to the data.

    Args:
    data (pd.DataFrame): The dataset to be clustered.
    clustering_model (BaseEstimator): A clustering model object configured by the user (e.g., K-Means).
    params (Dict[str, Any]): A dictionary containing parameters necessary for the clustering model.

    Returns:
    pd.DataFrame: The input DataFrame with an additional column named 'cluster_assignment'.
    """
    try:
        # Configure the clustering model with the provided parameters
        model = clustering_model.set_params(**params)
        
        # Fit the model and predict cluster labels
        cluster_labels = model.fit_predict(data)
        
        # Add cluster labels as a new column
        data_with_clusters = data.copy()
        data_with_clusters['cluster_assignment'] = cluster_labels
        
        return data_with_clusters
    
    except Exception as e:
        print(f"Error in generate_cluster_features: {str(e)}")
        # In case of error, return the original dataset
        return data