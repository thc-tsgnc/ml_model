# src/feature_engineering/feature_selector.py

import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_selection import VarianceThreshold
from typing import List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def remove_low_variance_features(df: pd.DataFrame, variance_threshold: float) -> List[int]:
    """
    Remove features with low variance from the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        variance_threshold (float): The variance threshold below which features will be removed.

    Returns:
        List[int]: Indices of features that pass the variance threshold.

    Raises:
        ValueError: If the input DataFrame is empty or if variance_threshold is negative.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if variance_threshold < 0:
        raise ValueError("Variance threshold must be non-negative.")

    try:
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit_transform(df)
        feature_indices = selector.get_support(indices=True)
        logger.info(f"Removed {df.shape[1] - len(feature_indices)} low variance features.")
        return feature_indices
    except Exception as e:
        logger.error(f"Error in remove_low_variance_features: {str(e)}")
        raise

def group_features_by_correlation(df: pd.DataFrame, n_groups: int) -> List[List[str]]:
    """
    Group features based on their correlation using Agglomerative Clustering.

    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        n_groups (int): Number of groups to cluster the features into.

    Returns:
        List[List[str]]: List of feature groups, where each group is a list of feature names.

    Raises:
        ValueError: If the input DataFrame is empty or if n_groups is less than 2.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if n_groups < 2:
        raise ValueError("Number of groups must be at least 2.")

    try:
        corr_matrix = df.corr().abs()
        clustering = AgglomerativeClustering(n_clusters=n_groups, metric='precomputed', linkage='average')
        clustering.fit(1 - corr_matrix)
        
        feature_groups = [[] for _ in range(n_groups)]
        for idx, label in enumerate(clustering.labels_):
            feature_groups[label].append(df.columns[idx])
        
        # Return only non-empty groups
        non_empty_groups = [group for group in feature_groups if group]
        logger.info(f"Grouped features into {len(non_empty_groups)} non-empty groups.")
        return non_empty_groups
    except Exception as e:
        logger.error(f"Error in group_features_by_correlation: {str(e)}")
        raise

def feature_groups_selection_process(df: pd.DataFrame, min_features_per_group: int) -> List[List[str]]:
    """
    Generate unique feature combinations based on correlation grouping.
    
    If no valid feature combinations are found, return the entire feature set as one group.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing features.
        min_features_per_group (int): Minimum number of features required in each group.

    Returns:
        List[List[str]]: List of unique feature combinations, where each combination is a sorted list of feature names.

    Raises:
        ValueError: If the input DataFrame is empty or if min_features_per_group is less than 2.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if min_features_per_group < 2:
        raise ValueError("Minimum features per group must be at least 2.")

    try:
        max_groups = len(df.columns) // 2 # Ensure there are no single-feature groups.
        unique_feature_combinations = []
        
        # Try generating groups with different numbers of clusters (from 2 up to max_groups)
        for n_groups in range(2, max_groups + 1):
            grouped_features = group_features_by_correlation(df, n_groups)
            valid_groups = [group for group in grouped_features if len(group) >= min_features_per_group]
            
            for group in valid_groups:
                sorted_group = sorted(group)
                if sorted_group not in unique_feature_combinations:
                    unique_feature_combinations.append(sorted_group)
        
        # If no valid groups were found, apply the fallback mechanism
        if not unique_feature_combinations:
            logger.warning(f"No valid feature combinations found. Fallback: returning all features as a single group.")
            
            # Fallback: Use the entire feature set as one group if it meets the min_features_per_group condition
            if len(df.columns) >= min_features_per_group:
                unique_feature_combinations = [df.columns.tolist()]
            else:
                logger.warning(f"Not enough features to meet the min_features_per_group requirement. No valid groups returned.")
        
        
        logger.info(f"Generated {len(unique_feature_combinations)} unique feature combinations.")
        return unique_feature_combinations
    except Exception as e:
        logger.error(f"Error in feature_groups_selection_process: {str(e)}")
        raise

if __name__ == "__main__":
    # Test case for feature_groups_selection_process
    np.random.seed(42)
    test_df = pd.DataFrame(np.random.rand(100, 6), columns=['A', 'B', 'C', 'D', 'E', 'F'])
    
    try:
        # Remove low variance features
        low_var_indices = remove_low_variance_features(test_df, variance_threshold=0.01)
        test_df_filtered = test_df.iloc[:, low_var_indices]
        
        print("Features after variance filtering:", test_df_filtered.columns.tolist())
        
        # Generate feature combinations
        feature_combinations = feature_groups_selection_process(test_df_filtered, min_features_per_group=2)
        
        print("\nGenerated feature combinations:")
        for combo in feature_combinations:
            print(combo)
        
    except Exception as e:
        logger.error(f"Error in test case: {str(e)}")