# src/task_workflows/tasks/clustering_feature_creation_task.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.pipeline import Pipeline
from task_workflows.task_registry import register_task
from pipelines.clustering_pipeline import create_clustering_pipeline
from utils.model_saver import save_model_and_pipeline
from feature_store.feature_store_manager import connect_to_feature_store, write_features_to_store
import logging

logger = logging.getLogger(__name__)

@register_task("clustering_feature_creation")
def clustering_feature_creation(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create cluster-based features using the specified configuration.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_groups (Dict[str, List[str]]): Dictionary of feature groups for clustering.
        config (Dict[str, Any]): Configuration for the entire process.

    Returns:
        Dict[str, Any]: Results including cluster pipelines, labels, and paths to saved artifacts.
    """
    # Step 1: Input Validation
    validate_task_input(df, feature_groups, config)

    # Step 2: Preprocessing
    log_missing_data(df, feature_groups)

    # Step 3: Clustering Process
    clustering_results = perform_clustering(df, feature_groups, config['clustering_config'])

    # Step 4: Save Models and Pipelines (if configured)
    if config.get('save_config'):
        save_results = save_clustering_artifacts(clustering_results, config['save_config'])
        clustering_results['save_results'] = save_results

    # Step 5: Save to Feature Store (if configured)
    if config.get('feature_store_config'):
        feature_store_results = save_to_feature_store(df, clustering_results, config['feature_store_config'])
        clustering_results['feature_store_results'] = feature_store_results

    return clustering_results

def validate_task_input(df: pd.DataFrame, feature_groups: Dict[str, List[str]], config: Dict[str, Any]):
    """Validate input parameters for the task."""
    # Validate feature groups
    for group, features in feature_groups.items():
        if not features:
            raise ValueError(f"Feature group '{group}' is empty.")
        if not all(feature in df.columns for feature in features):
            raise ValueError(f"Some features in group '{group}' are not present in the DataFrame.")

    # Validate config completeness
    if 'clustering_config' not in config:
        raise ValueError("Missing required 'clustering_config' in configuration.")

    # Validate feature types for clustering
    for group, features in feature_groups.items():
        numeric_features = [f for f in features if np.issubdtype(df[f].dtype, np.number)]
        if not numeric_features:
            raise ValueError(f"Group '{group}' does not contain any numeric features suitable for clustering.")

def log_missing_data(df: pd.DataFrame, feature_groups: Dict[str, List[str]]):
    """Log missing data for each feature group."""
    for group, features in feature_groups.items():
        missing_values = df[features].isnull().sum()
        if missing_values.any():
            logger.warning(f"Missing values detected in group '{group}': {missing_values[missing_values > 0]}")

def perform_clustering(
    df: pd.DataFrame,
    feature_groups: Dict[str, List[str]],
    clustering_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Perform clustering on each feature group."""
    clustering_pipelines = {}
    cluster_labels = {}
    
    for group_name, features in feature_groups.items():
        try:
            numeric_features = [f for f in features if np.issubdtype(df[f].dtype, np.number)]
            pipeline = create_clustering_pipeline(
                numeric_features=numeric_features,
                categorical_features=[],  # Exclude non-numeric features
                **clustering_config
            )
            clustering_pipelines[group_name] = pipeline
            
            # Fit and transform the data
            transformed_data = pipeline.fit_transform(df[numeric_features])
            cluster_labels[group_name] = transformed_data['cluster_label']
        except Exception as e:
            logger.error(f"Error in clustering for group '{group_name}': {str(e)}")
            raise

    return {
        'individual_pipelines': clustering_pipelines,
        'cluster_labels': cluster_labels
    }

def save_clustering_artifacts(
    clustering_results: Dict[str, Any],
    save_config: Dict[str, Any]
) -> Dict[str, str]:
    """Save clustering models and pipelines."""
    save_paths = {}
    
    for group_name, pipeline in clustering_results['individual_pipelines'].items():
        try:
            path = save_model_and_pipeline(
                pipeline=pipeline,
                name=f"{save_config['prefix']}_{group_name}_pipeline",
                directory=save_config['directory']
            )
            save_paths[f"{group_name}_pipeline_path"] = path
        except Exception as e:
            logger.error(f"Error saving pipeline for group '{group_name}': {str(e)}")
    
    return save_paths

def save_to_feature_store(
    df: pd.DataFrame,
    clustering_results: Dict[str, Any],
    feature_store_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Save cluster labels to feature store."""
    try:
        # Connect to the feature store
        connect_to_feature_store()
        
        # Prepare the features DataFrame
        features_df = df.copy()
        for group_name, labels in clustering_results['cluster_labels'].items():
            feature_name = f"cluster_{group_name}"
            features_df[feature_name] = labels
        
        # Get the primary key from the config
        primary_key = feature_store_config.get('primary_key', 'id')
        if primary_key not in features_df.columns:
            raise ValueError(f"Primary key '{primary_key}' not found in the DataFrame")
        
        # Write features to the feature store
        write_features_to_store(
            features_df=features_df,
            feature_table_name=feature_store_config['feature_table_name'],
            primary_key=primary_key,
            mode=feature_store_config.get('write_mode', 'merge')
        )
        
        return {"feature_store_update": "success"}
    except Exception as e:
        logger.error(f"Error saving to feature store: {str(e)}")
        return {"feature_store_update": "failed", "error": str(e)}