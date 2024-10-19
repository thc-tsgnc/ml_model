# src/model_training/lazy_predict_modeling.py

from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

def perform_lazy_prediction(df: pd.DataFrame, target_column: str, task_type: str = 'classification', test_size: float = 0.2, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform lazy prediction on the given dataset using LazyPredict library.
    
    Args:
    df (pd.DataFrame): The input dataframe
    target_column (str): The name of the target column
    feature_set (List[str]): List of feature columns to use
    task_type (str): Either 'classification' or 'regression'
    test_size (float): The proportion of the dataset to include in the test split
    random_state (int): Random state for reproducibility
    
    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the models' performance metrics DataFrame and predictions DataFrame
    """
    
    # Select only the specified features and target column
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    print(f"Selected features: {X.columns.tolist()}")
    print(X.head())
    
    # Split the data
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y if task_type == 'classification' else None
)

    if task_type == 'classification':
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    elif task_type == 'regression':
        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")
    
    return models, predictions

def run_lazy_predict_modeling(data_list: List[Dict[str, Any]], configuration: dict, model_params: dict) -> Dict[str, Any]:
    """
    Main entry point for running the lazy predict modeling process.
    
    Args:
    data_list (List[Dict[str, Any]]): List of dictionaries containing data for each dataset
    configuration (dict): Dictionary containing task-specific configuration
    model_params (dict): Dictionary containing model-specific parameters
    
    Returns:
    Dict[str, Any]: Dictionary containing the results of the lazy predict modeling process
    """
    lazy_predict_results = {}
    
    for entry in data_list:
        df = entry['data']
        identifier = entry.get('window_size', 'unknown')
        target_column = configuration['target_column']
        task_type = configuration['model_type']

        
        print(f"Performing lazy prediction for dataset: {identifier}")
        
        models, predictions = perform_lazy_prediction(df, target_column, task_type)
        print("perform_lazy_prediction complete for dataset")
        lazy_predict_results[identifier] = {
            'models': models,
            'predictions': predictions
        }
        
        print(f"Completed lazy prediction for dataset: {identifier}")
    
    # Find optimal window size based on the best performing model across all datasets
    optimal_identifier = max(lazy_predict_results, key=lambda x: lazy_predict_results[x]['models'].iloc[0]['Accuracy'])
    
    results = {
        'lazy_predict_results': lazy_predict_results,
        'optimal_identifier': optimal_identifier,
    }
    
    return results

def run_lazy_predict_modeling_for_window_size(
    data: List[pd.DataFrame], 
    target_column: str, 
    model_type: str, 
    return_choices: List[str] = ["models"]
) -> List[Dict[str, pd.DataFrame]]:
    """
    Runs LazyPredict modeling process for the given window size task.

    Args:
    data (List[pd.DataFrame]): List of DataFrames to be used for modeling.
    target_column (str): The target column for prediction.
    model_type (str): Type of model to run ('classification' or 'regression').
    return_choices (List[str]): List of choices to determine which results to return. Options: "models", "predictions".

    Returns:
    List[Dict[str, pd.DataFrame]]: List of dictionaries containing the requested modeling results for each DataFrame.
    """
    lazy_predict_results = []

    for df in data:
        # Perform lazy prediction for the current DataFrame
        models, predictions = perform_lazy_prediction(df, target_column, model_type)

        # Initialize result container for this DataFrame
        result_entry = {}

        # Add results based on return choices
        if "models" in return_choices:
            result_entry['models'] = models
        if "predictions" in return_choices:
            result_entry['predictions'] = predictions

        lazy_predict_results.append(result_entry)

    return lazy_predict_results
