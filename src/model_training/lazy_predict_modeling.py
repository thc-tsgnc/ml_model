# src/model_training/lazy_predict_modeling.py

from lazypredict.Supervised import LazyRegressor, LazyClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
import json
from datetime import datetime

def perform_lazy_prediction(df: pd.DataFrame, target_column: str, task_type: str = 'classification', test_size: float = 0.2, random_state: int = 42):
    """
    Perform lazy prediction on the given dataset using LazyPredict library.
    
    Args:
    df (pd.DataFrame): The input dataframe
    target_column (str): The name of the target column
    task_type (str): Either 'classification' or 'regression'
    test_size (float): The proportion of the dataset to include in the test split
    random_state (int): Random state for reproducibility
    
    Returns:
    Tuple[pd.DataFrame, Dict]: A tuple containing the models' performance metrics DataFrame and predictions dictionary
    """
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    if task_type == 'classification':
        clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = clf.fit(X_train, X_test, y_train, y_test)
    elif task_type == 'regression':
        reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
        models, predictions = reg.fit(X_train, X_test, y_train, y_test)
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")
    
    return models, predictions

def get_best_model(models: pd.DataFrame, metric: str = 'Accuracy'):
    """
    Get the best performing model based on the specified metric.
    
    Args:
    models (pd.DataFrame): The DataFrame of model results from perform_lazy_prediction
    metric (str): The metric to use for comparison (default is 'Accuracy' for classification)
    
    Returns:
    tuple: (best_model_name, best_score)
    """
    best_model = models.sort_values(by=metric, ascending=False).iloc[0]
    return best_model.name, best_model[metric]

def get_average_performance(models: pd.DataFrame, metric: str = 'Accuracy', top_n: int = None):
    """
    Calculate the average performance across all models or top N models based on the specified metric.
    
    Args:
    models (pd.DataFrame): The DataFrame of model results from perform_lazy_prediction
    metric (str): The metric to use for calculation (default is 'Accuracy' for classification)
    top_n (int, optional): The number of top models to consider. If None, all models are considered.
    
    Returns:
    float: The average score across all models or top N models
    """
    sorted_models = models.sort_values(by=metric, ascending=False)
    
    if top_n is not None:
        sorted_models = sorted_models.head(top_n)
    
    return sorted_models[metric].mean()


def run_lazy_predict(data_list: List[Dict[str, Any]], task_type: str = 'classification', metric: str = 'Accuracy', top_n: int = None) -> Dict[Any, Dict[str, Any]]:
    """
    Run lazy prediction for multiple datasets and return the results.
    
    Args:
    data_list (List[Dict[str, Any]]): List of dictionaries containing data for each dataset
    task_type (str): Either 'classification' or 'regression'
    metric (str): The metric to use for comparing models
    top_n (int, optional): The number of top models to consider for average calculation. If None, all models are considered.
    
    Returns:
    Dict[Any, Dict[str, Any]]: Dictionary with dataset identifiers as keys and results as values
    """
    lazy_predict_results = {}
    
    for entry in data_list:
        df = entry['data']
        identifier = entry.get('identifier', entry.get('window_size', 'unknown'))
        target_column = entry['params']['target_column']
        
        print(f"Performing lazy prediction for dataset: {identifier}")
        
        models, predictions = perform_lazy_prediction(df, target_column, task_type)
        avg_score = get_average_performance(models, metric, top_n)
        best_model, best_score = get_best_model(models, metric)
        
        lazy_predict_results[identifier] = {
            'avg_score': avg_score,
            'best_model': best_model,
            'best_score': best_score,
            'all_models': models.to_dict(),
            'predictions': predictions
        }
        
        print(f"Best model for dataset {identifier}: {best_model} with {metric}: {best_score}")  # Highlighted
    
    
    return lazy_predict_results


def find_optimal_dataset(lazy_predict_results: Dict[Any, Dict[str, Any]]) -> Tuple[Any, str, float]:
    """
    Find the optimal dataset based on the lazy predict results.
    
    Args:
    lazy_predict_results (Dict[Any, Dict[str, Any]]): Results from run_lazy_predict
    
    Returns:
    Tuple[Any, str, float]: Optimal dataset identifier, best model name, and best score
    """
    optimal_identifier = max(lazy_predict_results, key=lambda x: lazy_predict_results[x]['best_score'])
    best_model = lazy_predict_results[optimal_identifier]['best_model']
    best_score = lazy_predict_results[optimal_identifier]['best_score']
    
    return optimal_identifier, best_model, best_score

def save_results(results: Dict[str, Any], filename: str):
    """
    Save the results to a JSON file.
    
    Args:
    results (Dict[str, Any]): The results to save
    filename (str): The name of the file to save the results to
    """
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4, default=str)

def run_lazy_predict_modeling(data_list: List[Dict[str, Any]], task_params: Dict[str, Any], model_params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for running the lazy predict modeling process.
    
    Args:
    data_list (List[Dict[str, Any]]): List of dictionaries containing data for each dataset
    task_params (Dict[str, Any]): Dictionary containing task-specific parameters
    model_params (Dict[str, Any]): Dictionary containing model-specific parameters
    
    Returns:
    Dict[str, Any]: Dictionary containing the results of the lazy predict modeling process
    """
    # Extract necessary parameters
    task_type = model_params.get('task_type', 'classification')
    metric = model_params.get('metric', 'Accuracy')
    top_n = model_params.get('top_n', None)

    lazy_predict_results = run_lazy_predict(data_list, task_type, metric, top_n)
    optimal_identifier = max(lazy_predict_results, key=lambda x: lazy_predict_results[x]['avg_score'])
    
    results = {
        'task_params': task_params,
        'model_params': model_params,
        'lazy_predict_results': lazy_predict_results,
        'optimal_identifier': optimal_identifier,
        'optimal_avg_score': lazy_predict_results[optimal_identifier]['avg_score'],
        'best_model': lazy_predict_results[optimal_identifier]['best_model'],
        'best_score': lazy_predict_results[optimal_identifier]['best_score'],
        'timestamp': datetime.now().isoformat()
    }
    
    # Save results to a JSON file
    task_name = task_params.get('task_name', 'lazy_predict')
    filename = f"{task_name}_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    save_results(results, filename)
    
    return results
