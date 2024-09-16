# src/task_workflows/runners/run_eda_task.py

import pandas as pd
import os
import sys
import json
import argparse
from typing import Dict, Any, List

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.insert(0, project_root)

from src.task_workflows.tasks.eda_feature_target_analysis import eda_feature_target_analysis
from src.task_workflows.tasks.eda_binning_analysis import eda_binning_analysis
from src.task_workflows.tasks.eda_statistical_analysis import eda_statistical_analysis
from src.task_workflows.tasks.eda_clustering_analysis import eda_clustering_analysis

def run_eda_task(df: pd.DataFrame, task: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run a specific EDA task.

    Args:
    df (pd.DataFrame): The input dataframe
    task (str): The name of the EDA task to run
    params (Dict[str, Any]): Parameters for the task

    Returns:
    Dict[str, Any]: Results of the EDA task
    """
    task_functions = {
        'feature_target': eda_feature_target_analysis,
        'binning': eda_binning_analysis,
        'statistical': eda_statistical_analysis,
        'clustering': eda_clustering_analysis
    }

    if task not in task_functions:
        raise ValueError(f"Unsupported task: {task}")

    return task_functions[task](df, **params)

def run_eda_process(df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the entire EDA process based on a configuration.

    Args:
    df (pd.DataFrame): The input dataframe
    config (Dict[str, Any]): Configuration for the EDA process

    Returns:
    Dict[str, Any]: Results of all EDA tasks
    """
    results = {}
    for task, params in config.items():
        if params.get('enabled', True):
            results[task] = run_eda_task(df, task, params)
    return results

def load_data(file_path: str) -> pd.DataFrame:
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from a JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run EDA tasks")
    parser.add_argument('--data', type=str, required=True, help='Path to CSV data file')
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    parser.add_argument('--task', type=str, choices=['feature_target', 'binning', 'statistical', 'clustering'], help='Specific EDA task to run')
    parser.add_argument('--params', type=str, help='JSON string of task parameters')
    return parser.parse_args()


