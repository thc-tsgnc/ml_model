# src/task_workflows/tasks/eda_clustering_analysis.py

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import csv
import os
import itertools
from task_workflows.task_registry import register_task
from analysis.eda.clustering_analysis import run_clustering_analysis
from datetime import datetime
from pathlib import Path
from data.processing.data_cleaner import clean_data_pipeline
from feature_engineering.feature_selector import feature_groups_selection_process
from analysis.eda.clustering_analysis import run_clustering_for_feature_combinations
from analysis.eda.statistical_analysis import cluster_statistical_analysis
import logging


# Set up basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Correct project root path
project_root = Path(__file__).resolve().parent.parent.parent.parent / 'results'
# Generate a timestamp for the current run
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Centralized output directory configuration
OUTPUT_DIRS = {
    'detailed': project_root / 'clustering_analysis' / 'detailed'/ timestamp,
    'feature_combination': project_root / 'clustering_analysis' / 'feature_combination'
}

# Create all output directories
for dir_path in OUTPUT_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# Default parameter grid for initial analysis
DEFAULT_PARAM_GRID = {
    'kmeans': {'n_clusters': (2, 10)},
    'dbscan': {'eps': (0.1, 1.0), 'min_samples': (2, 10)},
    'gmm': {'n_components': (2, 10)}
}

@register_task("eda_clustering_analysis")
def eda_clustering_analysis(
    df: pd.DataFrame, 
    features: List[str], 
    algorithms: List[str],
    param_grid: Optional[Dict[str, Dict[str, List[Any]]]] = None, 
    output_dir: str = 'clustering_analysis/detailed',
    target_column : Optional[str] = None,
    true_labels: Optional[np.ndarray] = None
) -> None:
    """
    Perform a detailed clustering analysis with user-defined parameters.
    """
    output_dir = OUTPUT_DIRS['detailed']
    
    if param_grid is None:
        param_grid = DEFAULT_PARAM_GRID
        
    if target_column is None:
        raise ValueError("Target column must be provided for statistical analysis.")
 
    logs = []
    
    df_cleaned = clean_and_prepare_data(df, features, target_column)

    X = df_cleaned[features]

    for algorithm in algorithms:
        for params in generate_param_combinations(param_grid[algorithm]):
            # Run the clustering analysis
            result = run_clustering_analysis(X, algorithm, params, true_labels, output_dir)
            
            
            # Extract scalar and nested metrics separately
            evaluation_metrics = result['evaluation_metrics']
            distance_distribution = evaluation_metrics.pop('distance_distribution', None)

            # Save nested metrics like distance distribution separately
            if distance_distribution:
                detailed_metrics_path = save_detailed_metrics(algorithm, distance_distribution, output_dir)

            
            cluster_labels = result['cluster_labels']
            if cluster_labels is None:
                raise ValueError(f"Clustering failed to return labels for algorithm: {algorithm}")
            
            df_cleaned['cluster_labels'] = cluster_labels

            cluster_stats = cluster_statistical_analysis(
                df=df_cleaned,
                cluster_column='cluster_labels',
                target_column=target_column
            )

            
            # Prepare log entry for scalar metrics
            log_entry = {
                'algorithm': algorithm,
                'params': params,
                **flatten_dict(evaluation_metrics),  # Flatten scalar metrics
                'visualization_paths': ','.join(result['visualization_paths']),
                'detailed_metrics_path': detailed_metrics_path,
                'cluster_statistical_analysis': cluster_stats 
            }

            # Add to logs
            logs.append(log_entry)
            save_clustering_log(log_entry, output_dir)

    # Generate summary report for scalar metrics only
    generate_summary_report(logs, output_dir)


@register_task("perform_initial_clustering_analysis")
def perform_initial_clustering_analysis(
    df: pd.DataFrame,
    feature_groups: Optional[Dict[str, List[str]]] = None,
    target_column: Optional[str] = None,
    min_features_per_group: int = 2,
    algorithms: List[str] = ['kmeans', 'dbscan', 'gmm'],
    metric_name: str = 'silhouette',
    n_trials: int = 10,
    timeout: Optional[int] = None,
    custom_param_spaces: Optional[Dict[str, Dict[str, Any]]] = None,
    output_dir: Optional[str] = None  # Optional argument for output_dir
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Perform initial quick clustering analysis with default or custom parameters.
    """  
    
    if feature_groups is None:
        features = df.columns.tolist()
        if target_column and target_column in features:
            features.remove(target_column)
        feature_groups = {'all_features': features}
    
    param_spaces = custom_param_spaces or DEFAULT_PARAM_GRID
    
    all_results = []
    summary = {
        "processed_groups": 0,
        "skipped_groups": 0,
        "error_groups": 0
    }

    for group_name, feature_list in feature_groups.items():
        print(f"Processing feature group: {group_name}")

        try:
            df_group = df[feature_list].copy()
            
            """
            scaling_params = {
                'columns_to_scale': feature_list,
                'exclude_columns': [target_column] if target_column else None
            }

            df_cleaned = clean_data_pipeline(
                df_group,
                target_column=None,
                remove_duplicates=True,
                missing_value_strategy='drop',
                normalize_columns=True,
                type_conversion_map=None,
                scaling_params=scaling_params,
                encoding_strategy='one_hot'
            )
            """
            df_cleaned = clean_and_prepare_data(df_group, feature_list, target_column)
            
            print(df_cleaned.columns.tolist())  
        
            feature_combinations = feature_groups_selection_process(
                df_cleaned,
                min_features_per_group=min_features_per_group
            )

            if not feature_combinations:
                logging.warning(f"No valid feature combinations found for group '{group_name}'. "
                                f"This may be due to insufficient features (min required: {min_features_per_group}) "
                                f"or high correlation between features.")
                summary["skipped_groups"] += 1
                continue

            results = run_clustering_for_feature_combinations(
                df=df_cleaned,
                feature_combinations=feature_combinations,
                algorithms=algorithms,
                metric_name=metric_name,
                n_trials=n_trials,
                timeout=timeout,
                custom_param_spaces=custom_param_spaces
            )

            for result in results:
                result['group_name'] = group_name

            all_results.extend(results)
            summary["processed_groups"] += 1
            
        except Exception as e:
            logging.error(f"Error processing group '{group_name}': {str(e)}", exc_info=True)
            
    if all_results:
        print(f"All results: {all_results}")  # Debugging: Check if Group 1 results are present
        save_feature_combination_clustering_summary(all_results)
    else:
        logging.warning("No results to save. All groups were skipped or encountered errors.")

    return all_results, summary

def clean_and_prepare_data(df, features, target_column):
    scaling_params = {
        'columns_to_scale': features,
        'exclude_columns': [target_column] if target_column else None
    }
    df_cleaned = clean_data_pipeline(
        df,
        target_column=None,
        remove_duplicates=True,
        missing_value_strategy='drop',
        normalize_columns=True,
        scaling_params=scaling_params,
        encoding_strategy='one_hot'
    )
    return df_cleaned

def save_clustering_results(log_entry, output_dir, log_file_name='clustering_log.csv'):
    """
    Save clustering log entries in CSV format.

    Args:
        log_entry: Dictionary with clustering result metrics.
        output_dir: Path to save the log.
        log_file_name: The name of the log file (default: 'clustering_log.csv').
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_file = output_dir / log_file_name
    file_exists = csv_file.exists()

    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)


def save_feature_combination_clustering_summary(results: List[Dict[str, Any]], output_dir: Optional[Path] = None) -> str:
    """
    Save the summary of clustering results for various feature combinations.

    Args:
        results (List[Dict[str, Any]]): The results from run_clustering_for_feature_combinations.

    Returns:
        str: The path to the saved summary file.
    """
    
    # Use the default output_dir if none is provided
    if output_dir is None:
        output_dir = OUTPUT_DIRS['feature_combination']
    
    # Ensure the output directory exists before saving
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_file = output_dir / f'feature_combination_clustering_summary_{timestamp}.json'

    summary_data = []

    for result in results:
        # Handle errors in results
        if 'error' in result:
            summary_data.append({
                'features': result.get('features', []),
                'algorithm': result.get('algorithm', ''),
                'error': result['error']
            })
            continue

        group_name = result.get('group_name', '')
        features = result.get('features', [])
        algorithm = result.get('algorithm', '')
        metrics = result.get('metrics', {})
        best_params = result.get('best_params', {})
        
        # Handle the number of clusters based on the algorithm type
        if algorithm in ['kmeans', 'gmm']:
            optimal_clusters = best_params.get('n_clusters' if algorithm == 'kmeans' else 'n_components')
        elif algorithm == 'dbscan':
            labels = result.get('labels', [])
            optimal_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        else:
            print(f"Warning: Unsupported algorithm {algorithm}. Skipping entry.")
            continue

        summary_entry = {
            'group_name': group_name,
            'features': features,
            'algorithm': algorithm,
            'optimal_number_of_clusters': optimal_clusters,
            'best_params': best_params,
            'metrics': metrics
        }

        summary_data.append(summary_entry)

    # Save the summary data to a JSON file
    # summary_file = full_output_dir / f'feature_combination_clustering_summary_{timestamp}.json'
    try:
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=4)
        print(f"Feature combination clustering summary saved to {summary_file}")
    except IOError as e:
        print(f"Failed to save summary file: {str(e)}")

    return str(summary_file)

def save_clustering_log(log_entry: Dict[str, Any], output_dir: Path) -> None:
    """
    Save the log entry for clustering analysis in CSV format.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_file = output_dir / 'clustering_log.csv'
    file_exists = csv_file.exists()

    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)


def save_detailed_metrics(algorithm: str, distance_distribution: Any, output_dir: Path) -> str:
    """
    Save the detailed metrics for a given algorithm's clustering result.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save the detailed metrics as a JSON file
    detailed_metrics_path = output_dir / f"{algorithm}_detailed_metrics.json"
    with open(detailed_metrics_path, 'w') as f:
        json.dump(distance_distribution, f, indent=4)

    return str(detailed_metrics_path)


def generate_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = param_grid.keys()
    values = param_grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def log_clustering_run(log_entry: Dict[str, Any], output_dir: str) -> None:
    csv_file = os.path.join(output_dir, 'clustering_log.csv')
    file_exists = os.path.exists(csv_file)
    
    with open(csv_file, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

def generate_summary_report(logs: List[Dict[str, Any]], output_dir: str) -> str:
    df = pd.DataFrame(logs)
    
    columns_order = ['algorithm'] + \
                    [col for col in df.columns if col.startswith('params_')] + \
                    [col for col in df.columns if col not in ['algorithm', 'visualization_paths'] and not col.startswith('params_')] + \
                    ['visualization_paths']
    
    df = df.reindex(columns=columns_order)
    
    report_path = os.path.join(output_dir, 'clustering_summary_report.csv')
    df.to_csv(report_path, index=False)
    
    return report_path

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    
    # Create a sample DataFrame with numeric features
    df = pd.DataFrame({
        'feature1': np.random.rand(100),  # Random values for feature 1
        'feature2': np.random.rand(100),  # Random values for feature 2
        'target': np.random.randint(0, 3, 100)  # Adding a target column with 3 clusters
    })

    # Define the list of features to use for clustering
    features = ['feature1', 'feature2']

    # Define algorithms to be tested
    algorithms = ['kmeans']

    # Define a parameter grid for KMeans clustering
    param_grid = {
        'kmeans': {
            'n_clusters': [2, 3, 4],
            'random_state': [42]
        }
    }

    # Specify the output directory
    output_dir = 'results/eda_clustering'

    # Run the EDA clustering analysis task
    # eda_clustering_analysis(df, features, algorithms, param_grid, output_dir)
    eda_clustering_analysis(df, features, algorithms, param_grid, output_dir, target_column='target')

    # Print a message indicating completion
    print(f"EDA Clustering Analysis completed. Results are saved in '{output_dir}'.")

    # Load and print results from saved files for detailed output
    for algorithm in algorithms:
        for n_clusters in param_grid[algorithm]['n_clusters']:
            # Construct the detailed metrics file path
            detailed_metrics_path = os.path.join(output_dir, f"{algorithm}_detailed_metrics.json")
            
            # Load and print detailed metrics if available
            if os.path.exists(detailed_metrics_path):
                print(f"\nDetailed Metrics for {algorithm} with n_clusters = {n_clusters}:")
                with open(detailed_metrics_path, 'r') as f:
                    detailed_metrics = json.load(f)
                    for cluster, metrics in detailed_metrics.items():
                        print(f"{cluster}: {metrics}")
            
            # Load and print the clustering log CSV file
            clustering_log_path = os.path.join(output_dir, 'clustering_log.csv')
            if os.path.exists(clustering_log_path):
                print(f"\nClustering Log for {algorithm} with n_clusters = {n_clusters}:")
                clustering_log = pd.read_csv(clustering_log_path)
                print(clustering_log)

    # Load and print the summary report
    summary_report_path = os.path.join(output_dir, 'clustering_summary_report.csv')
    if os.path.exists(summary_report_path):
        print("\nSummary Report:")
        summary_report = pd.read_csv(summary_report_path)
        print(summary_report)
        
    print("\n--- Running perform_initial_clustering_analysis ---")
    
    # Create a larger sample DataFrame with more features
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        'feature1': np.random.rand(200),
        'feature2': np.random.rand(200),
        'feature3': np.random.rand(200),
        'feature4': np.random.rand(200),
        'feature5': np.random.rand(200),
        'target': np.random.choice(['A', 'B', 'C'], 200)
    })
    
    # Define feature groups
    feature_groups = {
        'group1': ['feature1', 'feature2', 'feature3'],
        'group2': ['feature3', 'feature4', 'feature5']
    }
    
    # Set up parameters
    params = {
        'df': df,
        'feature_groups': feature_groups,
        'target_column': 'target',
        'min_features_per_group': 2,
        'algorithms': ['kmeans', 'dbscan', 'gmm'],
        'metric_name': 'silhouette',
        'n_trials': 5,  # Reduced for quicker execution
        'timeout': 30,  # 30 seconds timeout
        'custom_param_spaces': {
            'kmeans': {'n_clusters': [2, 3, 4]},
            'dbscan': {'eps': [0.3, 0.5, 0.7], 'min_samples': [3, 5]}
        }
    }
    
    # Run the function
    all_results, summary = perform_initial_clustering_analysis(**params)

    # Print summary
    print("\nClustering Analysis Summary:")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    # Print a sample of results
    print("\nSample of Clustering Results:")
    for result in all_results[:2]:  # Print first two results
        print(f"\nGroup: {result['group_name']}")
        print(f"Algorithm: {result['algorithm']}")
        print(f"Features: {result['features']}")

        # Check if 'best_params' exists before trying to print it
        if 'best_params' in result:
            print(f"Best Parameters: {result['best_params']}")
        else:
            print("Best Parameters: Not available for this algorithm or invalid result.")
        
        print(f"Metrics: {result.get('metrics', 'No metrics available')}")