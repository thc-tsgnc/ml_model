# src/task_workflows/tasks/eda_clustering_analysis.py

import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import csv
import os
import itertools
from task_workflows.task_registry import register_task
from analysis.eda.clustering_analysis import run_clustering_analysis

@register_task("eda_clustering_analysis")
def eda_clustering_analysis(df: pd.DataFrame, 
                            features: List[str], 
                            algorithms: List[str],
                            param_grid: Dict[str, Dict[str, List[Any]]], 
                            output_dir: str,
                            true_labels: Optional[np.ndarray] = None) -> None:
    """
    Perform clustering analysis as part of the EDA process.
    """
    os.makedirs(output_dir, exist_ok=True)
    logs = []

    X = df[features]

    for algorithm in algorithms:
        for params in generate_param_combinations(param_grid[algorithm]):
            # Run the clustering analysis
            result = run_clustering_analysis(X, algorithm, params, true_labels)
            
            # Extract scalar and nested metrics separately
            evaluation_metrics = result['evaluation_metrics']
            distance_distribution = evaluation_metrics.pop('distance_distribution', None)

            # Prepare log entry for scalar metrics
            log_entry = {
                'algorithm': algorithm,
                'params': params,
                **flatten_dict(evaluation_metrics),  # Flatten scalar metrics
                'visualization_paths': ','.join(result['visualization_paths'])
            }

            # Save nested metrics like distance distribution separately
            if distance_distribution:
                detailed_metrics_path = os.path.join(output_dir, f"{algorithm}_detailed_metrics.json")
                with open(detailed_metrics_path, 'w') as f:
                    json.dump(distance_distribution, f, indent=4)
                log_entry['detailed_metrics_path'] = detailed_metrics_path

            # Add to logs
            logs.append(log_entry)
            log_clustering_run(log_entry, output_dir)

    # Generate summary report for scalar metrics only
    generate_summary_report(logs, output_dir)


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
    eda_clustering_analysis(df, features, algorithms, param_grid, output_dir)

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
