# src/task_workflows/tasks/clustering_workflow.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import csv
import os
from analysis.eda.clustering_analysis import run_clustering_analysis

def run_clustering_iterations(df: pd.DataFrame, algorithms: List[str], param_grid: Dict[str, List[Any]], output_dir: str, true_labels: Optional[np.ndarray] = None) -> None:
    os.makedirs(output_dir, exist_ok=True)
    logs = []

    for algorithm in algorithms:
        for params in generate_param_combinations(param_grid[algorithm]):
            result = run_clustering_analysis(df, algorithm, params, true_labels)
            log_entry = {
                'algorithm': algorithm,
                'params': params,
                **flatten_dict(result['evaluation_metrics']),
                'visualization_paths': ','.join(result['visualization_paths'])
            }
            logs.append(log_entry)
            log_clustering_run(log_entry, output_dir)

    generate_summary_report(logs, output_dir)

def generate_param_combinations(param_grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    import itertools
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
    
    # Reorder columns for better readability
    columns_order = ['algorithm'] + \
                    [col for col in df.columns if col.startswith('params_')] + \
                    [col for col in df.columns if col not in ['algorithm', 'visualization_paths'] and not col.startswith('params_')] + \
                    ['visualization_paths']
    
    df = df.reindex(columns=columns_order)
    
    report_path = os.path.join(output_dir, 'clustering_summary_report.csv')
    df.to_csv(report_path, index=False)
    
    return report_path

# Helper function to update parameter names in the log entry
def update_param_names(params: Dict[str, Any]) -> Dict[str, Any]:
    return {f'params_{k}': v for k, v in params.items()}