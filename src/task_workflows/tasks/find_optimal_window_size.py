# src/task_workflows/tasks/find_optimal_window_size.py

from data_fetcher.fts_data_fetcher import get_fts_data
from task_workflows.task_registry import register_task
from data_processing.data_cleaner import clean_data_list
from typing import List, Dict, Any, Tuple
import pandas as pd
from data_processing.eda_process import perform_eda_regression
from data_processing.eda_process import calculate_feature_variance
from model_training.lazy_predict_modeling import run_lazy_predict_modeling
from data_processing.eda_process import perform_eda_regression, calculate_feature_variance

def analyze_window_sizes(cleaned_data_list, model_type):
    results = {}
    for entry in cleaned_data_list:
        window_size = entry['window_size']
        df = entry['data']
        target_column = entry['params']['target_column']
        
        feature_variance = calculate_feature_variance(df, target_column)
        regression_results = perform_eda_regression(df, target_column, model_type)
        
        results[window_size] = {
            'feature_variance': feature_variance,
            'regression_results': regression_results
        }
    return results

def select_optimal_window_size(results, model_type):
    if isinstance(results, dict) and all(isinstance(v, dict) for v in results.values()):
        # Standard approach results
        metric = 'mean_r2' if model_type == 'regression' else 'mean_roc_auc'
        return max(results, key=lambda x: results[x]['regression_results'][f'{model_type}_regression'][metric] - results[x]['feature_variance'])
    else:
        # LazyPredict results
        return results['optimal_identifier']

def optimize_window_size(cleaned_data_list: List[Dict[str, Any]], regression_type: str) -> Tuple[Dict, int, pd.DataFrame]:
    print(f"Optimizing window size for regression type: {regression_type}")
    results = analyze_window_sizes(cleaned_data_list, regression_type)
    optimal_window_size = select_optimal_window_size(results, regression_type)
    optimal_df = get_optimal_dataframe(cleaned_data_list, optimal_window_size)
    return results, optimal_window_size, optimal_df

def get_optimal_dataframe(cleaned_data_list: List[Dict[str, Any]], optimal_window_size: int) -> pd.DataFrame:
    return next(entry['data'] for entry in cleaned_data_list if entry['window_size'] == optimal_window_size)


def save_results_to_csv(results: Dict[int, Dict[str, Any]], optimal_window_size: int, data_params: dict, task_params: dict, model_params: dict, output_file: str, regression_type: str):
    """Save the results of the window size optimization process to a CSV file."""
    print(f"Saving results to CSV file: {output_file}")
    rows = []
    for window_size, result in results.items():
        row = {
            'window_size': window_size,
            'feature_variance': result['feature_variance'],
            'regression_metric': result['regression_results'][f"{regression_type}_regression"]['mean_roc_auc' if regression_type == 'logistic' else 'mean_r2'],
            'is_optimal': window_size == optimal_window_size
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['data_params'] = str(data_params)
    df['task_params'] = str(task_params)
    df['model_params'] = str(model_params)
    df['regression_type'] = regression_type
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

@register_task("find_optimal_window_size")
def find_optimal_window_size(play_type: str, data_params: dict, model_params: dict, feature_params: list, pipeline_params: list, configuration: dict, query_params: dict, process_params: dict, output_params: dict):
    if play_type == "fts":
        return find_optimal_window_size_fts(data_params, model_params, feature_params, pipeline_params, configuration, query_params, process_params, output_params)
    elif play_type == "hdp":
        return find_optimal_window_size_hdp(data_params, model_params, feature_params, pipeline_params, configuration, query_params, process_params, output_params)
    else:
        raise ValueError(f"Unsupported play type: {play_type}")

def find_optimal_window_size_fts(data_params: dict, model_params: dict, feature_params: list, pipeline_params: list, configuration: dict, query_params: dict, process_params: dict, output_params: dict):
    print("Handling FTS play type")
    print("Running find_optimal_window_size task")
    print("Configuration:", configuration)
    
    # Extract necessary parameters
    table_name = data_params['table_name']
    conditions = data_params['conditions']
    target_column = configuration['target_column']
    window_sizes = query_params.get('window_sizes', [-1, 30])
    
    data_list = []
    for window_size in window_sizes:
        print(f"\nRetrieving data for window size: {window_size}")
        
        df = get_fts_data(
            start_season=conditions['season_year_start'],
            end_season=conditions['season_year_end'],
            is_feat=conditions['is_feat'],
            player_avg_type=conditions['player_avg_type'],
            player_window_size=window_size,
            team_avg_type=conditions['team_avg_type'],
            team_window_size=window_size,
            team_data_type=conditions['team_data_type'],
            lg_avg_type=conditions['lg_avg_type'],
            lg_window_size=window_size,
            excluded_columns=[],
            feat_columns=feature_params,
            target_column=target_column
        )

        if df is not None and not df.empty:
            print(f"Successfully retrieved data for window size {window_size}.")
            data_entry = {
                'window_size': window_size,
                'data': df,
                'params': {
                    'target_column': target_column,
                    'window_size': window_size,
                }
            }
            data_list.append(data_entry)
        else:
            print(f"Failed to retrieve data for window size {window_size} or the dataframe is empty.")

    # Clean the data
    cleaned_data_list = clean_data_list(data_list)

    # Determine which modeling approach to use based on configuration
    model_type = configuration['model_type']
    model_key = configuration['model_key']

    if model_type == 'classification' and model_key == 'lazy_predict':
        print("Using LazyPredict for classification")
        results = run_lazy_predict_modeling(cleaned_data_list, configuration, model_params)
    else:
        print(f"Using standard approach for {model_type}")
        results = analyze_window_sizes(cleaned_data_list, model_type)

    # Select optimal window size
    optimal_window_size = select_optimal_window_size(results, model_type)

    # Save results
    save_results(results, optimal_window_size, output_params)

    return results, optimal_window_size    

def find_optimal_window_size_hdp(data_params: dict, task_params: dict, model_params: dict):
    print("Handling BOX play type")
    print("Running find_optimal_window_size task")
    print("Data Params:", data_params)
    print("Task Params:", task_params)
    print("Model Params:", model_params)

def run_process_for_fts_window_size(datalist):
    results = []
    """
    for data in datalist:
    
       
        cleaned_data = clean_data(data)
        cv_results = cross_validate(cleaned_data)
        lr_results = run_logistic_regression(cleaned_data)
        stats_results = run_stats_test(cleaned_data)
        other_results = run_other_experiments(cleaned_data)
        
        results.append({
            'window_size': data['window_size'],
            'cv_results': cv_results,
            'lr_results': lr_results,
            'stats_results': stats_results,
            'other_results': other_results,
            'params': data['params']
        })
        """

def save_results(results, optimal_window_size, output_params):
    # Implement saving logic based on the structure of results and output_params
    pass


