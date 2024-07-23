# src/task_workflows/tasks/find_optimal_window_size.py

from data_fetcher.fts_data_fetcher import get_fts_data
from task_workflows.task_registry import register_task
from data_processing.data_cleaner import clean_data_list
from typing import List, Dict, Any, Tuple
import pandas as pd
from data_processing.data_cleaner import clean_data_list
from data_processing.eda_process import perform_eda_regression
from data_processing.eda_process import calculate_feature_variance


def analyze_window_sizes(cleaned_data_list: List[Dict[str, Any]], regression_type: str) -> Dict[int, Dict[str, Any]]:
    results = {}
    for entry in cleaned_data_list:
        window_size = entry['window_size']
        df = entry['data']
        target_column = entry['params']['target_column']
        
        feature_variance = calculate_feature_variance(df, target_column)
        print(f"analyzing window size: {window_size}, regression type: {regression_type}, feature variance: {feature_variance}")        
        regression_results = perform_eda_regression(df, target_column, regression_type)
        
        results[window_size] = {
            'feature_variance': feature_variance,
            'regression_results': regression_results
        }
    
    return results

def select_optimal_window_size(results: Dict[int, Dict[str, float]], regression_type: str) -> int:
    metric = 'mean_r2' if regression_type == 'linear' else 'mean_roc_auc'
    return max(results, key=lambda x: results[x]['regression_results'][f'{regression_type}_regression'][metric] - results[x]['feature_variance'])

def optimize_window_size(cleaned_data_list: List[Dict[str, Any]], regression_type: str) -> Tuple[Dict, int, pd.DataFrame]:
    print(f"Optimizing window size for regression type: {regression_type}")
    results = analyze_window_sizes(cleaned_data_list, regression_type)
    optimal_window_size = select_optimal_window_size(results, regression_type)
    optimal_df = get_optimal_dataframe(cleaned_data_list, optimal_window_size)
    return results, optimal_window_size, optimal_df

def get_optimal_dataframe(cleaned_data_list: List[Dict[str, Any]], optimal_window_size: int) -> pd.DataFrame:
    return next(entry['data'] for entry in cleaned_data_list if entry['window_size'] == optimal_window_size)


def save_results(results: Dict[int, Dict[str, Any]], optimal_window_size: int, optimal_df: pd.DataFrame, data_params: dict, task_params: dict):
    """Save the results of the window size optimization process."""
    # This is a placeholder function. Implement according to your specific needs.
    print(f"Saving results for optimal window size: {optimal_window_size}")
    print(f"Results summary: {results[optimal_window_size]}")
    print(f"Data params: {data_params}")
    print(f"Task params: {task_params}")

def save_results_to_csv(results: Dict[int, Dict[str, Any]], optimal_window_size: int, data_params: dict, task_params: dict, output_file: str, regression_type: str):
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
    df['regression_type'] = regression_type
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

@register_task("find_optimal_window_size")
def find_optimal_window_size(play_type: str, data_params: dict, task_params: dict):
    if play_type == "fts":
        return find_optimal_window_size_fts(data_params, task_params)
    elif play_type == "box":
        return find_optimal_window_size_box(data_params, task_params)
    else:
        raise ValueError(f"Unsupported play type: {play_type}")


def find_optimal_window_size_fts(data_params, task_params):
    print("Handling FTS play type")
    print("Running find_optimal_window_size task")
    print("Data Params:", data_params)
    print("Task Params:", task_params)
    
    table_name = data_params['table_name']
    conditions = data_params['conditions']
    start_season = conditions.get('season_year_start', '201617')
    end_season = conditions.get('season_year_end', '202324')
    is_feat = conditions.get('is_feat', 'Yes')
    player_avg_type = conditions.get('player_avg_type', 'SMA')
    team_avg_type = conditions.get('team_avg_type', 'SMA')
    team_data_type = conditions.get('team_data_type', 'Overall')
    lg_avg_type = conditions.get('lg_avg_type', 'SMA')
    
    data_scope = task_params.get('data_scope', 'player')
    excluded_columns = task_params.get('exclude_columns', [])
    feat_columns = task_params.get('feat_columns', [])
    target_column = task_params.get('target', 'team1_tip')
    window_sizes = task_params.get('window_sizes', [-1])

    data_list = []

    for window_size in window_sizes:
        print(f"\nRetrieving data for window size: {window_size}")
        
        df = get_fts_data(
            start_season=start_season,
            end_season=end_season,
            is_feat=is_feat,
            player_avg_type=player_avg_type,
            player_window_size=window_size if data_scope == 'player' else -1,
            team_avg_type=team_avg_type,
            team_window_size=window_size if data_scope == 'team' else -1,
            team_data_type=team_data_type,
            lg_avg_type=lg_avg_type,
            lg_window_size=window_size if data_scope == 'lg' else -1,
            excluded_columns=excluded_columns,
            feat_columns=feat_columns,
            target_column=target_column
        )

        if df is not None and not df.empty:
            print(f"Successfully retrieved data for window size {window_size}.")
            print(f"Shape of the dataframe: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            
            data_entry = {
                'window_size': window_size,
                'data': df,
                'params': {
                    'start_season': start_season,
                    'end_season': end_season,
                    'is_feat': is_feat,
                    'player_avg_type': player_avg_type,
                    'team_avg_type': team_avg_type,
                    'team_data_type': team_data_type,
                    'lg_avg_type': lg_avg_type,
                    'data_scope': data_scope,
                    'excluded_columns': excluded_columns,
                    'feat_columns': feat_columns,
                    'target_column': target_column
                }
            }
            data_list.append(data_entry)
        else:
            print(f"Failed to retrieve data for window size {window_size} or the dataframe is empty.")

    regression_type = 'logistic'
    
    cleaned_data_list = clean_data_list(data_list)
    print('clean data finished')
    for entry in cleaned_data_list:
        cleaned_df = entry['data']
        window_size = entry['window_size']
        params = entry['params']
        print(params)
    
    # this is where the process does after clean data
    cleaned_data_list = clean_data_list(data_list)
    results, optimal_window_size, optimal_df = optimize_window_size(cleaned_data_list, regression_type)
    
    # save_results(results, optimal_window_size, optimal_df, data_params, task_params)
    output_file = f"window_size_results_{data_params['table_name']}.csv"
    save_results_to_csv(results, optimal_window_size, data_params, task_params, output_file, regression_type)
    
    return data_list
    

def find_optimal_window_size_box(data_params, task_params):
    print("Handling BOX play type")
    # Implement the BOX-specific logic here
    print("Running find_optimal_window_size task")
    print("Data Params:", data_params)
    print("Task Params:", task_params)

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