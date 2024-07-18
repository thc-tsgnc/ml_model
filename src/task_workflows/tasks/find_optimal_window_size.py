# src/task_workflows/tasks/find_optimal_window_size.py

from data_fetcher.fts_data_fetcher import get_fts_data
from task_workflows.task_registry import register_task

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

    """
    # Extract parameters
    start_season = data_params.get('start_season', '201617')
    end_season = data_params.get('end_season', '202324')
    is_feat = data_params.get('is_feat', 'Yes')
    player_avg_type = data_params.get('player_avg_type', 'SMA')
    team_avg_type = data_params.get('team_avg_type', 'SMA')
    team_data_type = data_params.get('team_data_type', 'Overall')
    lg_avg_type = data_params.get('lg_avg_type', 'SMA')

    data_scope = task_params.get('data_scope', 'player')
    excluded_columns = task_params.get('exclude_columns', [])
    feat_columns = task_params.get('feat_columns', [])
    target_column = task_params.get('target', 'team1_tip')
    window_sizes = task_params.get('window_sizes', [-1])
    """
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