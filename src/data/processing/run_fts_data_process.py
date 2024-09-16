# src/data_processing/run_fts_data_process.py

from eda_process import load_data

# Example usage
if __name__ == "__main__":
    # Parameters for fetching FTS data
    params = {
        'start_season': "201617",
        'end_season': "202324",
        'is_feat': "Yes",
        'avg_type': "SMA",
        'player_window_size': 30,
        'team_window_size': 30,
        'team_avg_type': "SMA",
        'team_data_type': "Overall",
        'lg_window_size': 30,
        'lg_avg_type': "SMA",
        'excluded_columns': ["adv_idx_uuid", "season_year", "match_date"]
    }

    # Load FTS data
    fts_data = load_data('fts', **params)
    
    if fts_data is not None:
        print("FTS Data loaded successfully:")
        print(fts_data.head())
    else:
        print("Failed to load FTS data.")
