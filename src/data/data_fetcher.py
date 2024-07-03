import sys
import os

# Ensure the correct directory is included in the sys.path for VSCode environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.database import execute_query, get_columns_for_table, generate_query_with_conditions

def process_dataframe(df, postfix, excluded_columns):
    df.columns = [f"{col}_{postfix}" if col not in excluded_columns else col for col in df.columns]
    return df

def fetch_fts_data(start_season="201617", end_season="202324", is_feat="yes", avg_type="sma", 
                   player_window_size=-1, team_window_size=-1, team_avg_type="sma", 
                   team_data_type="overall", lg_window_size=-1, lg_avg_type="sma", 
                   excluded_columns=None):
    """
    Fetches FTS data based on user input and processes the DataFrame.
    """
    if excluded_columns is None:
        excluded_columns = ["adv_idx_uuid", "season_year", "match_date"]

    table_name = "dbo.fts_feature_table"
    columns = get_columns_for_table(table_name)
    conditions = {
        "season_year >= ": start_season,
        "season_year <= ": end_season,
        "is_feat = ": is_feat,
        "player_avg_type = ": avg_type,
        "player_window_size = ": player_window_size,
        "team_window_size = ": team_window_size,
        "team_avg_type = ": team_avg_type,
        "team_data_type = ": team_data_type,
        "lg_window_size = ": lg_window_size,
        "lg_avg_type = ": lg_avg_type
    }

    player_window_postfix = "cr" if player_window_size == -1 else str(player_window_size)
    team_window_postfix = "sn" if team_window_size == -1 else str(team_window_size)
    lg_window_postfix = "sn" if lg_window_size == -1 else str(lg_window_size)
    
    postfix = f"{avg_type.lower()}_{player_window_postfix}_{team_window_postfix}_{lg_window_postfix}_{'feat' if is_feat.lower() == 'yes' else 'stats'}"

    query, params = generate_query_with_conditions(table_name, columns, conditions)  # Ensure unpacking tuple
    print(f"Generated Query: {query}")  # Debug print
    print(f"Query Parameters: {params}")  # Debug print
    
    df = execute_query(query, params)  # Pass query and params separately
    if df is not None:
        df = process_dataframe(df, postfix, excluded_columns)
    return df
