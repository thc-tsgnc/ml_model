import sys
import os
import pandas as pd

# Ensure the correct directory is included in the sys.path for the VSCode environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.database import execute_query, get_columns_for_table, generate_query_with_conditions

def rename_columns_with_postfix(df, postfix, excluded_columns):
    """
    Renames the DataFrame columns to include user input variables in the column names, ensuring unique column names.
    :param df: The pandas DataFrame to process.
    :param postfix: The postfix to add to the column names.
    :param excluded_columns: Columns to exclude from having postfix added.
    :return: DataFrame with unique column names.
    """
    df.columns = [f"{col}_{postfix}" if col not in excluded_columns else col for col in df.columns]
    return df

def fetch_fts_feature_data(start_season, end_season, is_feat, avg_type, 
                           player_window_size, team_window_size, team_avg_type, 
                           team_data_type, lg_window_size, lg_avg_type, 
                           excluded_columns):
    """
    Fetches feature data based on user input and processes the DataFrame.
    :param start_season: Start season year.
    :param end_season: End season year.
    :param is_feat: Feature flag.
    :param avg_type: Average type.
    :param player_window_size: Player window size.
    :param team_window_size: Team window size.
    :param team_avg_type: Team average type.
    :param team_data_type: Team data type.
    :param lg_window_size: League window size.
    :param lg_avg_type: League average type.
    :param excluded_columns: Columns to exclude from having postfix added.
    :return: Processed pandas DataFrame with unique column names, or None if an error occurs.
    """
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

    query, params = generate_query_with_conditions(table_name, columns, conditions)
    print(f"Generated Query: {query}")
    print(f"Query Parameters: {params}")
    
    df = execute_query(query, params)
    if df is not None:
        df = rename_columns_with_postfix(df, postfix, excluded_columns)
        if df.empty:
            print("Feature DataFrame is empty.")
    else:
        print("Failed to fetch feature data.")
    return df

def fetch_fts_target_data(start_season, end_season):
    """
    Fetches target data based on user input.
    :param start_season: Start season year.
    :param end_season: End season year.
    :return: pandas DataFrame, or None if an error occurs.
    """
    query = """
    SELECT idx.uuid AS adv_idx_uuid, fts.team1_tip, fts.team1_fts
    FROM dbo.adv_idx AS idx
    INNER JOIN dbo.adv_fts AS fts ON idx.uuid = fts.adv_idx_uuid
    WHERE idx.season_year >= ? AND idx.season_year <= ? 
    """

    params = (start_season, end_season)
    print(f"Generated Query: {query}")
    print(f"Query Parameters: {params}")

    df = execute_query(query, params)
    if df is not None:
        if df.empty:
            print("Target DataFrame is empty.")
    else:
        print("Failed to fetch target data.")
    return df

def combine_fts_feature_target_data(feature_df, target_df, join_key="adv_idx_uuid"):
    """
    Combines feature and target dataframes on the specified join key.
    :param feature_df: DataFrame containing feature data.
    :param target_df: DataFrame containing target data.
    :param join_key: Key to join the DataFrames on.
    :return: Combined DataFrame.
    """
    combined_df = pd.merge(feature_df, target_df, on=join_key, how="inner")
    
    return combined_df

def fetch_fts_feature_and_target(start_season="201314", end_season="202324", is_feat="yes", avg_type="sma", 
                                 player_window_size=-1, team_window_size=-1, team_avg_type="sma", 
                                 team_data_type="overall", lg_window_size=-1, lg_avg_type="sma", 
                                 excluded_columns=None):
    """
    Fetches feature and target data, and combines them.
    :param start_season: Start season year.
    :param end_season: End season year.
    :param is_feat: Feature flag.
    :param avg_type: Average type.
    :param player_window_size: Player window size.
    :param team_window_size: Team window size.
    :param team_avg_type: Team average type.
    :param team_data_type: Team data type.
    :param lg_window_size: League window size.
    :param lg_avg_type: League average type.
    :param excluded_columns: Columns to exclude from having postfix added.
    :return: Combined DataFrame with features and target data, or None if an error occurs.
    """
    feature_df = fetch_fts_feature_data(start_season, end_season, is_feat, avg_type, 
                                        player_window_size, team_window_size, team_avg_type, 
                                        team_data_type, lg_window_size, lg_avg_type, excluded_columns)

    target_df = fetch_fts_target_data(start_season, end_season)

    if feature_df is not None and target_df is not None:
        combined_df = combine_fts_feature_target_data(feature_df, target_df)
        combined_df = combined_df.sort_values(by="adv_idx_uuid").reset_index(drop=True)
       
        return combined_df

    return None

def get_fts_data(start_season="201314", end_season="202324", is_feat="yes", avg_type="sma", 
                 player_window_size=-1, team_window_size=-1, team_avg_type="sma", 
                 team_data_type="overall", lg_window_size=-1, lg_avg_type="sma", 
                 excluded_columns=None):
    """
    Entry point function for users to fetch and combine FTS feature and target data.
    :param start_season: Start season year.
    :param end_season: End season year.
    :param is_feat: Feature flag.
    :param avg_type: Average type.
    :param player_window_size: Player window size.
    :param team_window_size: Team window size.
    :param team_avg_type: Team average type.
    :param team_data_type: Team data type.
    :param lg_window_size: League window size.
    :param lg_avg_type: League average type.
    :param excluded_columns: Columns to exclude from having postfix added.
    :return: Combined DataFrame with FTS features and target data, or None if an error occurs.
    """
    return fetch_fts_feature_and_target(start_season, end_season, is_feat, avg_type, 
                                        player_window_size, team_window_size, team_avg_type, 
                                        team_data_type, lg_window_size, lg_avg_type, 
                                        excluded_columns)
