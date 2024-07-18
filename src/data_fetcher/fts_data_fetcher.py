# src/data_fetcher/fts_data_fetcher.py

import sys
import os
import pandas as pd

# Ensure the correct directory is included in the sys.path for the VSCode environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data_fetcher.database import execute_query, get_columns_for_table, generate_query_with_conditions

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

def fetch_fts_feature_data(start_season, end_season, is_feat, player_avg_type, 
                           player_window_size, team_window_size, team_avg_type, 
                           team_data_type, lg_window_size, lg_avg_type, 
                           excluded_columns, feat_columns=None):
    """
    Fetches feature data based on user input and processes the DataFrame.
    :param start_season: Start season year.
    :param end_season: End season year.
    :param is_feat: Feature flag.
    :param player_avg_type: Player average type.
    :param player_window_size: Player window size.
    :param team_window_size: Team window size.
    :param team_avg_type: Team average type.
    :param team_data_type: Team data type.
    :param lg_window_size: League window size.
    :param lg_avg_type: League average type.
    :param excluded_columns: Columns to exclude from having postfix added.
    :param feat_columns: Specific feature columns to fetch (optional).
    :return: Processed pandas DataFrame with unique column names, or None if an error occurs.
    """
    table_name = "dbo.fts_feature_table"
    columns = feat_columns if feat_columns else get_columns_for_table(table_name)
    
    conditions = {
        "season_year >= ": start_season,
        "season_year <= ": end_season,
        "is_feat = ": is_feat,
        "player_avg_type = ": player_avg_type,
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
    
    # postfix = f"{player_avg_type.lower()}_{player_window_postfix}_{team_window_postfix}_{lg_window_postfix}_{'feat' if is_feat.lower() == 'yes' else 'stats'}"
    index_columns = ["adv_idx_uuid", "season_year"]
    columns = index_columns + columns
    query, params = generate_query_with_conditions(table_name, columns, conditions)
    # print(f"Generated Query: {query}")
    # print(f"Query Parameters: {params}")
    
    df = execute_query(query, params)
    if df is not None:
        #df = rename_columns_with_postfix(df, postfix, excluded_columns)
        df = df.loc[:, columns]
        if df.empty:
            print("Feature DataFrame is empty.")
    else:
        print("Failed to fetch feature data.")
    return df

def fetch_fts_target_data(start_season, end_season, target_column=None):
    """
    Fetches target data based on user input.
    :param start_season: Start season year.
    :param end_season: End season year.
    :param target_column: Specific target column to fetch (optional).
    :return: pandas DataFrame, or None if an error occurs.
    """
    print("target column", target_column)
    if target_column:
        # Ensure target_column is treated as a single column name
        columns = f"idx.uuid AS adv_idx_uuid, fts.{target_column}"
    else:
        columns = "idx.uuid AS adv_idx_uuid, fts.team1_tip, fts.team1_fts"

    query = f"""
    SELECT {columns}
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
def combine_fts_feature_target_data(feature_df, target_df):
    """
    Combines feature data with target data for FTS analysis.

    Parameters:
    feature_df (pd.DataFrame): DataFrame containing feature data
    target_df (pd.DataFrame): DataFrame containing target data

    Returns:
    pd.DataFrame: Combined DataFrame with features and targets
    """
    # Ensure both DataFrames are not empty
    if feature_df.empty or target_df.empty:
        raise ValueError("One or both input DataFrames are empty")

    # Assuming 'adv_idx_uuid' is the common key between feature and target DataFrames
    if 'adv_idx_uuid' not in feature_df.columns or 'adv_idx_uuid' not in target_df.columns:
        raise KeyError("'adv_idx_uuid' column not found in one or both DataFrames")

    # Merge the feature and target DataFrames on 'adv_idx_uuid'
    combined_df = pd.merge(feature_df, target_df, on='adv_idx_uuid', how='inner')

    # Check if the merge resulted in any data
    if combined_df.empty:
        raise ValueError("Merging resulted in an empty DataFrame. Check if 'adv_idx_uuid' values match between DataFrames")

    # Optionally, you can drop duplicates if any
    combined_df = combined_df.drop_duplicates()

    # Optionally, reset the index of the combined DataFrame
    combined_df = combined_df.reset_index(drop=True)

    print(f"Combined DataFrame shape: {combined_df.shape}")
    print(f"Number of features: {len(feature_df.columns) - 1}")  # Subtract 1 for 'adv_idx_uuid'
    print(f"Number of targets: {len(target_df.columns) - 1}")  # Subtract 1 for 'adv_idx_uuid'

    return combined_df


def fetch_fts_feature_and_target(start_season="201314", end_season="202324", is_feat="yes", player_avg_type="sma", 
                                 player_window_size=-1, team_window_size=-1, team_avg_type="sma", 
                                 team_data_type="overall", lg_window_size=-1, lg_avg_type="sma", 
                                 excluded_columns=None, feat_columns=None, target_column=None):
    """
    Fetches feature and target data, and combines them.
    :param start_season: Start season year.
    :param end_season: End season year.
    :param is_feat: Feature flag.
    :param player_avg_type: Player average type.
    :param player_window_size: Player window size.
    :param team_window_size: Team window size.
    :param team_avg_type: Team average type.
    :param team_data_type: Team data type.
    :param lg_window_size: League window size.
    :param lg_avg_type: League average type.
    :param excluded_columns: Columns to exclude from having postfix added.
    :param feat_columns: Specific feature columns to fetch (optional).
    :param target_columns: Specific target columns to fetch (optional).
    :return: Combined DataFrame with features and target data, or None if an error occurs.
    """
    feature_df = fetch_fts_feature_data(start_season, end_season, is_feat, player_avg_type, 
                                        player_window_size, team_window_size, team_avg_type, 
                                        team_data_type, lg_window_size, lg_avg_type, 
                                        excluded_columns, feat_columns)

    target_df = fetch_fts_target_data(start_season, end_season, target_column)

    if feature_df is not None and target_df is not None:
        combined_df = combine_fts_feature_target_data(feature_df, target_df)
        combined_df = combined_df.sort_values(by="adv_idx_uuid").reset_index(drop=True)
       
        return combined_df

    return None

def get_fts_data(start_season="201314", end_season="202324", is_feat="yes", player_avg_type="sma", 
                 player_window_size=-1, team_window_size=-1, team_avg_type="sma", 
                 team_data_type="overall", lg_window_size=-1, lg_avg_type="sma", 
                 excluded_columns=None, feat_columns=None, target_column=None):
    """
    Entry point function for users to fetch and combine FTS feature and target data.
    :param start_season: Start season year.
    :param end_season: End season year.
    :param is_feat: Feature flag.
    :param player_avg_type: Player average type.
    :param player_window_size: Player window size.
    :param team_window_size: Team window size.
    :param team_avg_type: Team average type.
    :param team_data_type: Team data type.
    :param lg_window_size: League window size.
    :param lg_avg_type: League average type.
    :param excluded_columns: Columns to exclude from having postfix added.
    :param feat_columns: Specific feature columns to fetch (optional).
    :param target_columns: Specific target columns to fetch (optional).
    :return: Combined DataFrame with FTS features and target data, or None if an error occurs.
    """
    return fetch_fts_feature_and_target(start_season, end_season, is_feat, player_avg_type, 
                                        player_window_size, team_window_size, team_avg_type, 
                                        team_data_type, lg_window_size, lg_avg_type, 
                                        excluded_columns, feat_columns, target_column)
