# src/data_fetcher/database.py

import sys
import os

# Add the parent directory of src to the sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pyodbc
import pandas as pd
import time
from data.config import DATABASE_CONFIG

def get_connection():
    """
    Establishes and returns a connection to the database.
    Retries the connection up to three times if it fails initially.
    :return: Connection object or None if connection fails.
    """
    retry_attempts = 3
    retry_delay = 10  # seconds

    for attempt in range(retry_attempts):
        try:
            conn = pyodbc.connect(
                f"DRIVER={{ODBC Driver 17 for SQL Server}};"
                f"SERVER={DATABASE_CONFIG['server']};"
                f"DATABASE={DATABASE_CONFIG['database']};"
                f"UID={DATABASE_CONFIG['username']};"
                f"PWD={DATABASE_CONFIG['password']}"
            )
            return conn
        except pyodbc.Error as e:
            if attempt < retry_attempts - 1:
                print(f"Connection failed. Retrying in {retry_delay} seconds... (Attempt {attempt + 1}/{retry_attempts})")
                time.sleep(retry_delay)
            else:
                print(f"Error in connection after {retry_attempts} attempts: {e}")
                return None

def execute_query(query, params=None):
    conn = get_connection()
    if not conn:
        return None

    try:
        with conn.cursor() as cursor:
            cursor.execute(query, params or ())
            columns = [column[0] for column in cursor.description]
            data = cursor.fetchall()
            
            # Convert list of tuples to list of lists
            data = [list(row) for row in data]
            
            return pd.DataFrame(data, columns=columns)
    except pyodbc.Error as e:
        print(f"Error executing query: {e}")
        return None
    finally:
        conn.close()


def generate_query_with_conditions(table_name, columns, conditions=None):
    """
    Generates a SQL query for the given table with predefined columns based on user-specified conditions.
    :param table_name: The name of the table to query.
    :param columns: A list of columns to select.
    :param conditions: A dictionary of conditions where keys are column names and values are the required values.
    :return: A SQL query string and list of parameter values.
    """
    columns_part = ', '.join(columns)
    query = f"SELECT {columns_part} FROM {table_name}"

    if conditions:
        condition_parts = [f"{key}?" for key in conditions.keys()]
        query += " WHERE " + " AND ".join(condition_parts)
        values = list(conditions.values())
    else:
        values = []

    print(f"Generated Query: {query}")  # Debug print
    print(f"Values: {values}")  # Debug print
    return query, values

def get_columns_for_table(table_name):
    """
    Returns the columns for a specified table.
    :param table_name: The name of the table to get columns for.
    :return: A list of columns.
    """
    if table_name == "dbo.fts_feature_table":
        return [
            "adv_idx_uuid", "season_year", "match_date",
            "t1_player_games_count", "t1_player_tip_avg_overall", "t1_player_tip_avg_home",
            "t2_player_games_count", "t2_player_tip_avg_overall", "t2_player_tip_avg_away",
            "t1_team_games_count", "t1_team_fps_avg", "t1_team_fts_avg", "t1_team_tip1_fts_avg", "t1_team_tip0_fts_avg",
            "t2_team_games_count", "t2_team_fps_avg_overall", "t2_team_fts_avg_overall", "t2_team_tip1_fts_avg", "t2_team_tip0_fts_avg",
            "lg_games_count", "lg_t1_tip1_avg", "lg_t1_tip0_avg", "lg_t2_tip1_avg", "lg_t2_tip0_avg"
        ]
    else:
        raise ValueError("Unknown table name")

def fetch_data(query, params=None):
    """
    Fetches data using the provided query.
    :param query: SQL query string.
    :param params: Tuple of parameters to use in the query.
    :return: Fetched data as a list of dictionaries, or None if an error occurs.
    """
    df = execute_query(query, params)
    if df is not None:
        return df.to_dict(orient='records')
    return None
