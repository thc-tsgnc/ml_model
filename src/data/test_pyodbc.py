import sys
import os

# Adjust sys.path to ensure src is included
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import pyodbc
from src.data.data_fetcher import fetch_fts_data
from src.data.database import fetch_data

print(pyodbc.version)

def test_connection():
    query = "SELECT TOP 1 * FROM adv_idx"  # Modify this query to match your database schema
    data = fetch_data(query)
    
    if data:
        print("Data fetched successfully:")
        for row in data:
            print(row)
    else:
        print("No data fetched or error occurred.")

def test_fetch_fts_data():
    start_season = "201617"
    end_season = "202324"
    is_feat = "Yes"
    avg_type = "SMA"
    player_window_size = 30
    team_window_size = 30
    team_avg_type = "SMA"
    team_data_type = "Overall"
    lg_window_size = 30
    lg_avg_type = "SMA"

    df = fetch_fts_data(start_season, end_season, is_feat, avg_type, player_window_size, team_window_size, team_avg_type, team_data_type, lg_window_size, lg_avg_type)
    
    if df is not None:
        print("Data fetched successfully:")
        print(df.head())
    else:
        print("No data fetched or error occurred.")

if __name__ == "__main__":
    # Uncomment the desired test function to run
    # test_connection()
    test_fetch_fts_data()
