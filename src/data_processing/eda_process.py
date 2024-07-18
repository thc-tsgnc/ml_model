# src/data_processing/eda_process.py

import sys
import os

# Ensure the correct directory is included in the sys.path for the VSCode environment
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from data_fetcher.fts_data_fetcher import get_fts_data

def load_data(data_type, **kwargs):
    """
    Load data based on the specified data type.
    :param data_type: Type of data to load (e.g., 'fts').
    :param kwargs: Additional parameters for data fetching functions.
    :return: Loaded DataFrame.
    """
    if data_type == 'fts':
        return get_fts_data(**kwargs)
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

