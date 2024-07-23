# data_cleaner.py

import pandas as pd
from typing import Union, List, Dict, Any


def clean_data(data: Union[pd.DataFrame, List[pd.DataFrame]]) -> Union[pd.DataFrame, List[pd.DataFrame]]:
    """
    Main function to clean data. Accepts either a single DataFrame or a list of DataFrames.
    Returns cleaned data in the same format as input.
    """
    if isinstance(data, pd.DataFrame):
        return clean_single_dataframe(data)
    elif isinstance(data, list):
        return [clean_single_dataframe(df) for df in data]
    else:
        raise ValueError("Input must be a DataFrame or a list of DataFrames")

def clean_single_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to clean a single DataFrame.
    """
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = normalize_column_names(df)
    df = convert_data_types(df)
    return df

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    # Implementation for removing duplicate rows
    return df.drop_duplicates()

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    # Implementation for handling missing values
    # For example, drop rows with any NaN values
    return df.dropna()

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    # Implementation for normalizing column names
    # For example, convert to lowercase and replace spaces with underscores
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    return df

def convert_data_types(df: pd.DataFrame) -> pd.DataFrame:
    # Implementation for converting data types
    # This would depend on your specific needs
    return df

def clean_data_list(data_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean the DataFrames in the data_list structure.
    """
    cleaned_data_list = []
    
    for entry in data_list:
        cleaned_df = clean_data(entry['data'])
        
        cleaned_entry = {
            'window_size': entry['window_size'],
            'data': cleaned_df,
            'params': entry['params']
        }
        
        cleaned_data_list.append(cleaned_entry)
    
    return cleaned_data_list
