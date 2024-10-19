# src/data_processing/data_cleaner.py

import pandas as pd
import numpy as np
from typing import Union, Dict, Any, Optional, List
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_data_pipeline(
    df: pd.DataFrame,
    target_column: Optional[str] = None, 
    remove_duplicates: bool = True,
    missing_value_strategy: Union[str, Dict[str, str]] = 'drop',
    normalize_columns: bool = False,
    type_conversion_map: Optional[Dict[str, type]] = None,
    scaling_params: Optional[Dict[str, Any]] = None,
    encoding_strategy: str = 'one_hot'
) -> pd.DataFrame:
    """
    Main entry function to clean the data using common data processing steps.
    
    :param df: Input DataFrame to be cleaned.
    :param target_column: The name of the target column that should not be processed like features.
    :param remove_duplicates: Whether to remove duplicate rows. Default is True.
    :param missing_value_strategy: Strategy for handling missing values; either global or column-specific. Default is 'drop'.
    :param normalize_columns: Whether to normalize column names. Default is True.
    :param type_conversion_map: Dictionary defining the data type conversion for each column.
    :param scaling_params: Dictionary containing the columns to scale and exclude from scaling.
    :param encoding_strategy: Strategy for encoding categorical variables ('one_hot', 'label'). Default is 'one_hot'.
    :return: Cleaned DataFrame.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")

    # Separate target from features    
    target = None
    if target_column:
        if target_column in df.columns:
            target = df[[target_column]].copy()
            df = df.drop(columns=[target_column])
            logger.info(f"Target column '{target_column}' separated from features")
        else:
            logger.warning(f"Target column '{target_column}' not found in DataFrame.")


    if remove_duplicates:
        initial_rows = len(df)
        df = df.drop_duplicates()
        removed_rows = initial_rows - len(df)
        logger.info(f"Removed {removed_rows} duplicate rows")

    df = handle_missing_values(df, missing_value_strategy)
    logger.info("Missing values handled")


    if normalize_columns:
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        logger.info("Column names normalized")

    if type_conversion_map:
        df = convert_data_types(df, type_conversion_map)
        logger.info("Data types converted as specified")

    if scaling_params:
        if not isinstance(scaling_params, dict) or 'columns_to_scale' not in scaling_params:
            raise ValueError("scaling_params must be a dictionary with 'columns_to_scale' key")
        df = scale_features(df, **scaling_params)
        logger.info("Features scaled according to provided parameters")

    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_columns) > 0:
        df = encode_categorical(df, encoding_strategy)
        logger.info(f"Categorical variables encoded using '{encoding_strategy}' strategy")
    else:
        logger.info("No categorical columns found. Skipping encoding step.")


    # Reattach target column if it was separated and indices match
    if target is not None:
        # Check if indices match before concatenation
        if not df.index.equals(target.index):
            raise ValueError("Indices of features and target do not match for concatenation.")
        df = pd.concat([df, target], axis=1)
        logger.info("Target column reattached to features")

    logger.info("Data cleaning process completed")
    return df

def handle_missing_values(df: pd.DataFrame, strategy: Union[str, Dict[str, str]]) -> pd.DataFrame:
    """
    Handle missing values in the DataFrame.
    
    :param df: Input DataFrame.
    :param strategy: Strategy for handling missing values. Can be a string for global strategy or a dict for column-specific strategies.
    :return: DataFrame with missing values handled.
    """
    if isinstance(strategy, str):
        if strategy == 'drop':
            return df.dropna()
        else:
            # Process all columns with the same strategy in bulk
            imputer = SimpleImputer(strategy=strategy)
            df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)
            return df_imputed
    elif isinstance(strategy, dict):
        for col, strat in strategy.items():
            if col in df.columns:
                if strat == 'drop':
                    df = df.dropna(subset=[col])
                else:
                    imputer = SimpleImputer(strategy=strat)
                    # Flatten the 2D array to 1D using .ravel() when assigning back to the DataFrame column
                    df[col] = imputer.fit_transform(df[[col]]).ravel()
    else:
        raise ValueError("Strategy must be either a string or a dictionary")
    return df


def convert_data_types(df: pd.DataFrame, type_conversion_map: Dict[str, type]) -> pd.DataFrame:
    """
    Convert data types of specified columns.

    :param df: Input DataFrame.
    :param type_conversion_map: Dictionary mapping column names to desired data types.
    :return: DataFrame with converted data types.
    """
    for col, dtype in type_conversion_map.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except ValueError as e:
                print(f"Error converting column '{col}' to {dtype}: {str(e)}")
    return df

def scale_features(df: pd.DataFrame, columns_to_scale: List[str], exclude_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Scale specified numeric features using StandardScaler.

    :param df: Input DataFrame.
    :param columns_to_scale: List of columns to scale.
    :param exclude_columns: List of columns to exclude from scaling.
    :return: DataFrame with scaled features.
    """
    if not isinstance(columns_to_scale, list):
        raise ValueError("columns_to_scale must be a list of column names")

    # Convert column names to lowercase to match normalized DataFrame
    columns_to_scale = [col.lower() for col in columns_to_scale]

    # Validate that columns_to_scale exist in the DataFrame
    missing_columns = [col for col in columns_to_scale if col not in df.columns]
    if missing_columns:
        raise KeyError(f"The following columns are not in the DataFrame: {missing_columns}")

    if exclude_columns:
        exclude_columns = [col.lower() for col in exclude_columns]
        columns_to_scale = [col for col in columns_to_scale if col not in exclude_columns]

    numeric_columns = df[columns_to_scale].select_dtypes(include=[np.number]).columns
    non_numeric = set(columns_to_scale) - set(numeric_columns)
    if non_numeric:
        print(f"Warning: The following columns are not numeric and will be skipped for scaling: {non_numeric}")

    if len(numeric_columns) > 0:
        scaler = StandardScaler()
        df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    
    return df



def encode_categorical(df: pd.DataFrame, strategy: str = 'one_hot') -> pd.DataFrame:
    """
    Encode categorical variables in the DataFrame.

    :param df: Input DataFrame.
    :param strategy: Encoding strategy ('one_hot' or 'label').
    :return: DataFrame with encoded categorical variables.
    """
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_columns) == 0:
        logger.info("No categorical columns to encode.")
        return df
    
    if strategy not in ['one_hot', 'label']:
        raise ValueError("Invalid encoding strategy. Use 'one_hot' or 'label'.")


    if strategy == 'one_hot':
        # Updated parameter 'sparse_output' instead of 'sparse'
        encoder = OneHotEncoder(sparse_output=True, handle_unknown='ignore')
        encoded = encoder.fit_transform(df[categorical_columns])
        feature_names = encoder.get_feature_names_out(categorical_columns)
        encoded_df = pd.DataFrame.sparse.from_spmatrix(
            encoded,
            columns=feature_names,
            index=df.index
        )
        df = pd.concat([df.drop(columns=categorical_columns), encoded_df], axis=1)
        logger.info(f"Applied one-hot encoding to {len(categorical_columns)} columns")
    elif strategy == 'label':
        for col in categorical_columns:
            df[col] = df[col].astype('category').cat.codes
            print(f"Warning: '{col}' encoded as integers. Ensure this is suitable for downstream processing.")
            logger.info(f"Applied label encoding to {len(categorical_columns)} columns")
    return df


def clean_data_list(
    data_list: List[pd.DataFrame],  # List of DataFrames to be cleaned
    target_column: str,  # Target column must be specified
    **kwargs
) -> List[pd.DataFrame]:
    """
    Clean a list of DataFrames using the same data cleaning parameters.

    :param data_list: List of DataFrames to be cleaned.
    :param target_column: The name of the target column that should not be processed like features.
    :param kwargs: Additional keyword arguments to pass to clean_data_pipeline.
    :return: List of cleaned DataFrames.
    """
    cleaned_data_list = []

    for df in data_list:
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Each entry in data_list must be a pandas DataFrame, got {type(df)} instead.")
        
        # Clean the DataFrame using the provided parameters
        cleaned_df = clean_data_pipeline(df, target_column=target_column, **kwargs)
        cleaned_data_list.append(cleaned_df)
    
    return cleaned_data_list

def process_data_entries(
    data_list: List[Dict[str, Any]], 
    target_column: Optional[str] = None, 
    remove_duplicates: bool = True,
    missing_value_strategy: Union[str, Dict[str, str]] = 'drop',
    normalize_columns: bool = True,
    type_conversion_map: Optional[Dict[str, type]] = None,
    scaling_params: Optional[Dict[str, Any]] = None,
    encoding_strategy: str = 'one_hot'
) -> List[Dict[str, Any]]:
    """
    Handle data entry preparation and call clean_data_list to clean the DataFrames.

    :param data_list: List of dictionaries containing DataFrames and associated metadata.
    :param target_column: The name of the target column that should not be processed like features.
    :param remove_duplicates: Whether to remove duplicate rows. Default is True.
    :param missing_value_strategy: Strategy for handling missing values; either global or column-specific. Default is 'drop'.
    :param normalize_columns: Whether to normalize column names. Default is True.
    :param type_conversion_map: Dictionary defining the data type conversion for each column.
    :param scaling_params: Dictionary containing the columns to scale and exclude from scaling.
    :param encoding_strategy: Strategy for encoding categorical variables ('one_hot', 'label'). Default is 'one_hot'.
    :return: List of dictionaries with cleaned DataFrames, preserving original metadata.
    """
    # Prepare the list of DataFrames to be cleaned
    df_list = [entry['data'] for entry in data_list]

    # Clean the DataFrames using the same parameters
    cleaned_dataframes = clean_data_list(
        df_list,
        target_column=target_column,
        remove_duplicates=remove_duplicates,
        missing_value_strategy=missing_value_strategy,
        normalize_columns=normalize_columns,
        type_conversion_map=type_conversion_map,
        scaling_params=scaling_params,
        encoding_strategy=encoding_strategy
    )

    # Create cleaned data entries with original metadata preserved
    cleaned_entries = []
    for original_entry, cleaned_df in zip(data_list, cleaned_dataframes):
        cleaned_entry = original_entry.copy()
        cleaned_entry['data'] = cleaned_df  # Replace the original DataFrame with the cleaned one
        cleaned_entries.append(cleaned_entry)

    return cleaned_entries



import pandas as pd
import numpy as np

if __name__ == "__main__":
    # Sample DataFrames
    df1 = pd.DataFrame({'A': [1, 2, np.nan, 4], 'B': ['x', 'y', 'z', 'x'], 'C': [0.1, 0.2, 0.3, 0.4], 'target': [1, 0, 1, 0]})
    df2 = pd.DataFrame({'A': [5, 6, np.nan, 8], 'B': ['a', 'b', 'c', 'a'], 'C': [0.5, 0.6, 0.7, 0.8], 'target': [0, 1, 1, 0]})

    # List of dictionaries containing the DataFrames to be cleaned
    data_entries = [
        {
            'window_size': 5,
            'data': df1,
            'params': {
                'target_column': 'target',
                'window_size': 5,
                'remove_duplicates': True,
                'missing_value_strategy': {'A': 'mean', 'B': 'most_frequent'},
                'scaling_params': {'columns_to_scale': ['A', 'C']},
                'encoding_strategy': 'one_hot'
            }
        },
        {
            'window_size': 10,
            'data': df2,
            'params': {
                'target_column': 'target',
                'window_size': 10,
                'remove_duplicates': True,
                'missing_value_strategy': {'A': 'mean', 'B': 'most_frequent'},
                'scaling_params': {'columns_to_scale': ['A', 'C']},
                'encoding_strategy': 'one_hot'
            }
        }
    ]

    # Process data entries using the adjusted function
    cleaned_entries = process_data_entries(
        data_list=data_entries,
        target_column='target',
        remove_duplicates=True,
        missing_value_strategy={'A': 'mean', 'B': 'most_frequent'},
        scaling_params={'columns_to_scale': ['A', 'C']},
        encoding_strategy='one_hot'
    )

    # Output the cleaned DataFrames
    for cleaned_entry in cleaned_entries:
        print("Window Size:", cleaned_entry['window_size'])
        print("Cleaned DataFrame:\n", cleaned_entry['data'])
        print("=" * 40)
