# src/feature_store/feature_store_manager.py

import pandas as pd
from typing import List, Dict, Optional, Any, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_client(fs_client: Any) -> None:
    if fs_client is None:
        raise ValueError("Feature Store client is not provided.")

def write_features_to_store(fs_client: Any, features_df: pd.DataFrame, feature_table_name: str, primary_key: str, mode: str = 'merge') -> None:
    validate_client(fs_client)
    
    if not validate_inputs(features_df, [primary_key]):
        raise ValueError(f"Invalid DataFrame or missing primary key: {primary_key}")

    if mode not in ['merge', 'overwrite', 'append']:
        raise ValueError(f"Invalid mode: {mode}. Must be 'merge', 'overwrite', or 'append'.")

    try:
        fs_client.write_table(
            name=feature_table_name,
            df=features_df,
            mode=mode,
            primary_keys=[primary_key]
        )
        logger.info(f"Successfully wrote features to table '{feature_table_name}' using mode '{mode}'")
    except Exception as e:
        handle_error(e, f"Failed to write features to {feature_table_name}")

def read_features_from_store(fs_client: Any, feature_table_name: str, feature_names: List[str], entity_keys: pd.DataFrame) -> pd.DataFrame:
    validate_client(fs_client)
    
    if not validate_feature_table_exists(fs_client, feature_table_name):
        raise ValueError(f"Feature table '{feature_table_name}' does not exist.")

    try:
        feature_lookups = [fs_client.get_table(feature_table_name).select(*feature_names)]
        return fs_client.read_table(
            name=feature_table_name,
            features=feature_lookups,
            lookup_key=entity_keys
        )
    except Exception as e:
        handle_error(e, f"Failed to read features from {feature_table_name}")

def read_filtered_features(fs_client: Any, feature_table_name: str, feature_names: List[str], 
                           entity_keys: pd.DataFrame, conditions: Dict[str, Any]) -> pd.DataFrame:
    validate_client(fs_client)
    
    if not validate_feature_table_exists(fs_client, feature_table_name):
        raise ValueError(f"Feature table '{feature_table_name}' does not exist.")

    try:
        # Convert conditions to a SQL-like string
        filter_conditions = " AND ".join([f"{k} = '{v}'" if isinstance(v, str) else f"{k} = {v}" for k, v in conditions.items()])
        
        # Use the feature store client's filtering capabilities
        feature_lookups = [fs_client.get_table(feature_table_name).filter(filter_conditions).select(*feature_names)]
        
        return fs_client.read_table(
            name=feature_table_name,
            features=feature_lookups,
            lookup_key=entity_keys
        )
    except Exception as e:
        handle_error(e, f"Failed to read filtered features from {feature_table_name}")

def list_available_features(fs_client: Any, feature_table_name: Optional[str] = None) -> List[str]:
    validate_client(fs_client)
    
    try:
        if feature_table_name:
            if not validate_feature_table_exists(fs_client, feature_table_name):
                logger.warning(f"Feature table '{feature_table_name}' does not exist.")
                return []
            return fs_client.get_table(feature_table_name).schema.names
        else:
            return [table.name for table in fs_client.list_tables()]
    except Exception as e:
        logger.error(f"Failed to list available features: {str(e)}")
        return []

def get_feature_table_metadata(fs_client: Any, feature_table_name: str) -> Dict:
    validate_client(fs_client)
    
    if not validate_feature_table_exists(fs_client, feature_table_name):
        raise ValueError(f"Feature table '{feature_table_name}' does not exist.")

    try:
        table = fs_client.get_table(feature_table_name)
        return {
            "name": table.name,
            "schema": {col.name: str(col.datatype) for col in table.schema},
            "description": table.description,
            "creation_timestamp": table.creation_timestamp,
            "primary_keys": table.primary_keys
        }
    except Exception as e:
        handle_error(e, f"Failed to get metadata for {feature_table_name}")

def update_feature_table_schema(fs_client: Any, feature_table_name: str, new_schema: Dict[str, str]) -> None:
    validate_client(fs_client)
    
    if not validate_feature_table_exists(fs_client, feature_table_name):
        raise ValueError(f"Feature table '{feature_table_name}' does not exist.")

    try:
        table = fs_client.get_table(feature_table_name)
        current_schema = {col.name: str(col.datatype) for col in table.schema}
        
        for col, dtype in new_schema.items():
            if col in current_schema and current_schema[col] != dtype:
                raise ValueError(f"Incompatible schema change for column '{col}': {current_schema[col]} to {dtype}")
        
        table.update_schema(new_schema)
        logger.info(f"Successfully updated schema for table '{feature_table_name}'")
    except Exception as e:
        handle_error(e, f"Failed to update schema for {feature_table_name}")

def delete_features(fs_client: Any, feature_table_name: str, feature_names: Optional[List[str]] = None) -> None:
    validate_client(fs_client)
    
    if not validate_feature_table_exists(fs_client, feature_table_name):
        raise ValueError(f"Feature table '{feature_table_name}' does not exist.")

    try:
        if feature_names:
            table = fs_client.get_table(feature_table_name)
            existing_features = set(table.schema.names)
            non_existent_features = set(feature_names) - existing_features
            if non_existent_features:
                raise ValueError(f"The following features do not exist in the table: {non_existent_features}")
            
            table.delete_features(feature_names)
            logger.info(f"Successfully deleted features {feature_names} from table '{feature_table_name}'")
        else:
            fs_client.drop_table(feature_table_name)
            logger.info(f"Successfully deleted entire table '{feature_table_name}'")
    except Exception as e:
        handle_error(e, f"Failed to delete features from {feature_table_name}")

def validate_inputs(data: pd.DataFrame, required_columns: List[str]) -> bool:
    if data.empty:
        raise ValueError("DataFrame is empty.")
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame.")
    if not all(col in data.columns for col in required_columns):
        raise ValueError(f"DataFrame is missing required columns: {set(required_columns) - set(data.columns)}")
    return True

def validate_feature_table_exists(fs_client: Any, feature_table_name: str) -> bool:
    try:
        fs_client.get_table(feature_table_name)
        return True
    except Exception:
        return False

def handle_error(exception: Exception, message: str) -> None:
    full_message = f"{message}: {str(exception)}"
    logger.error(full_message)
    raise type(exception)(full_message)

def add_columns_to_schema(fs_client: Any, feature_table_name: str, new_columns: Dict[str, str]) -> None:
    """
    Adds new columns to the schema of the specified feature table in the Feature Store.
    
    Parameters:
        fs_client: The feature store client.
        feature_table_name (str): The name of the feature table.
        new_columns (Dict[str, str]): A dictionary where keys are column names and values are data types.
    
    Raises:
        ValueError: If the table does not exist or new columns conflict with existing ones.
    """
    validate_client(fs_client)
    
    if not validate_feature_table_exists(fs_client, feature_table_name):
        raise ValueError(f"Feature table '{feature_table_name}' does not exist.")
    
    try:
        table = fs_client.get_table(feature_table_name)
        current_schema = {col.name: str(col.datatype) for col in table.schema}

        conflicting_columns = set(new_columns.keys()).intersection(current_schema.keys())
        if conflicting_columns:
            raise ValueError(f"The following columns already exist: {conflicting_columns}")

        updated_schema = {**current_schema, **new_columns}
        table.update_schema(updated_schema)
        logger.info(f"Successfully added new columns to table '{feature_table_name}'")
    
    except Exception as e:
        handle_error(e, f"Failed to add columns to schema for {feature_table_name}")

def remove_columns_from_schema(fs_client: Any, feature_table_name: str, columns_to_remove: List[str]) -> None:
    """
    Removes specified columns from the schema of the feature table in the Feature Store.
    
    Parameters:
        fs_client: The feature store client.
        feature_table_name (str): The name of the feature table.
        columns_to_remove (List[str]): A list of column names to remove.
    
    Raises:
        ValueError: If any column in `columns_to_remove` does not exist.
    """
    validate_client(fs_client)
    
    if not validate_feature_table_exists(fs_client, feature_table_name):
        raise ValueError(f"Feature table '{feature_table_name}' does not exist.")
    
    try:
        table = fs_client.get_table(feature_table_name)
        current_schema = {col.name: str(col.datatype) for col in table.schema}

        non_existent_columns = set(columns_to_remove) - set(current_schema.keys())
        if non_existent_columns:
            raise ValueError(f"The following columns do not exist: {non_existent_columns}")

        updated_schema = {k: v for k, v in current_schema.items() if k not in columns_to_remove}
        table.update_schema(updated_schema)
        logger.info(f"Successfully removed columns from table '{feature_table_name}'")
    
    except Exception as e:
        handle_error(e, f"Failed to remove columns from schema for {feature_table_name}")

def rename_column_in_schema(fs_client: Any, feature_table_name: str, old_column_name: str, new_column_name: str) -> None:
    """
    Renames a column in the schema of the feature table in the Feature Store.
    
    Parameters:
        fs_client: The feature store client.
        feature_table_name (str): The name of the feature table.
        old_column_name (str): The existing column name to rename.
        new_column_name (str): The new column name.
    
    Raises:
        ValueError: If the old column name does not exist or the new column name conflicts with existing columns.
    """
    validate_client(fs_client)
    
    if not validate_feature_table_exists(fs_client, feature_table_name):
        raise ValueError(f"Feature table '{feature_table_name}' does not exist.")
    
    try:
        table = fs_client.get_table(feature_table_name)
        current_schema = {col.name: str(col.datatype) for col in table.schema}

        if old_column_name not in current_schema:
            raise ValueError(f"Column '{old_column_name}' does not exist.")
        if new_column_name in current_schema:
            raise ValueError(f"Column '{new_column_name}' already exists.")

        updated_schema = {new_column_name if k == old_column_name else k: v for k, v in current_schema.items()}
        table.update_schema(updated_schema)
        logger.info(f"Successfully renamed column '{old_column_name}' to '{new_column_name}' in table '{feature_table_name}'")
    
    except Exception as e:
        handle_error(e, f"Failed to rename column in schema for {feature_table_name}")

def modify_column_data_type(fs_client: Any, feature_table_name: str, column_name: str, new_data_type: str) -> None:
    """
    Modifies the data type of an existing column in the schema of the feature table in the Feature Store.
    
    Parameters:
        fs_client: The feature store client.
        feature_table_name (str): The name of the feature table.
        column_name (str): The name of the column to modify.
        new_data_type (str): The new data type (e.g., 'float', 'string').
    
    Raises:
        ValueError: If the column does not exist or the new data type is incompatible.
    """
    validate_client(fs_client)
    
    if not validate_feature_table_exists(fs_client, feature_table_name):
        raise ValueError(f"Feature table '{feature_table_name}' does not exist.")
    
    try:
        table = fs_client.get_table(feature_table_name)
        current_schema = {col.name: str(col.datatype) for col in table.schema}

        if column_name not in current_schema:
            raise ValueError(f"Column '{column_name}' does not exist.")

        updated_schema = {k: (new_data_type if k == column_name else v) for k, v in current_schema.items()}
        table.update_schema(updated_schema)
        logger.info(f"Successfully modified data type of column '{column_name}' to '{new_data_type}' in table '{feature_table_name}'")
    
    except Exception as e:
        handle_error(e, f"Failed to modify column data type in schema for {feature_table_name}")