# src/task_workflows/tasks/find_optimal_window_size.py

from data.fetcher.fts_data_fetcher import get_fts_data
from task_workflows.task_registry import register_task
from data.processing.data_cleaner import clean_data_list
from typing import List, Dict, List, Any, Tuple
import pandas as pd
from modeling.lazy_predict_modeling import run_lazy_predict_modeling_for_window_size
from data.processing.eda_process import perform_eda_regression, calculate_feature_variance
import os
import logging
from datetime import datetime
import pprint


logger = logging.getLogger(__name__)

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Use this at the beginning of your main script

@register_task("find_optimal_window_size")
def find_optimal_window_size(task_params: dict, data_params: dict, preprocess_params: dict, model_params: dict, output_params: dict, data_input_mode: str = "database", provided_data: pd.DataFrame = None):
    """
    Main function to find the optimal window size for different play types.
    
    Args:
    task_params (dict): Task-specific parameters including play_type
    data_params (dict): Data retrieval parameters
    preprocess_params (dict): Data preprocessing parameters
    model_params (dict): Model configuration parameters
    output_params (dict): Output and reporting parameters
    data_input_mode (str): Mode of data input ("database" or "dataframe")
    provided_data (pd.DataFrame): DataFrame provided directly for processing (used when data_input_mode is "dataframe")
    
    Returns:
    tuple: Optimal window size and modeling results
    """
    setup_logging()
    play_type = task_params.get('play_type')
    if not play_type:
        raise ValueError("play_type must be specified in task_params")
    
    # Validate inputs
    validate_inputs(play_type, task_params, data_params, preprocess_params, model_params, output_params, data_input_mode, provided_data)

    # Retrieve data based on play_type
    data_list = retrieve_data(play_type, task_params, data_params, data_input_mode, provided_data)

    # Preprocess data (potentially different for each play_type)
    preprocessed_data_list = preprocess_data(play_type, data_list, task_params, preprocess_params)

    # Analyze and model data (may vary based on play_type)
    modeling_results = analyze_and_model(play_type, preprocessed_data_list, model_params, task_params)

    # Process results (potentially different criteria for each play_type)
    save_types = output_params.get('save_type', ['csv', 'json'])
    processed_results, optimal_window_size = process_results(modeling_results, task_params, model_params, save_types)

    # Save results (may have different formats or locations based on play_type)
    save_results(play_type, processed_results, optimal_window_size, output_params)

    return optimal_window_size, processed_results


def validate_inputs(play_type: str, task_params: dict, data_params: dict, preprocess_params: dict, model_params: dict, output_params: dict, data_input_mode: str, provided_data: pd.DataFrame = None):
    """
    Validates the essential input parameters for the find_optimal_window_size function.

    Args:
    play_type (str): Type of play (e.g., "fts", "hdp")
    task_params (dict): Task-specific parameters
    data_params (dict): Data retrieval parameters
    preprocess_params (dict): Data preprocessing parameters
    model_params (dict): Model configuration parameters
    output_params (dict): Output and reporting parameters
    data_input_mode (str): Mode of data input ("database" or "dataframe")
    provided_data (pd.DataFrame): DataFrame provided directly for processing

    Raises:
    ValueError: If any required parameter is missing or invalid
    """
    # Validate play_type
    if play_type not in ["fts", "hdp"]:
        raise ValueError(f"Invalid play_type: {play_type}. Must be 'fts' or 'hdp'.")

    # Check if required parameter dictionaries exist
    if not all(isinstance(param, dict) for param in [task_params, data_params, preprocess_params, model_params, output_params]):
        raise ValueError("task_params, data_params, preprocess_params, model_params, and output_params must be dictionaries.")

    # Validate data_input_mode and provided_data
    if data_input_mode not in ["database", "dataframe"]:
        raise ValueError(f"Invalid data_input_mode: {data_input_mode}. Must be 'database' or 'dataframe'.")

    if data_input_mode == "dataframe":
        if provided_data is None:
            raise ValueError("provided_data must be supplied when data_input_mode is 'dataframe'")


    # Check for essential task_params
    if "window_sizes" not in task_params or not isinstance(task_params["window_sizes"], list):
        raise ValueError("task_params must contain 'window_sizes' as a list")

    if "target_column" not in task_params:
        raise ValueError("task_params must contain 'target_column'")

    logger.info("Basic input validation completed successfully.")

    

def retrieve_data(play_type: str, task_params: dict, data_params: dict, data_input_mode: str, provided_data: pd.DataFrame = None) -> List[Dict[str, Any]]:
    """
    Entry point for data retrieval. Directs to appropriate data retrieval method based on input mode and play type.
    """
    logger.info(f"Starting data retrieval process for play_type: {play_type}, data_input_mode: {data_input_mode}")

    # Quick validation
    """
    if data_input_mode not in ["database", "dataframe"]:
        raise ValueError(f"Invalid data_input_mode: {data_input_mode}. Must be 'database' or 'dataframe'.")
    if data_input_mode == "dataframe" and provided_data is None:
        raise ValueError("provided_data must be supplied when data_input_mode is 'dataframe'")
    """
    
    if data_input_mode == "database":
        return get_data_from_database(play_type, task_params, data_params)
    else:
        # return process_provided_dataframes(provided_data, task_params)
        # window_data = {window_size: provided_data for window_size in task_params['window_sizes']}
        # return process_provided_dataframes(window_data, task_params)
        return process_provided_dataframes(provided_data, task_params)
        # return process_provided_dataframes({task_params['window_size']: provided_data}, task_params)

    


def get_data_from_database(play_type: str, task_params: dict, data_params: dict) -> List[Dict[str, Any]]:
    """
    Retrieves data from the database based on play type.
    """
    logger.info(f"Retrieving data from database for play_type: {play_type}")
    
    window_sizes = task_params['window_sizes']
    data_list = []

    for window_size in window_sizes:
        try:
            if play_type == "fts":
                df = get_fts_data(
                    start_season=data_params['conditions']['season_year_start'],
                    end_season=data_params['conditions']['season_year_end'],
                    is_feat=data_params['conditions']['is_feat'],
                    player_avg_type=data_params['conditions']['player_avg_type'],
                    player_window_size=window_size,
                    team_avg_type=data_params['conditions']['team_avg_type'],
                    team_window_size=window_size,
                    team_data_type=data_params['conditions']['team_data_type'],
                    lg_avg_type=data_params['conditions']['lg_avg_type'],
                    lg_window_size=window_size,
                    excluded_columns=[],
                    feat_columns=task_params['feature_set'],
                    target_column=task_params['target_column']
                )
            elif play_type == "hdp":
                # Assuming we have a similar function for HDP data
                # df = get_hdp_data(...)
                raise NotImplementedError("HDP data retrieval not yet implemented")
            else:
                raise ValueError(f"Unsupported play type: {play_type}")

            if df is not None and not df.empty:
                data_list.append(create_data_entry(df, window_size, task_params, data_params))
            else:
                logger.warning(f"No data retrieved for window size {window_size}")
        except Exception as e:
            logger.error(f"Error retrieving {play_type} data for window size {window_size}: {str(e)}")
            raise

    return data_list

def process_provided_dataframes(provided_data: Dict[int, pd.DataFrame], task_params: dict) -> List[Dict[str, Any]]:
    """
    Processes the provided DataFrames for each window size.

    Args:
    provided_data (Dict[int, pd.DataFrame]): A dictionary where keys are window sizes and values are corresponding DataFrames
    task_params (dict): Task-specific parameters

    Returns:
    List[Dict[str, Any]]: List of data entries for each window size
    """
    logger.info("Processing provided DataFrames")
    
    data_list = []

    for window_size, df in provided_data.items():
        # Validate each provided DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Provided data for window size {window_size} must be a pandas DataFrame")
        if task_params['target_column'] not in df.columns:
            raise ValueError(f"Target column '{task_params['target_column']}' not found in provided DataFrame for window size {window_size}")
        missing_features = set(task_params['feature_set']) - set(df.columns)
        if missing_features:
            raise ValueError(f"The following features are missing from the provided DataFrame for window size {window_size}: {missing_features}")

        # Create a data entry for each provided DataFrame
        data_list.append(create_data_entry(df, window_size, task_params))

    return data_list

def create_data_entry(df: pd.DataFrame, window_size: int, task_params: dict, data_params: dict = None) -> Dict[str, Any]:
    """
    Creates a standardized data entry dictionary.
    """
    logger.info(f"Creating data entry for window size {window_size}. Shape: {df.shape}")
    entry = {
        'window_size': window_size,
        'data': df,
        'metadata': {
            'target_column': task_params['target_column'],
            'feature_set': task_params['feature_set'],
            'shape': df.shape
        }
    }

    # Ensure data_params is provided before accessing it
    if data_params:
        entry['metadata']['conditions'] = data_params.get('conditions', {})
    else:
        # Handle the case where data_params is None
        entry['metadata']['conditions'] = {}
    return entry


def preprocess_data(play_type: str, data_list: List[Dict[str, Any]], task_params: dict, preprocess_params: dict) -> List[Dict[str, Any]]:
    """
    Preprocesses the data based on the specified parameters using the clean_data_list function.

    Args:
    play_type (str): Type of play (e.g., "fts", "hdp")
    data_list (List[Dict[str, Any]]): List of data entries retrieved from retrieve_data
    preprocess_params (dict): Configurations for data preprocessing

    Returns:
    List[Dict[str, Any]]: List of preprocessed data entries
    """
    logger.info(f"Starting data preprocessing for play_type: {play_type}")
    
    # Extract DataFrames and their metadata
    dataframes = [entry['data'] for entry in data_list]
    metadata = [{'window_size': entry['window_size'], 'metadata': entry.get('metadata', {})} for entry in data_list]
    
    # Extract preprocessing parameters
    target_column = task_params.get('target_column')
    remove_duplicates = preprocess_params.get('remove_duplicates', True)
    missing_value_strategy = preprocess_params.get('missing_value_strategy', 'drop')
    normalize_columns = preprocess_params.get('normalize_columns', True)
    type_conversion_map = preprocess_params.get('type_conversion_map')
    scaling_params = preprocess_params.get('scaling_params')
    encoding_strategy = preprocess_params.get('encoding_strategy', 'one_hot')
    
    try:
        # Clean the data using clean_data_list
        cleaned_dataframes = clean_data_list(
            dataframes,
            target_column=target_column,
            remove_duplicates=remove_duplicates,
            missing_value_strategy=missing_value_strategy,
            normalize_columns=normalize_columns,
            type_conversion_map=type_conversion_map,
            scaling_params=scaling_params,
            encoding_strategy=encoding_strategy
        )
        
        # Reconstruct the data entries with cleaned DataFrames
        preprocessed_data_list = []
        for cleaned_df, meta in zip(cleaned_dataframes, metadata):
            preprocessed_entry = {
                'window_size': meta['window_size'],
                'data': cleaned_df,
                'metadata': meta['metadata']
            }
            preprocessed_data_list.append(preprocessed_entry)
        
        logger.info(f"Preprocessing completed for {len(preprocessed_data_list)} data entries")
        return preprocessed_data_list
    
    except Exception as e:
        logger.error(f"Error in data preprocessing: {str(e)}")
        raise
    

def analyze_and_model(
    play_type: str, 
    preprocessed_data_list: List[Dict[str, Any]], 
    model_params: dict, 
    task_params: dict
) -> List[Dict[str, Any]]:
    """
    Analyzes the preprocessed data, performs EDA, and runs modeling using LazyPredict for all window sizes in batch mode.
    Returns a structured list containing results and metadata for each window size.
    
    Args:
    play_type (str): Type of play (e.g., 'fts', 'hdp').
    preprocessed_data_list (List[Dict[str, Any]]): List of dictionaries containing preprocessed data and metadata.
    model_params (dict): Dictionary containing model-specific parameters.
    task_params (dict): Dictionary containing task-specific parameters.

    Returns:
    List[Dict[str, Any]]: List of dictionaries containing EDA results, model results, and metadata for each window size.
    """
    # Prepare a list to store DataFrames and window sizes
    data_frames = []
    window_sizes = []

    # Perform EDA for each DataFrame and store necessary metadata
    eda_results = []

    for entry in preprocessed_data_list:
        window_size = entry['window_size']
        df = entry['data']

        # Perform EDA analysis
        
        feature_variance = calculate_feature_variance(df, task_params['target_column'])
        regression_type = 'logistic'
        regression_results = perform_eda_regression(df, task_params['target_column'], regression_type)

        
        
        # Collect DataFrame and window size for batch processing later
        data_frames.append(df)
        window_sizes.append(window_size)

        # Store EDA results for later merging with the model results
        eda_results.append({
            'window_size': window_size,
            'feature_variance': feature_variance,
            'regression_results': regression_results
        })

    # Now perform batch modeling for all DataFrames at once
    lazy_predict_outputs = run_lazy_predict_modeling_for_window_size(
        data_frames, 
        task_params['target_column'], 
        model_params['model_type'],
        return_choices=["models"]  # Or modify if predictions are needed too
    )

    # Combine EDA results and LazyPredict model results into a single organized result
    organized_results = []

    for i, output in enumerate(lazy_predict_outputs):
        result_entry = {
            'window_size': window_sizes[i],  # Get the correct window size from earlier collection
            'feature_variance': eda_results[i]['feature_variance'],
            'regression_results': eda_results[i]['regression_results'],
            'models': output['models']  # Add the corresponding model results
        }

        organized_results.append(result_entry)

    return organized_results


def select_optimal_window_size(organized_results: List[Dict[str, Any]], primary_metric: str) -> int:
    """
    Selects the optimal window size based on the specified primary metric.
    """
    optimal_window_size = None
    best_score = float('-inf')

    for entry in organized_results:
        window_size = entry['window_size']
        models_df = entry['models']

        # Compute the best score for the primary metric
        max_score = models_df[primary_metric].max()
        if max_score > best_score:
            best_score = max_score
            optimal_window_size = window_size

    if optimal_window_size is not None:
        return optimal_window_size
    else:
        raise ValueError("Unable to determine optimal window size based on the primary metric")


# def process_results(organized_results: List[Dict[str, Any]], save_type: List[str], task_params: Dict[str, Any]) -> Dict[str, Any]:

def process_results(
    organized_results: List[Dict[str, Any]],
    task_params: Dict[str, Any],
    model_params: Dict[str, Any],
    save_types: List[str],
    primary_metric: str = 'Accuracy'
) -> Tuple[Dict[str, Any], int]:
    results = {}
    play_type = task_params.get('play_type', 'unknown')

    # Step 1: Call select_optimal_window_size to find the best window size
    optimal_window_size = select_optimal_window_size(organized_results, primary_metric)

    # Step 2: Process CSV Data
    if "csv" in save_types:
        csv_data = []
        for entry in organized_results:
            window_size = entry['window_size']

            # Extract feature_variance
            feature_variance = entry.get('feature_variance', None)

            # Extract regression_results
            regression_results = entry.get('regression_results', {})
            regression_type = 'logistic'  # Adjust if necessary
            regression_key = f'{regression_type}_regression'
            regression_metrics = regression_results.get(regression_key, {})
            mean_scoring = regression_metrics.get(f'mean_{model_params.get("scoring", "roc_auc")}', None)
            std_scoring = regression_metrics.get(f'std_{model_params.get("scoring", "roc_auc")}', None)

            # Extract top features
            feature_importance_key = f'feature_importance_{regression_type}'
            feature_importance_df = regression_results.get(feature_importance_key, pd.DataFrame())
            if not feature_importance_df.empty:
                top_features = feature_importance_df.head(3)['feature'].tolist()
                top_features_str = ', '.join(top_features)
            else:
                top_features_str = None

            is_optimal = 'Yes' if window_size == optimal_window_size else 'No'

            models_df = entry['models'].copy()
            
            models_df.reset_index(inplace=True)

            # Add new columns to models_df
            models_df['window_size'] = window_size
            models_df['feature_variance'] = feature_variance
            models_df['mean_scoring'] = mean_scoring
            models_df['std_scoring'] = std_scoring
            models_df['top_features'] = top_features_str
            models_df['is_optimal'] = is_optimal

            # Append models_df to csv_data
            csv_data.append(models_df)

        # Concatenate all DataFrames
        csv_results_df = pd.concat(csv_data, ignore_index=True)

        # Reorder columns to match desired CSV format
        base_columns = ['window_size', 'feature_variance', 'mean_scoring', 'std_scoring', 'top_features', 'is_optimal']
        model_columns = [col for col in csv_results_df.columns if col not in base_columns]
        csv_columns = base_columns + model_columns

        csv_results_df = csv_results_df[csv_columns]
        results['csv'] = {'model_results': csv_results_df}

    # Step 3: Process JSON Data
    if "json" in save_types:
        lazy_predict_results = {}

        for entry in organized_results:
            window_size = entry['window_size']
            models_df = entry['models']

            # Compute 'avg_score' based on primary_metric
            avg_score = models_df[primary_metric].mean()

            # Identify 'best_model' and 'best_score'
            best_score = models_df[primary_metric].max()
            best_model_idx = models_df[primary_metric].idxmax()
            best_model = best_model_idx

            # Build 'all_models' dictionary
            all_models = models_df.to_dict(orient='index')

            # Include additional data
            feature_variance = entry.get('feature_variance', None)
            regression_results = entry.get('regression_results', {})
            # Serialize DataFrames in regression_results
            for key, value in regression_results.items():
                if isinstance(value, pd.DataFrame):
                    regression_results[key] = value.to_dict(orient='list')

            lazy_predict_results[str(window_size)] = {
                'avg_score': avg_score,
                'best_model': best_model,
                'best_score': best_score,
                'feature_variance': feature_variance,
                'regression_results': regression_results,
                'all_models': all_models
            }

        # Get timestamp
        from datetime import datetime
        timestamp = datetime.now().isoformat()

        # Build the final JSON output
        json_output = {
            'task_params': task_params,
            'model_params': model_params,
            'lazy_predict_results': lazy_predict_results,
            'optimal_identifier': optimal_window_size,
            'optimal_avg_score': lazy_predict_results[str(optimal_window_size)]['avg_score'],
            'best_model': lazy_predict_results[str(optimal_window_size)]['best_model'],
            'best_score': lazy_predict_results[str(optimal_window_size)]['best_score'],
            'timestamp': timestamp
        }

        results['json'] = {'model_results': json_output}

    return results, optimal_window_size



    

def save_results(play_type: str, modeling_results: Dict[str, Any], optimal_window_size: int, output_params: Dict[str, Any]) -> None:
    """
    Save modeling results to appropriate directories.
    
    Args:
    play_type (str): Type of play (e.g., "fts", "hdp")
    modeling_results (Dict[str, Any]): The results from the modeling process.
    optimal_window_size (int): The determined optimal window size.
    output_params (Dict[str, Any]): Parameters for output configuration.
    """
    if play_type not in ['fts', 'hdp']:
        raise ValueError(f"Invalid play_type: {play_type}. Must be 'fts' or 'hdp'.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # base_dir = os.path.join("../../results", play_type, "find_optimal_window_size", timestamp)
    base_dir = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),  # Get the current directory of the script
    "../../../results", 
    play_type, 
    "find_optimal_window_size", 
    timestamp)
    # print(base_dir)

    # Identify result types present in modeling_results
    valid_result_types = ['csv', 'json', 'plots', 'models']
    result_types = [key for key in modeling_results if key in valid_result_types]
    
    if not result_types:
        raise ValueError("No valid result types found in modeling_results.")
    
    # Create necessary directories
    for result_type in result_types:
        os.makedirs(os.path.join(base_dir, result_type), exist_ok=True)
    
    # Save each result type
    for result_type in result_types:
        save_function = globals().get(f'save_{result_type}_results')
        if save_function:
            save_function(base_dir, modeling_results[result_type])
        else:
            logger.warning(f"No save function found for result type '{result_type}'")

    # Save optimal window size
    with open(os.path.join(base_dir, 'optimal_window_size.txt'), 'w') as f:
        f.write(f"Optimal Window Size: {optimal_window_size}")

    logger.info(f"Results saved to {base_dir}")

def save_csv_results(base_dir: str, csv_data: Dict[str, pd.DataFrame]) -> None:
    """Save CSV results."""
    csv_dir = os.path.join(base_dir, 'csv')
    for filename, df in csv_data.items():
        df.to_csv(os.path.join(csv_dir, f"{filename}.csv"), index=False)

def save_json_results(base_dir: str, json_data: Dict[str, Any]) -> None:
    """Save JSON results."""
    import json
    
    json_dir = os.path.join(base_dir, 'json')
    for filename, data in json_data.items():
        print(f"Saving JSON data for {filename}:")
        pprint.pprint(data)
        with open(os.path.join(json_dir, f"{filename}.json"), 'w') as f:
            try:
                json.dump(data, f, indent=4)
            except TypeError as e:
                print(f"Error serializing data: {e}")
                pprint.pprint(data)
    


def save_plots_results(base_dir: str, plot_data: Dict[str, Any]) -> None:
    """Save plot results."""
    import matplotlib.pyplot as plt
    plot_dir = os.path.join(base_dir, 'plots')
    for filename, fig in plot_data.items():
        fig.savefig(os.path.join(plot_dir, f"{filename}.png"))
        plt.close(fig)

def save_models_results(base_dir: str, model_data: Dict[str, Any]) -> None:
    """Save model results."""
    import joblib
    model_dir = os.path.join(base_dir, 'models')
    for filename, model in model_data.items():
        joblib.dump(model, os.path.join(model_dir, f"{filename}.joblib"))
        

# Sample code to execute the find_optimal_window_size function
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import logging
    # Assume that the find_optimal_window_size function is imported from the module
    # from task_workflows.tasks.find_optimal_window_size import find_optimal_window_size

    # Since we cannot actually import from the module in this environment,
    # let's define a placeholder for the find_optimal_window_size function.
    # Remove this placeholder when running in your environment where the function is available.

    # Set up logging
    logging.basicConfig(level=logging.INFO)


    # Generate sample DataFrames for different window sizes
    window_sizes = [5, 10, 15]
    provided_data = {}
    num_samples = 100
    num_features = 10

    # Instead of creating a single DataFrame, create one for each window size
    for window_size in window_sizes:
        # Generate synthetic features
        data = np.random.rand(num_samples, num_features)
        # Generate a binary target variable
        target = np.random.randint(0, 2, size=(num_samples, 1))
        # Create a DataFrame with feature columns
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(num_features)])
        # Add the target column
        df['target'] = target
        # Store the DataFrame with its corresponding window size
        provided_data[window_size] = df
    

    # Set up task parameters
    task_params = {
        'play_type': 'fts',  # Since we're ignoring 'hdp'
        'window_sizes': window_sizes,
        'target_column': 'target',
        'feature_set': [f'feature_{i}' for i in range(num_features)],
    }

    # Data parameters (empty since we're providing the data directly)
    data_params = {}

    # Preprocessing parameters
    preprocess_params = {
        'remove_duplicates': True,
        'missing_value_strategy': 'drop',  # Options: 'drop', 'mean_imputation', etc.
        'normalize_columns': True,
        'type_conversion_map': None,
        'scaling_params': None,
        'encoding_strategy': 'one_hot',  # Options: 'one_hot', 'label_encoding', etc.
    }

    # Model parameters for binary classification
    model_params = {
        'model_type': 'classification',  # Since the target is binary (0 or 1)
    }

    # Output parameters specifying how to save results
    output_params = {
        'save_type': ['csv', 'json'],  # Save results as CSV and JSON
    }

    # Execute the function with the provided parameters and data
    optimal_window_size, processed_results = find_optimal_window_size(
        task_params=task_params,
        data_params=data_params,
        preprocess_params=preprocess_params,
        model_params=model_params,
        output_params=output_params,
        data_input_mode='dataframe',
        provided_data=provided_data
    )

    # Print the outputs
    print(f"Optimal Window Size: {optimal_window_size}")
    print("Processed Results:")
    # print(processed_results)
    # Print only the JSON part of processed_results to avoid printing large DataFrames
    if 'json' in processed_results:
        pprint.pprint(processed_results['json'])

    # Optionally, you can print a message about CSV results without printing the DataFrame itself
    if 'csv' in processed_results:
        print("CSV results saved. DataFrame content is too large to print.")
