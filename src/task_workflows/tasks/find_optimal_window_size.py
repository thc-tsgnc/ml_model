# src/task_workflows/tasks/find_optimal_window_size.py

from data.fetcher.fts_data_fetcher import get_fts_data
from task_workflows.task_registry import register_task
from data.processing.data_cleaner import clean_data_list, process_data_entries
from typing import List, Dict, Any, Tuple
import pandas as pd
from modeling.lazy_predict_modeling import run_lazy_predict_modeling
from data.processing.eda_process import perform_eda_regression, calculate_feature_variance
import os


def analyze_window_sizes(cleaned_data_list, model_type):
    
    results = {}
    for entry in cleaned_data_list:
        window_size = entry['window_size']
        df = entry['data']
        target_column = entry['params']['target_column']
        
        feature_variance = calculate_feature_variance(df, target_column)
        regression_results = perform_eda_regression(df, target_column, model_type)
        
        results[window_size] = {
            'feature_variance': feature_variance,
            'regression_results': regression_results
        }
    return results
    
def select_optimal_window_size(results, model_type):
    if isinstance(results, dict) and 'lazy_predict_results' in results:
        # LazyPredict results
        return results['optimal_identifier']
    else:
        # Standard approach results
        metric = 'mean_r2' if model_type == 'regression' else 'mean_roc_auc'
        return max(results, key=lambda x: results[x]['regression_results'][f'{model_type}_regression'][metric] - results[x]['feature_variance'])


def optimize_window_size(cleaned_data_list: List[Dict[str, Any]], regression_type: str) -> Tuple[Dict, int, pd.DataFrame]:
    print(f"Optimizing window size for regression type: {regression_type}")
    results = analyze_window_sizes(cleaned_data_list, regression_type)
    optimal_window_size = select_optimal_window_size(results, regression_type)
    optimal_df = get_optimal_dataframe(cleaned_data_list, optimal_window_size)
    return results, optimal_window_size, optimal_df

def get_optimal_dataframe(cleaned_data_list: List[Dict[str, Any]], optimal_window_size: int) -> pd.DataFrame:
    return next(entry['data'] for entry in cleaned_data_list if entry['window_size'] == optimal_window_size)

def save_model_metrics(lazy_predict_results: Dict[str, Dict[str, pd.DataFrame]], output_file: str):
    """
    Save the model metrics from lazy predict results to a CSV file, including the model name.
    
    :param lazy_predict_results: Dictionary containing lazy predict results for each window size
    :param output_file: Path to save the CSV file
    """
    
    print(f"Saving model metrics to {output_file}")
    all_metrics = []
    for window_size, result in lazy_predict_results.items():
        metrics = result['models'].reset_index()  # Reset index to make 'Model' a column
        metrics.rename(columns={'index': 'Model'}, inplace=True)  # Rename 'index' to 'Model'
        metrics['window_size'] = window_size
        all_metrics.append(metrics)
    
    combined_metrics = pd.concat(all_metrics, ignore_index=True)
    combined_metrics.to_csv(output_file, index=False)
    print(f"Model metrics saved successfully to {output_file}")

    
def save_predictions_by_window_size(lazy_predict_results: Dict[str, Dict[str, pd.DataFrame]], output_dir: str):
    """
    Save predictions from all algorithms in a single file per window size.
    
    :param lazy_predict_results: Dictionary containing lazy predict results for each window size
    :param output_dir: Directory to save the prediction CSV files
    """
    os.makedirs(output_dir, exist_ok=True)
    for window_size, result in lazy_predict_results.items():
        output_file = os.path.join(output_dir, f"window_size_{window_size}_predictions.csv")
        print(f"Saving predictions for window size {window_size} to {output_file}")
        result['predictions'].to_csv(output_file, index=False)
        print(f"Predictions for window size {window_size} saved successfully")


def save_results_to_csv(results: Dict[int, Dict[str, Any]], optimal_window_size: int, data_params: dict, task_params: dict, model_params: dict, output_file: str, regression_type: str):
    """Save the results of the window size optimization process to a CSV file."""
    print(f"Saving results to CSV file: {output_file}")
    rows = []
    for window_size, result in results.items():
        row = {
            'window_size': window_size,
            'feature_variance': result['feature_variance'],
            'regression_metric': result['regression_results'][f"{regression_type}_regression"]['mean_roc_auc' if regression_type == 'logistic' else 'mean_r2'],
            'is_optimal': window_size == optimal_window_size
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df['data_params'] = str(data_params)
    df['task_params'] = str(task_params)
    df['model_params'] = str(model_params)
    df['regression_type'] = regression_type
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

@register_task("find_optimal_window_size")
def find_optimal_window_size(play_type: str, data_params: dict, model_params: dict, feature_params: list, pipeline_params: list, configuration: dict, query_params: dict, process_params: dict, output_params: dict, data_input_mode: str = "database", provided_data: pd.DataFrame = None):
    """
    Main function to find the optimal window size for different play types.
    
    :param play_type: Type of play (e.g., "fts", "hdp")
    :param data_params: Data parameters for fetching data
    :param model_params: Parameters for the model
    :param feature_params: List of feature parameters
    :param pipeline_params: List of pipeline parameters
    :param configuration: Configuration dictionary
    :param query_params: Query parameters dictionary
    :param process_params: Process parameters dictionary
    :param output_params: Output parameters dictionary
    :param data_input_mode: Mode of data input ("database" or "dataframe")
    :param provided_data: DataFrame provided directly for processing (used when data_input_mode is "dataframe")
    :return: Results and optimal window size
    """
    if play_type == "fts" and data_input_mode == "database":
        # Existing functionality for database input
        return find_optimal_window_size_fts(data_params, model_params, feature_params, pipeline_params, configuration, query_params, process_params, output_params)
    elif play_type == "fts" and data_input_mode == "dataframe":
        if provided_data is None:
            raise ValueError("Provided DataFrame is required when data_input_mode is set to 'dataframe'.")
        # New functionality for DataFrame input
        return find_optimal_window_size_from_dataframe(provided_data, query_params['window_sizes'], configuration['target_column'], model_params, feature_params, pipeline_params, configuration, output_params)
    elif play_type == "hdp":
        # Existing functionality for HDP input
        return find_optimal_window_size_hdp(data_params, model_params, feature_params, pipeline_params, configuration, query_params, process_params, output_params)
    else:
        raise ValueError(f"Unsupported play type or data input mode: {play_type} with mode {data_input_mode}")


def generate_metrics_filename(configuration: dict, data_params: dict, model_params: dict) -> str:
    model_type = configuration['model_type']
    model_key = configuration.get('model_key', 'standard')
    target_column = configuration['target_column']
    start_season = data_params['conditions']['season_year_start']
    end_season = data_params['conditions']['season_year_end']
    
    filename = f"{model_type}_{model_key}_{target_column}_s{start_season}_e{end_season}_metrics.csv"
    return filename

def process_optimal_window_size(cleaned_data_list, model_params, feature_params, pipeline_params, configuration, output_params):
    """
    Process the data to find the optimal window size.
    
    :param cleaned_data_list: List of cleaned DataFrame entries for each window size
    :param model_params: Parameters for the model
    :param feature_params: List of feature parameters
    :param pipeline_params: List of pipeline parameters
    :param configuration: Configuration dictionary
    :param output_params: Output parameters dictionary
    :return: Results and optimal window size
    """
    # Determine which modeling approach to use based on configuration
    model_type = configuration['model_type']
    model_key = configuration['model_key']

    # Use the appropriate modeling approach
    if model_type == 'classification' and model_key == 'lazy_predict':
        print("Using LazyPredict for classification")
        results = run_lazy_predict_modeling(cleaned_data_list, configuration, model_params)
    else:
        print(f"Using standard approach for {model_type}")
        results = analyze_window_sizes(cleaned_data_list, model_type)

    # Select optimal window size
    optimal_window_size = select_optimal_window_size(results, model_type)

    # Save results
    save_results(results, optimal_window_size, output_params)

    return results, optimal_window_size


def find_optimal_window_size_fts(data_params, model_params, feature_params, pipeline_params, configuration, query_params, process_params, output_params):
    print("Handling FTS play type")
    print("Running find_optimal_window_size task")
    print("Configuration:", configuration)
    
    # Extract necessary parameters
    conditions = data_params['conditions']
    target_column = configuration['target_column']
    window_sizes = query_params.get('window_sizes', [-1, 30])
    
    data_list = []
    for window_size in window_sizes:
        print(f"\nRetrieving data for window size: {window_size}")
        
        df = get_fts_data(
            start_season=conditions['season_year_start'],
            end_season=conditions['season_year_end'],
            is_feat=conditions['is_feat'],
            player_avg_type=conditions['player_avg_type'],
            player_window_size=window_size,
            team_avg_type=conditions['team_avg_type'],
            team_window_size=window_size,
            team_data_type=conditions['team_data_type'],
            lg_avg_type=conditions['lg_avg_type'],
            lg_window_size=window_size,
            excluded_columns=[],
            feat_columns=feature_params,
            target_column=target_column
        )

        if df is not None and not df.empty:
            print(f"Successfully retrieved data for window size {window_size}.")
            data_entry = {
                'window_size': window_size,
                'data': df,
                'params': {
                    'target_column': target_column,
                    'window_size': window_size,
                }
            }
            data_list.append(data_entry)
        else:
            print(f"Failed to retrieve data for window size {window_size} or the dataframe is empty.")

    # Clean the data
    
    cleaned_data_list = process_data_entries(
    data_list,
    target_column='team1_tip',
    remove_duplicates=True,
    missing_value_strategy={'A': 'mean', 'B': 'most_frequent'},  # Example strategy
    normalize_columns=True,
    scaling_params={'columns_to_scale': ['numeric_column1', 'numeric_column2'], 'exclude_columns': ['target_column']},
    encoding_strategy='one_hot'
)

    # Process the data to find optimal window size
    return process_optimal_window_size(cleaned_data_list, model_params, feature_params, pipeline_params, configuration, output_params)

def find_optimal_window_size_from_dataframe(dataframe, window_sizes, target_column, model_params, feature_params, pipeline_params, configuration, output_params):
    """
    Find the optimal window size using a provided DataFrame instead of connecting to a database.
    
    :param dataframe: The input DataFrame containing the data
    :param window_sizes: List of window sizes to evaluate
    :param target_column: Target column for regression or classification
    :param model_params: Parameters for the model
    :param feature_params: List of feature parameters
    :param pipeline_params: List of pipeline parameters
    :param configuration: Configuration dictionary
    :param output_params: Output parameters dictionary
    :return: Results and optimal window size
    """
    print("Running find_optimal_window_size from DataFrame")
    
    data_list = []
    for window_size in window_sizes:
        print(f"\nProcessing data for window size: {window_size}")

        # Filter or process the DataFrame according to the window size
        # Here, you might apply a rolling window or other operations based on the window size
        df = dataframe.copy()  # Example: simply copy DataFrame; adjust as needed

        if df is not None and not df.empty:
            print(f"Successfully processed data for window size {window_size}.")
            data_entry = {
                'window_size': window_size,
                'data': df,
                'params': {
                    'target_column': target_column,
                    'window_size': window_size,
                }
            }
            data_list.append(data_entry)
        else:
            print(f"Failed to process data for window size {window_size} or the dataframe is empty.")

    # Dynamically determine numeric columns to scale
    numeric_columns = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    # Ensure target_column is excluded from scaling
    numeric_columns_to_scale = [col for col in numeric_columns if col != target_column]

    # Clean the data using dynamically determined columns
    cleaned_data_list = process_data_entries(
        data_list,
        target_column='team1_tip',
        remove_duplicates=True,
        missing_value_strategy='mean',  # Example strategy
        normalize_columns=True,
        scaling_params={'columns_to_scale': numeric_columns_to_scale, 'exclude_columns': [target_column]},
        encoding_strategy='one_hot'
    )

    # Process the data to find the optimal window size
    return process_optimal_window_size(cleaned_data_list, model_params, feature_params, pipeline_params, configuration, output_params)


def find_optimal_window_size_hdp(data_params: dict, task_params: dict, model_params: dict):
    print("Handling BOX play type")
    print("Running find_optimal_window_size task")
    print("Data Params:", data_params)
    print("Task Params:", task_params)
    print("Model Params:", model_params)

def run_process_for_fts_window_size(datalist):
    results = []
    """
    for data in datalist:
    
       
        cleaned_data = clean_data(data)
        cv_results = cross_validate(cleaned_data)
        lr_results = run_logistic_regression(cleaned_data)
        stats_results = run_stats_test(cleaned_data)
        other_results = run_other_experiments(cleaned_data)
        
        results.append({
            'window_size': data['window_size'],
            'cv_results': cv_results,
            'lr_results': lr_results,
            'stats_results': stats_results,
            'other_results': other_results,
            'params': data['params']
        })
        """

def save_results(results, optimal_window_size, output_params):
    # Implement saving logic based on the structure of results and output_params
    pass


if __name__ == "__main__":
    import numpy as np
    import pandas as pd
    import os
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest
    from sklearn.ensemble import RandomForestClassifier
    from lazypredict.Supervised import LazyClassifier
    
    # Function to generate sample data
    def generate_sample_data(feature_names, target_name, n_samples=200):
        """
        Generate sample DataFrame for testing based on provided feature names.
        
        :param feature_names: List of feature names to generate random data for.
        :param target_name: Name of the target column.
        :param n_samples: Number of samples to generate.
        :return: A pandas DataFrame with the generated features and target column.
        """
        np.random.seed(42)
        data = {}
        for feature in feature_names:
            data[feature] = np.random.rand(n_samples)  # Random float data for features
        
        # Random binary classification target
        data[target_name] = np.random.randint(0, 2, size=n_samples)
        df = pd.DataFrame(data)
        
        return df

    # Configuration matching the provided setup for 'fts'
    feature_set = [
        "t1_player_tip_avg_overall",
        "t1_player_tip_avg_home",
        "t2_player_tip_avg_overall",
        "t2_player_tip_avg_away",
        "t1_team_fps_avg",
        "t1_team_fts_avg",
        "lg_t1_tip1_avg",
        "lg_t1_tip0_avg"
    ]
    
    # Generate sample DataFrame based on the feature set
    dataframe = generate_sample_data(feature_set, 'team1_tip', n_samples=200)

    # Task-specific parameters
    data_params = {
        'conditions': {
            'season_year_start': "201617",
            'season_year_end': "202324",
            'is_feat': "Yes",
            'player_avg_type': "SMA",
            'team_avg_type': "SMA",
            'team_data_type': "Overall",
            'lg_avg_type': "SMA"
        }
    }

    model_params = {
        'model_type': 'classification',  # Set to 'classification' for classification tasks
        'hyperparameters': {
            'random_state': 42  # Ensure reproducibility
        }
    }

    # Configurations that will be passed to the `find_optimal_window_size` function
    configuration = {
        'data_key': 'fts_basic',
        'feature_set': 'fts_set1',
        'target_column': 'team1_tip',
        'model_type': 'classification',  # Model type for classification
        'model_key': 'lazy_predict'  # Use LazyClassifier
    }

    query_params = {
        'window_sizes': [-1, 30]  # Example window sizes to be tested
    }

    # Output and process parameters
    process_params = {
        "max_iterations": 100,
        "convergence_threshold": 0.001
    }

    output_params = {
        'results_dir': './results/fts/window_size_optimization/',
        'model_save_dir': './models/fts/window_size_optimization/',
        'plots_dir': './plots/fts/window_size_optimization/'
    }

    # Ensure directories exist
    os.makedirs(output_params['results_dir'], exist_ok=True)
    os.makedirs(output_params['model_save_dir'], exist_ok=True)
    os.makedirs(output_params['plots_dir'], exist_ok=True)

    # Execute the `find_optimal_window_size` function using DataFrame input mode
    results, optimal_window_size = find_optimal_window_size(
        play_type="fts",
        data_params=data_params,
        model_params=model_params,
        feature_params=feature_set,
        pipeline_params=[],
        configuration=configuration,
        query_params=query_params,
        process_params=process_params,
        output_params=output_params,
        data_input_mode='dataframe',  # Use the 'dataframe' mode
        provided_data=dataframe  # Provide the DataFrame directly
    )

    # Print results and optimal window size
    print(f"\nResults using DataFrame input: {results}")
    print(f"Optimal Window Size using DataFrame input: {optimal_window_size}")
