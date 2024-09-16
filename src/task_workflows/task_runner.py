# src/task_workflows/task_runner.py

import pandas as pd
from typing import Dict, List, Optional, Any, Callable
import logging
from task_workflows.task_registry import get_task_function, get_all_task_names
import os
import json

"""
config_parser.py

This module is responsible for handling the parsing and preparation of input parameters from the `playtype_config` for the TaskRunner.

TODO:
1. **Load and Validate Configurations**:
   - Create a function to load configurations from a file, environment variable, or other sources as needed.
   - Implement a validation function to check if the configuration contains all required fields:
     - `data_sources`: Define the data sources available for the play type.
     - `feature_sets`: Define the feature sets to be used for tasks.
     - `model_configs`: Define model configurations, including model parameters and preprocessing steps.
     - `tasks`: Define the tasks to be executed, their configurations, and specific parameters.
     - `pipelines`: Define the sequence of steps (e.g., load data, preprocess, train model) to execute tasks.

2. **Parse Configuration for Data Sources**:
   - Implement a function to extract and prepare data source parameters based on the `data_key` specified in each task configuration.
   - Ensure it handles conditional parameters such as `season_year_start`, `season_year_end`, `player_avg_type`, etc.
   - Provide utility functions to fetch data based on these parameters.

3. **Parse Configuration for Feature Sets**:
   - Create a function to retrieve feature sets based on the `feature_set` key provided in the task configuration.
   - Include logic to handle different feature sets and their specific parameters.

4. **Parse Configuration for Model Configurations**:
   - Implement a function to retrieve model configurations based on the `model_key` and `model_type`.
   - Include preprocessing steps and evaluation metrics as defined in the `model_configs`.
   - Add support for multiple model types, such as classification and regression, and their respective configurations.

5. **Prepare Task-Specific Parameters**:
   - Implement a function to extract `query_params`, `process_params`, `output`, and other task-specific parameters needed for task execution.
   - Provide utility functions to merge these parameters with default configurations if necessary.

6. **Integrate with TaskRunner**:
   - Ensure that `TaskRunner` can use this module to retrieve all necessary parameters in a clean and modular way.
   - Refactor `TaskRunner` to delegate configuration parsing and parameter preparation to `config_parser`.

7. **Add Unit Tests**:
   - Write comprehensive unit tests for each function in this module.
   - Test cases should cover normal scenarios as well as edge cases, including missing or malformed configurations.

8. **Document the Module**:
   - Provide clear docstrings and usage examples for each function.
   - Document how to extend the configuration structure to support new data sources, feature sets, models, or tasks.

9. **Optimize and Refactor as Needed**:
   - After integration and testing, review the module for potential optimizations.
   - Refactor code to improve readability, maintainability, and adherence to clean code principles.
   
By following these steps, this module will help streamline the configuration management process, making it easier to run tasks and manage dependencies within the TaskRunner framework.
"""


class TaskRunner:
    def __init__(self, task_config: Optional[Dict[str, Any]] = None, data_config: Optional[Dict[str, Any]] = None, playtype_config: Optional[Dict[str, Any]] = None):
        self.task_config = task_config if task_config is not None else {}
        self.data_config = data_config if data_config is not None else {}
        self.playtype_config = playtype_config
        self.results = {}
        self.logger = logging.getLogger(__name__)
        self.task_registry = {}
        self._register_all_tasks()

        # Validate configs only if provided and contain specific keys
        if self.task_config and self.data_config:
            self._validate_configs()

    def _validate_configs(self):
        """Validate the structure and contents of the provided configurations."""
        if not isinstance(self.task_config, dict) or not isinstance(self.data_config, dict):
            raise ValueError("Both task_config and data_config must be dictionaries.")
        if 'play_type' not in self.task_config:
            raise ValueError("task_config must contain 'play_type'.")
        self.logger.info("Configurations validated successfully.")

    def _register_all_tasks(self):
        """Automatically register all available tasks from the task registry."""
        for task_name in get_all_task_names():
            task_function = get_task_function(task_name)
            if task_function:
                self.task_registry[task_name] = task_function
                self.logger.info(f"Task '{task_name}' registered.")
            else:
                self.logger.warning(f"Task function for '{task_name}' not found.")
    

    def validate_data(self, data: Any) -> bool:
        """
        Validate the input data to ensure it is suitable for the task.

        Args:
        data (Any): Data input for validation.

        Returns:
        bool: True if data is valid, False otherwise.
        """
        if isinstance(data, pd.DataFrame):
            if data.empty:
                self.logger.error("DataFrame is empty.")
                return False
            return True
        else:
            self.logger.error("Unsupported data type. Expected a DataFrame.")
            return False

    def run_task(self, task_name: str, data: Any, **kwargs) -> Any:
        """
        Run a single task with provided data and parameters.

        Args:
        task_name (str): Name of the task to run.
        data (Any): Data input for the task (must be a DataFrame).
        **kwargs: Additional parameters for the task.

        Returns:
        Any: Result of the task execution.

        Raises:
        ValueError: If the task is not found or if the data is invalid.
        """
        if task_name not in self.task_registry:
            raise ValueError(f"Task '{task_name}' not found in registry.")
        
        if not self.validate_data(data):
            raise ValueError("Invalid data provided for the task.")
        
        try:
            task_function = self.task_registry[task_name]
            self.logger.info(f"Executing task '{task_name}' with parameters: {kwargs}")
            result = task_function(data, **kwargs)
            self.logger.info(f"Task '{task_name}' executed successfully.")
            self.results[task_name] = result  # New: Store the result
            return result
        except Exception as e:
            self.logger.error(f"Error executing task '{task_name}': {str(e)}")
            raise

    def retrieve_parameters(self, param_type: str, play_type: str, **kwargs) -> Dict[str, Any]:
        """
        Retrieve various types of parameters.

        Args:
        param_type (str): Type of parameters to retrieve (e.g., 'data', 'model', 'feature').
        play_type (str): Type of play.
        **kwargs: Additional arguments needed for specific parameter types.

        Returns:
        Dict[str, Any]: Retrieved parameters.

        Raises:
        ValueError: If the parameter type is unknown or if required configurations are missing.
        """
        retrieval_functions = {
            'data': self._retrieve_data_params,
            'model': self._retrieve_model_params,
            'feature': self._retrieve_feature_params
        }

        if param_type not in retrieval_functions:
            raise ValueError(f"Unknown parameter type: {param_type}")

        return retrieval_functions[param_type](play_type, **kwargs)

    def _retrieve_data_params(self, play_type: str, **kwargs) -> Dict[str, Any]:
        data_key = kwargs.get('data_key')
        if not data_key:
            raise ValueError("'data_key' is required for retrieving data parameters.")
        return self.data_config.get("play_type", {}).get(play_type, {}).get(data_key, {})

    def _retrieve_model_params(self, play_type: str, **kwargs) -> Dict[str, Any]:
        model_type = kwargs.get('model_type')
        model_key = kwargs.get('model_key')
        if not model_type or not model_key:
            raise ValueError("Both 'model_type' and 'model_key' are required for retrieving model parameters.")
        return self.playtype_config[play_type]["model_configs"].get(model_type, {}).get(model_key, {})

    def _retrieve_feature_params(self, play_type: str, **kwargs) -> List[str]:
        feature_set = kwargs.get('feature_set')
        if not feature_set:
            raise ValueError("'feature_set' is required for retrieving feature parameters.")
        return self.playtype_config[play_type]["feature_sets"].get(feature_set, [])

    def merge_task_parameters(self, data_params: Dict[str, Any], task_params: Dict[str, Any], debug: bool = False) -> Dict[str, Any]:
        """
        Merge data and task parameters, with task parameters taking precedence.

        Args:
        data_params (Dict[str, Any]): Data-specific parameters.
        task_params (Dict[str, Any]): Task-specific parameters.
        debug (bool): If True, log parameter conflicts.

        Returns:
        Dict[str, Any]: Merged parameters.
        """
        merged = data_params.copy()
        conflicts = {}
        for key, value in task_params.items():
            if key in merged and merged[key] != value:
                conflicts[key] = (merged[key], value)
            merged[key] = value

        if debug and conflicts:
            self.logger.debug(f"Parameter conflicts detected: {conflicts}")

        return merged

    def run_tasks_from_config(self):
        """
        Run tasks defined in the configuration.

        This method iterates through the task configuration, retrieves necessary parameters,
        and executes each enabled task.
        """
        for play_type, play_type_config in self.task_config.get("play_type", {}).items():
            for task in play_type_config.get("tasks", []):
                if not task.get("enabled", True):
                    self.logger.info(f"Task '{task['task_name']}' for play_type '{play_type}' is disabled. Skipping.")
                    continue
                
                task_name = task["task_name"]
                data_key = task.get("data_key")
                if not data_key:
                    self.logger.warning(f"Missing 'data_key' for task '{task_name}' in play_type '{play_type}'. Skipping.")
                    continue

                try:
                    data_params = self.retrieve_parameters("data", play_type, data_key=data_key)
                    task_params = task.get("task_params", {})
                    combined_params = self.merge_task_parameters(data_params, task_params, debug=True)
                    
                    self.logger.info(f"Executing task '{task_name}' for play_type '{play_type}' with parameters: {combined_params}")
                    self.run_task(task_name, **combined_params)
                except Exception as e:
                    self.logger.error(f"Error executing task '{task_name}' for play_type '{play_type}': {str(e)}")

    def execute_tasks(self, parsed_config):
        """
        Execute tasks based on the parsed configuration.

        Args:
        parsed_config (Dict): Parsed configuration containing task details.

        This method executes tasks based on a parsed configuration, handling different
        parameter types and ensuring proper merging of configurations.

        Note: Task-specific parameters take precedence over other configuration parameters.
        """
        for play_type, tasks in parsed_config.items():
            for task_name, task_configs in tasks.items():
                task_specific_params = task_configs[-1]
                configurations = task_configs[:-1]

                for config in configurations:
                    try:
                        combined_params = {
                            **config.get('data_params', {}),
                            **config.get('model_params', {}),
                            'feature_params': config.get('feature_params', []),
                            'pipeline_params': config.get('pipeline_params', []),
                            **config.get('configuration', {}),
                        }
                        # Task-specific params override other params
                        final_params = self.merge_task_parameters(combined_params, task_specific_params, debug=True)
                        
                        self.logger.info(f"Executing task '{task_name}' for play_type '{play_type}' with parameters: {final_params}")
                        self.run_task(task_name, play_type=play_type, **final_params)
                    except Exception as e:
                        self.logger.error(f"Error executing task '{task_name}' for play_type '{play_type}': {str(e)}")

    def get_results(self) -> Dict[str, Any]:
        """Return the stored results for all tasks."""
        return self.results

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    import os
    import json
    from task_workflows.task_runner import TaskRunner

    # Create a sample DataFrame with a numeric target column and categorical features
    df = pd.DataFrame({
        'feature1': np.random.rand(1000),  # Numeric feature
        'feature2': np.random.randint(0, 100, 1000),  # Another numeric feature
        'feature3': np.random.choice(['A', 'B', 'C'], 1000),  # Categorical feature
        'feature4': np.random.choice(['Low', 'Medium', 'High'], 1000),  # Another categorical feature
        'target': np.random.randint(0, 3, 1000)  # Numeric target
    })

    # Initialize TaskRunner with minimal configuration
    runner = TaskRunner(task_config={}, data_config={})

    # Set up parameters for eda_binning_analysis
    binning_task_params = {
        'column': 'feature1',
        'target_column': 'target',
        'min_bins': 3,
        'max_bins': 10,
        'strategy': 'quantile',
        'output_dir': 'results/eda_binning'
    }

    # Set up parameters for eda_clustering_analysis
    clustering_task_params = {
        'features': ['feature1', 'feature2'],
        'algorithms': ['kmeans'],
        'param_grid': {
            'kmeans': {
                'n_clusters': [2, 3, 4],
                'random_state': [42]
            }
        },
        'output_dir': 'results/eda_clustering'
    }

    # Set up parameters for eda_feature_target_analysis
    feature_target_params = {
        'features': ['feature1', 'feature2', 'feature3'],
        'target': 'target',
        'output_dir': 'results/eda_feature_target'
    }

        # Set up parameters for eda_statistical_analysis
    statistical_analysis_params = {
        'columns': ['feature1', 'feature2', 'feature3', 'feature4'],
        'target_column': 'target',
        'target_type': 'continuous',  # Specify target type as a string
        'output_dir': 'results/eda_statistical_analysis',
        'p_threshold': 0.05,
        'normality_test': 'shapiro',
        'homogeneity_test': 'levene'
    }

    # Execute all tasks
    try:
        # Run eda_binning_analysis
        result_binning = runner.run_task('eda_binning_analysis', df, **binning_task_params)
        print("\nEDA Binning Analysis Results:")
        for column, analysis_result in result_binning.get('column_results', {}).items():
            if 'optimal_bins' in analysis_result:
                print(f"Column: {column}")
                print(f"Optimal number of bins: {analysis_result['optimal_bins']}")
                print(f"Report saved at: {analysis_result['report_path']}")
                print("Visualization paths:")
                for path in analysis_result.get('visualization_paths', []):
                    print(f"- {path}")
            else:
                print(f"No optimal bins found for column: {column}. Error: {analysis_result.get('error', 'Unknown error')}")

        # Run eda_clustering_analysis
        result_clustering = runner.run_task('eda_clustering_analysis', df, **clustering_task_params)
        print("\nEDA Clustering Analysis Results:")
        print(f"Results are saved in: {clustering_task_params['output_dir']}")

        # Run eda_feature_target_analysis
        result_feature_target = runner.run_task('eda_feature_target_analysis', df, **feature_target_params)
        print("\nEDA Feature-Target Analysis Results:")
        for feature, feature_results in result_feature_target.items():
            print(f"\nResults for feature: {feature}")
            for key, value in feature_results.items():
                if isinstance(value, dict):
                    print(f"  {key}:")
                    for subkey, subvalue in value.items():
                        print(f"    {subkey}: {subvalue}")
                elif isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")

        print("\nGenerated files for Feature-Target Analysis:")
        for filename in os.listdir(feature_target_params['output_dir']):
            print(f"  {filename}")

        # Run eda_statistical_analysis
        result_statistical = runner.run_task('eda_statistical_analysis', df, **statistical_analysis_params)
        print("\nEDA Statistical Analysis Results:")
        for column, result in result_statistical.items():
            print(f"\nColumn: {column}")
            print(f"Test Type: {result.get('test_type', 'N/A')}")
            print(f"P-value: {result.get('p_value', 'N/A')}")
            print(f"Interpretation: {result.get('interpretation', 'N/A')}")
            if 'visualizations' in result:
                print("Visualizations generated:")
                for vis_type, vis_path in result['visualizations'].items():
                    print(f"  - {vis_type}: {vis_path}")

    except Exception as e:
        print(f"An error occurred while running the tasks: {str(e)}")

    # Print all results
    all_results = runner.get_results()
    print("\nAll task results:")
    print(all_results)