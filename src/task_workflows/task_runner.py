# src/task_workflows/task_runner.py

from typing import Callable, Dict, List, Optional, Any
import logging
import importlib
from task_workflows.task_registry import get_task_function

class TaskRunner:
    def __init__(self, task_config: Dict[str, Any], data_config: Dict[str, Any], playtype_config: Optional[Dict[str, Any]] = None):
        self.task_config = task_config
        self.data_config = data_config
        self.playtype_config = playtype_config
        self.logger = logging.getLogger(__name__)

    def register_task(self, task_name: str, task_function: Callable) -> None:
        """
        Register a task with a given name and function.
        """
        self.task_registry[task_name] = task_function

    def run_tasks_from_json(self) -> None:
        """
        Run tasks defined in the task_config, using data from data_config.
        """
        
        play_type = self.task_config.get("play_type")
        tasks = self.task_config.get("tasks", [])
        
        for task in tasks:
            task_name = task.get("task_name")
            
            if not task.get("enabled", True):
                self.logger.info(f"Task '{task_name}' is disabled. Skipping.")
                continue

            task_params = task.get("task_params", {})
            data_key = task.get("data_key")

            if data_key not in self.data_config:
                self.logger.warning(f"Data key '{data_key}' not found in data_config for task '{task_name}'. Skipping.")
                continue

            data_params = self.data_config[data_key]

            try:
                # self._execute_task(task_name, play_type, data_params, task_params)
                
                self.logger.info(f"Task '{task_name}' executed successfully.")
            except Exception as e:
                self.logger.error(f"Error executing task '{task_name}': {str(e)}")

    
    def run_tasks_from_config(self):
        play_types = self.task_config.get("play_type", {})
        for play_type, play_type_config in play_types.items():
            tasks = play_type_config.get("tasks", [])

            for task in tasks:
                task_name = task["task_name"]
                if not task.get("enabled", True):
                    print(f"Task '{task_name}' is disabled. Skipping.")
                    continue

                data_key = task["data_key"]
                data_params = self.get_data_params(play_type, data_key)
                task_params = task.get("task_params", {})

                try:
                    # from task_registry import get_task_function
                    task_function = get_task_function(task_name)
                    if task_function:
                        task_function(play_type, data_params, task_params)
                        print(f"Task '{task_name}' executed successfully.")
                    else:
                        print(f"Task '{task_name}' not found in registry.")
                except Exception as e:
                    print(f"Error executing task '{task_name}': {str(e)}")


    def get_data_params(self, play_type: str, data_key: str) -> Dict[str, Any]:
        play_type_data = self.data_config.get("play_type", {}).get(play_type, {})
        return play_type_data.get(data_key, {})


    def run_task_by_name(self, task_name: str, play_type: str, data_params: Dict[str, Any], task_params: Dict[str, Any]) -> None:
        """
        Run a specific task by name with given parameters.
        """
        self._execute_task(task_name, play_type, data_params, task_params)

    def _execute_task(self, task_name: str, play_type: str, data_params: Dict[str, Any], task_params: Dict[str, Any]) -> None:
        """
        Execute a task by name with the given parameters.
        """
        if task_name in self.task_registry:
            self.task_registry[task_name](play_type, data_params, task_params)
        else:
            raise ValueError(f"Unsupported task: {task_name}")
        
    def get_playtype_data_params(self, play_type: str, data_key: str) -> Dict[str, Any]:
        """
        Get data parameters for a specific play type and data key.
        """
        return self.playtype_config[play_type]["data_sources"].get(data_key, {})


    def get_playtype_model_params(self, play_type: str, model_type: str, model_key: str) -> Dict[str, Any]:
        """
        Get model parameters for a specific play type, model type, and model key.
        """
        return self.playtype_config[play_type]["model_configs"].get(model_type, {}).get(model_key, {})

    def get_playtype_feature_params(self, play_type: str, feature_set: str) -> List[str]:
        """
        Get feature parameters for a specific play type and feature set.
        """
        return self.playtype_config[play_type]["feature_sets"].get(feature_set, [])

    def get_playtype_pipeline_params(self, play_type: str) -> List[Dict[str, Any]]:
        """
        Get pipeline parameters for a specific play type.
        """
        return self.playtype_config[play_type]["pipelines"].get("default", [])

    def parse_and_run_tasks_from_playtype_config(self):
        """
        Run tasks defined in the playtype_config.
        """
        if not self.playtype_config:
            self.logger.error("Playtype config is not provided.")
            return

        for play_type, config in self.playtype_config.items():
            tasks = config.get("tasks", {})
            for task_name, task_config in tasks.items():
                if not task_config.get("enabled", True):
                    self.logger.info(f"Task '{task_name}' for play type '{play_type}' is disabled. Skipping.")
                    continue

                for configuration in task_config.get("configurations", []):
                    data_params = self.get_playtype_data_params(play_type, configuration)
                    model_params = self.get_playtype_model_params(play_type, configuration)
                    feature_params = self.get_playtype_feature_params(play_type, configuration)
                    pipeline_params = self.get_playtype_pipeline_params(play_type)

                    try:
                        task_function = get_task_function(task_name)
                        if task_function:
                            task_function(play_type, data_params, model_params, feature_params, pipeline_params, configuration)
                            self.logger.info(f"Task '{task_name}' for play type '{play_type}' executed successfully.")
                        else:
                            self.logger.warning(f"Task '{task_name}' not found in registry.")
                    except Exception as e:
                        self.logger.error(f"Error executing task '{task_name}' for play type '{play_type}': {str(e)}")    
                        
    def parse_playtype_config(self):
        """
        Parse the playtype_config and return a structured dictionary of tasks and their parameters.
        """
        parsed_config = {}
        if not self.playtype_config:
            self.logger.error("Playtype config is not provided.")
            return parsed_config

        for play_type, config in self.playtype_config.items():
            parsed_config[play_type] = {}
            tasks = config.get("tasks", {})
            for task_name, task_config in tasks.items():
                if not task_config.get("enabled", True):
                    continue

                parsed_config[play_type][task_name] = []
                for configuration in task_config.get("configurations", []):
                    task_params = {
                        "data_params": self.get_playtype_data_params(play_type, configuration.get("data_key")),
                        "model_params": self.get_playtype_model_params(play_type, configuration.get("model_type"), configuration.get("model_key")),
                        "feature_params": self.get_playtype_feature_params(play_type, configuration.get("feature_set")),
                        "pipeline_params": self.get_playtype_pipeline_params(play_type),
                        "configuration": configuration
                    }
                    parsed_config[play_type][task_name].append(task_params)

                # Add task-specific parameters
                parsed_config[play_type][task_name].append({
                    "query_params": task_config.get("query_params", {}),
                    "process_params": task_config.get("process_params", {}),
                    "output": task_config.get("output", {})
                })

        return parsed_config

    def show_and_validate_parameters(self, parsed_config):
        """
        Show and validate the parameters for each task in the parsed config.
        """
        for play_type, tasks in parsed_config.items():
            print(f"Play Type: {play_type}")
            for task_name, task_configs in tasks.items():
                print(f"  Task: {task_name}")
                for i, config in enumerate(task_configs[:-1]):  # Exclude the last item which contains task-specific params
                    print(f"    Configuration {i + 1}:")
                    for param_type, params in config.items():
                        print(f"      {param_type}:")
                        if isinstance(params, dict):
                            for key, value in params.items():
                                if isinstance(value, dict):
                                    print(f"        {key}:")
                                    for sub_key, sub_value in value.items():
                                        print(f"          {sub_key}: {sub_value}")
                                else:
                                    print(f"        {key}: {value}")
                        elif isinstance(params, list):
                            for item in params:
                                print(f"        - {item}")
                        else:
                            print(f"        {params}")
                
                # Print task-specific parameters
                task_specific_params = task_configs[-1]
                print("    Task-specific parameters:")
                for param_type, params in task_specific_params.items():
                    print(f"      {param_type}:")
                    if isinstance(params, dict):
                        for key, value in params.items():
                            print(f"        {key}: {value}")
                    else:
                        print(f"        {params}")
    def execute_tasks(self, parsed_config):
        """
        Execute the tasks based on the parsed config.
        """
        for play_type, tasks in parsed_config.items():
            for task_name, task_configs in tasks.items():
                # Extract task-specific parameters
                task_specific_params = task_configs[-1]
                configurations = task_configs[:-1]  # Exclude the last item which contains task-specific params

                for config in configurations:
                    try:
                        task_function = get_task_function(task_name)
                        if task_function:
                            task_function(
                                play_type=play_type,
                                data_params=config['data_params'],
                                model_params=config['model_params'],
                                feature_params=config['feature_params'],
                                pipeline_params=config['pipeline_params'],
                                configuration=config['configuration'],
                                query_params=task_specific_params['query_params'],
                                process_params=task_specific_params['process_params'],
                                output_params=task_specific_params['output']
                            )
                            self.logger.info(f"Task '{task_name}' for play type '{play_type}' executed successfully.")
                        else:
                            self.logger.warning(f"Task '{task_name}' not found in registry.")
                    except Exception as e:
                        self.logger.error(f"Error executing task '{task_name}' for play type '{play_type}': {str(e)}")

    def run_tasks_from_playtype_config(self):
        """
        Run tasks defined in the playtype_config.
        """
        parsed_config = self.parse_playtype_config()
        self.show_and_validate_parameters(parsed_config)
        self.execute_tasks(parsed_config)
