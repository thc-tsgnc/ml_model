# src/task_workflows/task_runner.py

from typing import Callable, Dict, List, Optional, Any
import logging
import importlib
from task_workflows.task_registry import get_task_function

class TaskRunner:
    def __init__(self, task_config: Dict[str, Any], data_config: Dict[str, Any]):
        self.task_config = task_config
        self.data_config = data_config
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
