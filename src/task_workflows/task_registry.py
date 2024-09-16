# src/task_workflows/task_registry.py

from typing import Dict, Callable, Any, List

# Dictionary to store registered tasks
task_registry: Dict[str, Callable] = {}

def register_task(task_name: str):
    """
    Decorator to register a task function under a specific task name.

    Args:
    task_name (str): The name to register the task function under.
    
    Returns:
    Callable: The decorator function.
    """
    def decorator(func: Callable):
        task_registry[task_name] = func
        return func
    return decorator

def get_task_function(task_name: str) -> Callable[[Any, Dict[str, Any], Dict[str, Any]], None]:
    """
    Retrieve a registered task function by its name.

    Args:
    task_name (str): The name of the task function to retrieve.

    Returns:
    Callable: The registered task function or None if not found.
    """
    return task_registry.get(task_name)

def get_all_task_names() -> List[str]:
    """
    Return a list of all registered task names.

    This function returns the names of all tasks that have been registered
    using the @register_task decorator.

    Returns:
    List[str]: A list of registered task names.
    """
    return list(task_registry.keys())

# Import all task modules here to ensure that tasks are registered when this file is imported.
# Add imports for all task modules that use @register_task to ensure they are registered.
from task_workflows.tasks import find_optimal_window_size
from task_workflows.tasks.eda_feature_target_analysis import eda_feature_target_analysis
from task_workflows.tasks.eda_binning_analysis import eda_binning_analysis
from task_workflows.tasks.eda_clustering_analysis import eda_clustering_analysis
from task_workflows.tasks.eda_statistical_analysis import eda_statistical_analysis
from task_workflows.tasks.clustering_workflow import run_clustering_iterations
