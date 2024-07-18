from typing import Dict, Callable, Any

task_registry: Dict[str, Callable] = {}

def register_task(task_name: str):
    def decorator(func: Callable):
        task_registry[task_name] = func
        return func
    return decorator

def get_task_function(task_name: str) -> Callable[[str, Dict[str, Any], Dict[str, Any]], None]:
    return task_registry.get(task_name)

# Import all task files here to ensure decorators are executed
from task_workflows.tasks import find_optimal_window_size
# Add other task imports as needed