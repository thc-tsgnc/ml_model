# src/task_workflows/task_decorator.py


from typing import Callable
from task_workflows.task_runner import task_runner

def register_task(name: str) -> Callable:
    """
    Decorator to register a task with the TaskRunner.
    """
    def decorator(func: Callable) -> Callable:
        task_runner.register_task(name, func)
        return func
    return decorator
