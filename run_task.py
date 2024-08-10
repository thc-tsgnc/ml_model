import sys
import os

# Ensure the correct directory is included in the sys.path for the VSCode environment
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(project_root)

from utils.config_loader import load_and_validate_configs, load_playtype_config
from task_workflows.task_runner import TaskRunner

# Check if a configuration file path is provided as a command-line argument
if len(sys.argv) > 1:
    task_config_path = sys.argv[1]
else:
    # Default configuration file path for VSCode "Run Python File"
    task_config_path = "config/fts_config.json"

# Ensure the data_config path is also included
data_config_path = "config/data_config.json"

# Load and validate the configuration files
task_config, data_config = load_and_validate_configs(task_config_path, data_config_path)
playtype_config_path = "config/playtype_config.json"
playtype_config = load_playtype_config(playtype_config_path)
# Initialize the TaskRunner with both configurations
task_runner = TaskRunner(task_config, data_config, playtype_config)
# Execute tasks
# task_runner.run_tasks_from_config()
task_runner.run_tasks_from_playtype_config()
