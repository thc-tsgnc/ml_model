import json
import os

def load_config(file_path):
    """
    Load a JSON configuration file.
    :param file_path: Path to the JSON configuration file.
    :return: Parsed JSON data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    with open(file_path, 'r') as file:
        config = json.load(file)
    
    return config

def validate_config(task_config, data_config):
    """
    Validate the task configuration against the data configuration.
    :param task_config: The task configuration data.
    :param data_config: The data configuration data.
    :return: None, raises ValueError if validation fails.
    """
    play_type_task = task_config.get("play_type", {})
    play_type_data = data_config.get("play_type", {})

    for play_type, tasks in play_type_task.items():
        if play_type not in play_type_data:
            raise ValueError(f"Play type '{play_type}' in task config does not exist in data config.")

        for task in tasks["tasks"]:
            data_key = task["data_key"]
            if data_key not in play_type_data[play_type]:
                raise ValueError(f"Data key '{data_key}' in task config does not exist in data config under play type '{play_type}'.")

    print("Validation successful.")

def load_and_validate_configs(task_config_path, data_config_path):
    """
    Load and validate both task and data configurations.
    :param task_config_path: Path to the task configuration file.
    :param data_config_path: Path to the data configuration file.
    :return: Tuple of (task_config, data_config)
    """
    task_config = load_config(task_config_path)
    data_config = load_config(data_config_path)

    # validate_config(task_config, data_config)

    return task_config, data_config

def load_playtype_config(playtype_config_path):
    """
    Load the playtype configuration file.
    :param playtype_config_path: Path to the playtype configuration file.
    :return: Parsed playtype configuration data.
    """
    if not os.path.exists(playtype_config_path):
        raise FileNotFoundError(f"Playtype configuration file not found: {playtype_config_path}")

    with open(playtype_config_path, 'r') as file:
        playtype_config = json.load(file)
    
    return playtype_config