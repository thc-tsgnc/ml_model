# src/modeling/hyperparameter_optimization.py

import optuna
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from typing import Dict, Any, Callable, Optional, Tuple, List
import pandas as pd
import numpy as np
from modeling.clustering_algorithms import run_kmeans, run_dbscan, run_gmm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Default parameter spaces for each algorithm
DEFAULT_PARAM_SPACES = {
    'kmeans': {
        'n_clusters': (2, 10)
    },
    'dbscan': {
        'eps': (0.05, 1.0),
        'min_samples': (2, 10)
    },
    'gmm': {
        'n_components': (2, 10)
    }
}

# Mapping of algorithm names to their respective functions
ALGORITHM_FUNCTIONS = {
    'kmeans': run_kmeans,
    'dbscan': run_dbscan,
    'gmm': run_gmm
}

# Available objective metrics with their functions and optimization directions
OBJECTIVE_METRICS = {
    'silhouette': {'function': silhouette_score, 'direction': 'maximize'},
    'calinski_harabasz': {'function': calinski_harabasz_score, 'direction': 'maximize'},
    'davies_bouldin': {'function': davies_bouldin_score, 'direction': 'minimize'}
}

def adjust_param_space(df: pd.DataFrame, algorithm_name: str, param_space: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dynamically adjust parameter ranges based on dataset characteristics.

    Args:
        df (pd.DataFrame): Input data for clustering.
        algorithm_name (str): Name of the clustering algorithm.
        param_space (Dict[str, Any]): Initial parameter space.

    Returns:
        Dict[str, Any]: Adjusted parameter space.
    """
    n_samples, n_features = df.shape
    adjusted_space = param_space.copy()

    if algorithm_name in ['kmeans', 'gmm']:
        param_name = 'n_clusters' if algorithm_name == 'kmeans' else 'n_components'
        max_clusters = min(int(np.sqrt(n_samples)), n_samples // 2)
        adjusted_space[param_name] = (2, max_clusters)
    elif algorithm_name == 'dbscan':
        # Adjust eps based on feature space
        from sklearn.neighbors import NearestNeighbors
        nn = NearestNeighbors(n_neighbors=2)
        nn.fit(df)
        distances, _ = nn.kneighbors(df)
        max_eps = np.percentile(distances[:, 1], 90)  # 90th percentile of nearest neighbor distances
        adjusted_space['eps'] = (0.01, max_eps)
        
        # Adjust min_samples based on dataset size
        if n_samples > 1000:
            adjusted_space['min_samples'] = (2, min(int(np.log(n_samples)), n_samples // 100))


    logger.info(f"Adjusted parameter space for {algorithm_name}: {adjusted_space}")
    return adjusted_space

def validate_param_space(algorithm_name: str, param_space: Dict[str, Any]) -> None:
    """
    Validate the parameter space for a given clustering algorithm.

    Args:
        algorithm_name (str): Name of the clustering algorithm.
        param_space (Dict[str, Any]): Parameter space to validate.

    Raises:
        ValueError: If the parameter space contains invalid values.

    This function checks if the parameters for each algorithm are within valid ranges.
    It's particularly important for parameters like 'eps' in DBSCAN or 'n_clusters' in KMeans.
    """
    logger.debug(f"Validating parameter space for {algorithm_name}")
    if algorithm_name == 'dbscan':
        if 'eps' in param_space and (param_space['eps'][0] <= 0 or param_space['eps'][1] <= 0):
            raise ValueError("eps values for DBSCAN must be positive.")
        if 'min_samples' in param_space and (param_space['min_samples'][0] < 2):
            raise ValueError("min_samples for DBSCAN must be greater than 1.")
    elif algorithm_name in ['kmeans', 'gmm']:
        param_name = 'n_clusters' if algorithm_name == 'kmeans' else 'n_components'
        if param_name in param_space and (param_space[param_name][0] < 2):
            raise ValueError(f"Number of {param_name} for {algorithm_name.upper()} must be at least 2.")
    logger.debug("Parameter space validation completed successfully")

def merge_param_spaces(default_space: Dict[str, Any], custom_space: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Merge default and custom parameter spaces.

    Args:
        default_space (Dict[str, Any]): Default parameter space.
        custom_space (Optional[Dict[str, Any]]): Custom parameter space to override defaults.

    Returns:
        Dict[str, Any]: Merged parameter space.

    This function combines the default parameter space with any custom parameters provided.
    It allows users to override default values or add new parameters as needed.
    """
    if custom_space is None:
        return default_space
    merged_space = default_space.copy()
    merged_space.update(custom_space)
    logger.debug(f"Merged parameter space: {merged_space}")
    return merged_space

def is_valid_clustering(
    labels: np.ndarray, 
    min_cluster_fraction: float = 0.05, 
    max_noise_fraction: float = 0.5, 
    max_cluster_fraction: float = 0.9
) -> bool:
    """
    Check if the clustering result is valid based on cluster sizes and noise.

    Args:
        labels (np.ndarray): Cluster labels.
        min_cluster_fraction (float): Minimum fraction of points that should be in the smallest cluster.
        max_noise_fraction (float): Maximum fraction of points that can be labeled as noise.
        max_cluster_fraction (float): Maximum fraction of points that can be in the largest cluster.

    Returns:
        bool: True if clustering is valid, False otherwise.
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    total_points = len(labels)
    
    if len(unique_labels) <= 1:
        logger.warning("Only one cluster found, considered invalid")
        return False
    
    # Check for minimum cluster size
    smallest_cluster_fraction = min(counts[counts > 0]) / total_points
    if smallest_cluster_fraction < min_cluster_fraction:
        logger.warning(f"Smallest cluster fraction {smallest_cluster_fraction:.4f} is below threshold {min_cluster_fraction}")
        return False
    
    # Check for noise (if applicable, e.g., for DBSCAN)
    if -1 in unique_labels:
        noise_fraction = counts[unique_labels == -1][0] / total_points
        if noise_fraction > max_noise_fraction:
            logger.warning(f"Noise fraction {noise_fraction:.4f} exceeds threshold {max_noise_fraction}")
            return False
    
    # Check for dominating cluster
    largest_cluster_fraction = max(counts) / total_points
    if largest_cluster_fraction > max_cluster_fraction:
        logger.warning(f"Largest cluster fraction {largest_cluster_fraction:.4f} exceeds threshold {max_cluster_fraction}")
        return False
    
    logger.debug("Clustering result is valid")
    return True

def create_objective(
    df: pd.DataFrame, 
    algorithm_name: str, 
    param_space: Dict[str, Any], 
    metric_name: str = 'silhouette',
    poor_score_threshold: float = 0.1,
    early_stop_threshold: int = 5,
    clustering_params: Dict[str, float] = {}
) -> Callable:
    """
    Create an objective function for Optuna optimization.

    Args:
        df (pd.DataFrame): Input data for clustering.
        algorithm_name (str): Name of the clustering algorithm.
        param_space (Dict[str, Any]): Parameter space for the algorithm.
        metric_name (str): Name of the objective metric to optimize.
        poor_score_threshold (float): Threshold for considering a trial's score as poor.
        early_stop_threshold (int): Number of consecutive poor trials before early stopping.
        clustering_params (Dict[str, float]): Parameters for is_valid_clustering function.

    Returns:
        Callable: Objective function for Optuna.
    """
    metric_info = OBJECTIVE_METRICS[metric_name]
    direction = metric_info['direction']
    poor_trial_count = 0

    def objective(trial):
        nonlocal poor_trial_count
        params = {}
        for param, space in param_space.items():
            if isinstance(space, tuple):
                if all(isinstance(x, int) for x in space):
                    params[param] = trial.suggest_int(param, space[0], space[1])
                else:
                    params[param] = trial.suggest_float(param, space[0], space[1])
            elif isinstance(space, list):
                params[param] = trial.suggest_categorical(param, space)
        
        logger.debug(f"Trying parameters: {params}")
        try:
            labels = ALGORITHM_FUNCTIONS[algorithm_name](df, **params)
        except Exception as e:
            logger.warning(f"Error during clustering: {str(e)}. Skipping this trial.")
            poor_trial_count += 1
            if poor_trial_count >= early_stop_threshold:
                raise optuna.exceptions.TrialPruned()
            return float('-inf') if direction == 'maximize' else float('inf')
        
        if not is_valid_clustering(labels, **clustering_params):
            logger.warning(f"Invalid clustering result for parameters: {params}")
            poor_trial_count += 1
            if poor_trial_count >= early_stop_threshold:
                raise optuna.exceptions.TrialPruned()
            return float('-inf') if direction == 'maximize' else float('inf')
        
        try:
            score = metric_info['function'](df, labels)
            logger.debug(f"Metric {metric_name} score: {score}")
            if score < poor_score_threshold:
                poor_trial_count += 1
                if poor_trial_count >= early_stop_threshold:
                    raise optuna.exceptions.TrialPruned()
            else:
                poor_trial_count = 0  # Reset counter for good scores
            return score if direction == 'maximize' else -score
        except ValueError as e:
            logger.warning(f"Error calculating metric: {str(e)}. Parameters: {params}")
            poor_trial_count += 1
            if poor_trial_count >= early_stop_threshold:
                raise optuna.exceptions.TrialPruned()
            return float('-inf') if direction == 'maximize' else float('inf')

    return objective

def optimize_clustering_algorithm(
    df: pd.DataFrame, 
    algorithm_name: str, 
    custom_param_space: Optional[Dict[str, Any]] = None,
    metric_name: str = 'silhouette',
    n_trials: int = 50,
    timeout: Optional[int] = None,
    poor_score_threshold: float = 0.1,
    early_stop_threshold: int = 5,
    clustering_params: Dict[str, float] = {}
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Optimize hyperparameters for a clustering algorithm using Optuna.
    
    Args:
        df (pd.DataFrame): Input data for clustering.
        algorithm_name (str): Name of the clustering algorithm ('kmeans', 'dbscan', or 'gmm').
        custom_param_space (Optional[Dict[str, Any]]): Custom parameter space to override defaults.
        metric_name (str): Objective metric to optimize (e.g., 'silhouette', 'davies_bouldin').
        n_trials (int): Number of optimization trials.
        timeout (Optional[int]): Timeout for the optimization process in seconds.
        poor_score_threshold (float): Threshold for considering a trial's score as poor.
        early_stop_threshold (int): Number of consecutive poor trials before early stopping.
        clustering_params (Dict[str, float]): Parameters for is_valid_clustering function.
    
    Returns:
        Tuple[Dict[str, Any], List[Dict[str, Any]]]: Best hyperparameters and list of all trial results.
    
    Raises:
        ValueError: If the algorithm or metric is not supported.
    """
    logger.info(f"Starting optimization for {algorithm_name} with metric {metric_name}")
    
    if algorithm_name not in DEFAULT_PARAM_SPACES:
        raise ValueError(f"Unsupported algorithm: {algorithm_name}")
    
    if metric_name not in OBJECTIVE_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name}")

    default_param_space = adjust_param_space(df, algorithm_name, DEFAULT_PARAM_SPACES[algorithm_name])
    param_space = merge_param_spaces(default_param_space, custom_param_space)
    validate_param_space(algorithm_name, param_space)

    objective = create_objective(
        df, 
        algorithm_name, 
        param_space, 
        metric_name, 
        poor_score_threshold, 
        early_stop_threshold,
        clustering_params
    )
    direction = OBJECTIVE_METRICS[metric_name]['direction']

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    
    # Collect all trial results
    trial_results = []
    for trial in study.trials:
        result = {
            'params': trial.params,
            'value': trial.value,
            'state': trial.state.name,
            'number': trial.number,
            'duration': trial.duration.total_seconds(),
            'datetime_start': trial.datetime_start.isoformat(),
            'datetime_complete': trial.datetime_complete.isoformat() if trial.datetime_complete else None
        }
        trial_results.append(result)
    
    logger.info(f"Optimization completed. Best parameters: {study.best_params}")
    return study.best_params, trial_results

# Example usage
if __name__ == "__main__":
    import numpy as np
    
    # Generate sample data
    X = np.random.rand(100, 2)
    df = pd.DataFrame(X, columns=['feature1', 'feature2'])
    
    # Example of using the optimize_clustering_algorithm function for KMeans
    kmeans_params, kmeans_trials = optimize_clustering_algorithm(
        df, 
        algorithm_name='kmeans',
        metric_name='silhouette',
        n_trials=20,
        poor_score_threshold=0.2,
        early_stop_threshold=3,
        clustering_params={'min_cluster_fraction': 0.1, 'max_cluster_fraction': 0.8}
    )
    
    print("Best parameters for KMeans:", kmeans_params)
    print(f"Number of KMeans trials: {len(kmeans_trials)}")
    print("Sample trial result:", kmeans_trials[0])

    # Example of using the optimize_clustering_algorithm function for GMM
    gmm_params, gmm_trials = optimize_clustering_algorithm(
        df, 
        algorithm_name='gmm',
        metric_name='calinski_harabasz',
        n_trials=20,
        poor_score_threshold=0.15,
        early_stop_threshold=4,
        clustering_params={'min_cluster_fraction': 0.08, 'max_cluster_fraction': 0.85}
    )
    
    print("Best parameters for GMM:", gmm_params)
    print(f"Number of GMM trials: {len(gmm_trials)}")
    print("Sample trial result:", gmm_trials[0])