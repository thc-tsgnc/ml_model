# src/analysis/eda/clustering_analysis.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.cluster import KMeans, DBSCAN
from scipy.spatial.distance import cdist
from analysis.visualization.plot_functions import (
    plot_elbow, plot_silhouette, plot_cluster_scatter, 
    plot_cluster_sizes, plot_feature_importance_heatmap, 
    plot_algorithm_comparison, plot_metrics_comparison
)
import logging
from modeling.hyperparameter_optimization import optimize_clustering_algorithm, OBJECTIVE_METRICS
from modeling.hyperparameter_optimization import ALGORITHM_FUNCTIONS
from analysis.metrics.clustering_metrics import calculate_internal_metrics
from modeling.clustering_algorithms import run_kmeans, run_dbscan, run_gmm
from pathlib import Path



# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_clustering_for_feature_combinations(
    df: pd.DataFrame,
    feature_combinations: List[List[str]],
    algorithms: List[str],
    metric_name: str = 'silhouette',
    n_trials: int = 50,
    timeout: Optional[int] = None,
    custom_param_spaces: Optional[Dict[str, Dict[str, Any]]] = None
) -> List[Dict[str, Any]]:
    """
    Run clustering analysis for multiple feature combinations and algorithms.

    Args:
        df (pd.DataFrame): Full input data for clustering.
        feature_combinations (List[List[str]]): List of feature subsets to analyze.
        algorithms (List[str]): List of algorithms to run (e.g., ['kmeans', 'dbscan', 'gmm']).
        metric_name (str): Metric to optimize (default: 'silhouette').
        n_trials (int): Number of optimization trials (default: 50).
        timeout (Optional[int]): Timeout for optimization in seconds (default: None).
        custom_param_spaces (Optional[Dict[str, Dict[str, Any]]]): Custom parameter spaces for each algorithm.

    Returns:
        List[Dict[str, Any]]: List of results for each feature combination and algorithm.
    """
    results = []

    # Handle None value for custom_param_spaces
    custom_param_spaces = custom_param_spaces or {}

    # Validate metric_name
    if metric_name not in OBJECTIVE_METRICS:
        raise ValueError(f"Unsupported metric: {metric_name}. Supported metrics are: {', '.join(OBJECTIVE_METRICS.keys())}")

    for features in feature_combinations:
        df_subset = df[features]
        
        for algorithm in algorithms:
            logger.info(f"Running optimization for {algorithm} with features: {features}")
            
            # Get custom parameter space for the current algorithm
            custom_param_space = custom_param_spaces.get(algorithm)

            try:
                # Run optimization
                best_params, trial_results = optimize_clustering_algorithm(
                    df_subset,
                    algorithm_name=algorithm,
                    custom_param_space=custom_param_space,
                    metric_name=metric_name,
                    n_trials=n_trials,
                    timeout=timeout
                )

                # Run clustering with best parameters
                clustering_function = ALGORITHM_FUNCTIONS[algorithm]
                labels = clustering_function(df_subset, **best_params)

                # **Check for the number of unique clusters**
                unique_labels = set(labels) - {-1}  # Exclude noise (-1) for DBSCAN
                if len(unique_labels) < 2:
                    logger.warning(f"Only one cluster found, considered invalid for {algorithm} with features: {features}")
                    results.append({
                        'features': features,
                        'algorithm': algorithm,
                        'error': 'Only one valid cluster found. Invalid result.'
                    })
                    continue  # Skip further processing for this result

                # Calculate evaluation metrics
                metrics = calculate_internal_metrics(df_subset, labels)

                # Collect results
                result = {
                    'features': features,
                    'algorithm': algorithm,
                    'best_params': best_params,
                    'labels': labels,
                    'metrics': metrics,
                    'trial_results': trial_results
                }
                results.append(result)
                
            except Exception as e:
                error_msg = f"Error occurred for algorithm {algorithm} with features {features}: {str(e)}"
                logger.exception(error_msg)  # This logs the full stack trace
                results.append({
                    'features': features,
                    'algorithm': algorithm,
                    'error': error_msg
                })

    return results

def run_clustering_analysis(df: pd.DataFrame, algorithm: str, params: Dict[str, Any], true_labels: Optional[np.ndarray] = None, output_dir: Optional[str] = 'results/clustering_analysis' ) -> Dict[str, Any]:
    project_root = Path(__file__).resolve().parent.parent.parent.parent / 'results'
    output_dir = Path(output_dir) if output_dir else project_root 

    # Select and run the clustering algorithm
    if algorithm == 'kmeans':
        # model = KMeans(**params)
        # labels = model.fit_predict(df)
        # centroids = model.cluster_centers_
        labels, centroids = run_kmeans(df, return_centers=True, **params)
    elif algorithm == 'dbscan':
        labels = run_dbscan(df, **params)
        centroids = None
    elif algorithm == 'gmm':
        labels, centroids = run_gmm(df, **params)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    
    # Evaluate clustering results
    evaluation_metrics = evaluate_clustering_results(df, labels, true_labels, centroids)
    
    # Interpret clusters
    interpretation = interpret_clusters(df, labels)
    
    # Generate visualizations using the updated generate_visualizations function
    visualization_paths = generate_visualizations(
        df=df,
        labels=labels,
        evaluation_metrics=evaluation_metrics,
        interpretation=interpretation,
        cluster_centers=centroids,
        output_dir=output_dir
    )
    
    return {
        'cluster_labels': labels,
        'evaluation_metrics': evaluation_metrics,
        'visualization_paths': visualization_paths,
        'interpretation': interpretation
    }

def evaluate_clustering_results(
    df: pd.DataFrame, 
    labels: np.ndarray, 
    true_labels: Optional[np.ndarray] = None, 
    centroids: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluate clustering results and return metrics with scalar and nested metrics separated.
    """
    # Internal clustering metrics
    evaluation_metrics = calculate_internal_metrics(df, labels)
    
    # External clustering metrics if true labels are provided
    if true_labels is not None:
        external_metrics = calculate_external_metrics(df, labels, true_labels)
        evaluation_metrics.update(external_metrics)
    
    # Distance-based metrics to centroids
    if centroids is not None:
        distance_metrics = calculate_distance_to_centroids(df, labels, centroids)
        
        # Include scalar metrics suitable for plotting
        evaluation_metrics['average_distance_to_centroid'] = distance_metrics['average_distance_to_centroid']
        
        # Include detailed metrics for analysis
        evaluation_metrics['distance_distribution'] = distance_metrics['distance_distribution']

    return evaluation_metrics


def calculate_internal_metrics(df: pd.DataFrame, labels: np.ndarray) -> Dict[str, float]:
    return {
        'silhouette_score': silhouette_score(df, labels),
        'davies_bouldin_score': davies_bouldin_score(df, labels),
        'calinski_harabasz_score': calinski_harabasz_score(df, labels)
    }

def calculate_external_metrics(df: pd.DataFrame, labels: np.ndarray, true_labels: np.ndarray) -> Dict[str, float]:
    return {
        'adjusted_rand_score': adjusted_rand_score(true_labels, labels),
        'normalized_mutual_info_score': normalized_mutual_info_score(true_labels, labels),
        'v_measure_score': v_measure_score(true_labels, labels)
    }

def calculate_distance_to_centroids(df: pd.DataFrame, labels: np.ndarray, centroids: np.ndarray) -> Dict[str, Any]:
    distances = cdist(df, centroids)
    cluster_distances = [distances[labels == i, i] for i in range(len(centroids))]
    
    average_distance = np.mean([np.mean(dist) for dist in cluster_distances])
    distance_distribution = {
        f'cluster_{i}': {
            'min': np.min(dist),
            'max': np.max(dist),
            'mean': np.mean(dist),
            'std': np.std(dist)
        } for i, dist in enumerate(cluster_distances)
    }
    
    return {
        'average_distance_to_centroid': average_distance,
        'distance_distribution': distance_distribution
    }

def interpret_clusters(df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
    interpretation = {}
    unique_labels = np.unique(labels)
    
    for label in unique_labels:
        cluster_data = df[labels == label]
        interpretation[f'cluster_{label}'] = {
            'size': len(cluster_data),
            'mean': cluster_data.mean().to_dict(),
            'std': cluster_data.std().to_dict()
        }
    
    return interpretation


def generate_visualizations(
    df: pd.DataFrame,
    labels: np.ndarray,
    evaluation_metrics: Dict[str, float],
    interpretation: Dict[str, Dict[str, Any]],
    cluster_centers: Optional[np.ndarray] = None,
    output_dir: Optional[str] = None,
) -> Dict[str, str]:
    """
    Generate visualizations for clustering results using existing data.
    
    Args:
    df (pd.DataFrame): The input data used for clustering.
    labels (np.ndarray): The cluster labels for each data point.
    evaluation_metrics (Dict[str, float]): Dictionary of evaluation metrics.
    interpretation (Dict[str, Dict[str, Any]]): Cluster interpretation data.
    cluster_centers (Optional[np.ndarray]): The coordinates of cluster centers (if available).
    
    Returns:
    Dict[str, str]: Dictionary of visualization names and their file paths.
    """
    project_root = Path(__file__).resolve().parent.parent.parent.parent / 'results'
    output_dir = Path(output_dir) if output_dir else project_root
    
    visualizations = {}

    # Cluster Scatter Plot
    visualizations['cluster_scatter'] = plot_cluster_scatter(df.values, labels, cluster_centers, output_dir)
    
    # Silhouette Plot (if silhouette score is calculated)
    if 'silhouette_score' in evaluation_metrics:
        visualizations['silhouette_plot'] = plot_silhouette(df.values, labels, output_dir)
    
    # Cluster Size Bar Plot
    visualizations['cluster_sizes'] = plot_cluster_sizes(labels, output_dir)
    
    # Feature Importance Heatmap (using cluster means)
    feature_importance = pd.DataFrame({f"Cluster {k}": v['mean'] for k, v in interpretation.items()}).T
    visualizations['feature_importance'] = plot_feature_importance_heatmap(feature_importance, output_dir)
    
    # Metrics Comparison Plot
    visualizations['metrics_comparison'] = plot_metrics_comparison(evaluation_metrics, output_dir)

    return visualizations

# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Create a sample DataFrame with numeric features
    df = pd.DataFrame({
        'feature1': np.random.rand(100),  # Random values for feature 1
        'feature2': np.random.rand(100),  # Random values for feature 2
    })

    # Define parameters for KMeans clustering
    kmeans_params = {
        'n_clusters': 3,
        'random_state': 42
    }

    # Run clustering analysis with KMeans algorithm
    result = run_clustering_analysis(df, algorithm='kmeans', params=kmeans_params)

    # Print the results
    print("Cluster Labels:")
    print(result['cluster_labels'])

    print("\nEvaluation Metrics:")
    for metric, value in result['evaluation_metrics'].items():
        print(f"{metric}: {value}")

    print("\nInterpretation of Clusters:")
    for cluster, interpretation in result['interpretation'].items():
        print(f"{cluster}: {interpretation}")

    print("\nVisualization Paths:")
    for viz_name, path in result['visualization_paths'].items():
        print(f"{viz_name}: {path}")
        
    from sklearn.datasets import make_blobs
    # Generate synthetic data
    def generate_sample_data(n_samples: int = 1000, n_features: int = 5, n_clusters: int = 3) -> pd.DataFrame:
        X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters)
        columns = [f'feature_{i}' for i in range(n_features)]
        return pd.DataFrame(X, columns=columns)
    
    df = generate_sample_data()
    print("Sample data shape:", df.shape)

    # Define feature combinations
    feature_combinations: List[List[str]] = [
        ['feature_0', 'feature_1'],
        ['feature_0', 'feature_1', 'feature_2'],
        ['feature_2', 'feature_3', 'feature_4'],
        ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
    ]

    # Define algorithms to test
    algorithms = ['kmeans', 'dbscan', 'gmm']

    # Define custom parameter spaces (optional)
    custom_param_spaces: Dict[str, Dict[str, Any]] = {
        'kmeans': {'n_clusters': (2, 10)},
        'dbscan': {'eps': (0.1, 1.0), 'min_samples': (2, 10)},
        'gmm': {'n_components': (2, 10)}
    }

    try:
        # Run clustering analysis
        results = run_clustering_for_feature_combinations(
            df=df,
            feature_combinations=feature_combinations,
            algorithms=algorithms,
            metric_name='silhouette',
            n_trials=10,  # Reduced for quicker testing
            timeout=600,  # 10 minutes timeout
            custom_param_spaces=custom_param_spaces
        )

        # Process and display results
        for result in results:
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Algorithm: {result['algorithm']}")
                print(f"Features: {result['features']}")
                print(f"Best parameters: {result['best_params']}")
                print(f"Metrics: {result['metrics']}")
                print(f"Number of trials: {len(result['trial_results'])}")
                print("---")

        # Additional analysis (example)
        best_result = max(results, key=lambda x: x['metrics'].get('silhouette_score', -1) if 'metrics' in x else -1)
        print("\nBest overall result:")
        print(f"Algorithm: {best_result['algorithm']}")
        print(f"Features: {best_result['features']}")
        print(f"Best parameters: {best_result['best_params']}")
        print(f"Silhouette score: {best_result['metrics']['silhouette_score']}")

    except ValueError as e:
        print(f"Invalid input: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
