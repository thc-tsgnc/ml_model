# src/analysis/eda/clustering_analysis.py

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from analysis.visualization.plot_functions import (
    plot_elbow, plot_silhouette, plot_cluster_scatter, 
    plot_cluster_sizes, plot_feature_importance_heatmap, 
    plot_algorithm_comparison, plot_metrics_comparison
)

def run_clustering_analysis(df: pd.DataFrame, algorithm: str, params: Dict[str, Any], true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
    # Select and run the clustering algorithm
    if algorithm == 'kmeans':
        model = KMeans(**params)
        labels = model.fit_predict(df)
        centroids = model.cluster_centers_
    elif algorithm == 'dbscan':
        from sklearn.cluster import DBSCAN
        model = DBSCAN(**params)
        labels = model.fit_predict(df)
        centroids = None
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
        cluster_centers=centroids
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
    cluster_centers: Optional[np.ndarray] = None
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
    visualizations = {}

    # Cluster Scatter Plot
    visualizations['cluster_scatter'] = plot_cluster_scatter(df.values, labels, cluster_centers)
    
    # Silhouette Plot (if silhouette score is calculated)
    if 'silhouette_score' in evaluation_metrics:
        visualizations['silhouette_plot'] = plot_silhouette(df.values, labels)
    
    # Cluster Size Bar Plot
    visualizations['cluster_sizes'] = plot_cluster_sizes(labels)
    
    # Feature Importance Heatmap (using cluster means)
    feature_importance = pd.DataFrame({f"Cluster {k}": v['mean'] for k, v in interpretation.items()}).T
    visualizations['feature_importance'] = plot_feature_importance_heatmap(feature_importance)
    
    # Metrics Comparison Plot
    visualizations['metrics_comparison'] = plot_metrics_comparison(evaluation_metrics)

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