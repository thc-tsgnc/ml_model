# src/analysis/visualization/plot_functions.py

from analysis.visualization.plot_utils import create_figure, add_labels, save_figure, add_colorbar, set_axis_tick_params
import pandas as pd
from typing import Dict, Any, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_samples
import numpy as np
from statsmodels.graphics.gofplots import qqplot
import os

def plot_histogram(data: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, output_dir: str = 'results/plots') -> str:
    """Create and save a histogram plot."""
    _validate_data_series(data)
    fig, ax = create_figure()
    ax.hist(data, bins='auto', edgecolor='black')
    add_labels(ax, title, xlabel, ylabel)
    return save_figure(fig, filename, output_dir)

def plot_line(x: List[float], y: List[float], title: str, xlabel: str, ylabel: str, filename: str, output_dir: str = 'results/plots') -> str:
    """Create and save a line plot."""
    _validate_list(x, 'x')
    _validate_list(y, 'y')
    fig, ax = create_figure()
    ax.plot(x, y, marker='o')
    add_labels(ax, title, xlabel, ylabel)
    return save_figure(fig, filename, output_dir)

def plot_cluster_scatter(data: np.ndarray, labels: np.ndarray, centers: Optional[np.ndarray] = None, output_dir: str = 'results/plots') -> str:
    """Create and save a scatter plot of clusters."""
    _validate_ndarray(data, 'data')
    _validate_ndarray(labels, 'labels')
    
    pca = PCA(n_components=2)
    data_2d = pca.fit_transform(data)

    fig, ax = create_figure()
    scatter = ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='viridis')
    
    if centers is not None:
        centers_2d = pca.transform(centers)
        ax.scatter(centers_2d[:, 0], centers_2d[:, 1], c='red', marker='x', s=200, linewidths=3)
    
    add_labels(ax, 'Cluster Scatter Plot', 'First Principal Component', 'Second Principal Component')
    plt.colorbar(scatter)
    return save_figure(fig, 'cluster_scatter.png', output_dir)

def plot_normality(data: pd.Series, test_results: Dict[str, Any], output_dir: str = 'results/plots') -> str:
    """Create and save a plot showing histogram with normal curve and Q-Q plot."""
    _validate_data_series(data)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Histogram with normal curve
    sns.histplot(data, kde=True, ax=ax1)
    ax1.set_title(f"Histogram with Normal Curve\n{test_results['test_type']}: p={test_results['p_value']:.4f}")
    
    # Q-Q plot
    qqplot(data, line='s', ax=ax2)
    ax2.set_title("Q-Q Plot")
    
    plt.tight_layout()
    return save_figure(fig, f"{data.name}_normality.png", output_dir)

def plot_homogeneity(groups: List[pd.Series], output_dir: str = 'results/plots') -> str:
    """Create and save a box plot for homogeneity of variance."""
    if not groups or not all(isinstance(group, pd.Series) for group in groups):
        raise ValueError("Groups must be a non-empty list of pandas Series.")
    
    fig, ax = create_figure()
    ax.boxplot(groups)
    add_labels(ax, "Homogeneity of Variance", "Groups", "Values")
    return save_figure(fig, "homogeneity_boxplot.png", output_dir)

def plot_anova(data: pd.Series, groups: pd.Series, test_results: Dict[str, Any], filename: str, output_dir: str = 'results/plots') -> str:
    """Create and save an ANOVA box plot."""
    _validate_data_series(data)
    _validate_data_series(groups)

    fig, ax = create_figure()
    sns.boxplot(x=groups, y=data, ax=ax)
    add_labels(ax, f"{test_results['test_type']} Results", "Groups", "Values")
    ax.text(0.05, 0.95, f"p-value: {test_results['p_value']:.4f}", transform=ax.transAxes, verticalalignment='top')
    
    # Corrected: Accepts 'filename' and passes it to 'save_figure'
    return save_figure(fig, filename, output_dir)

def plot_correlation(x: pd.Series, y: pd.Series, test_results: Dict[str, Any], filename: str, output_dir: str = 'results/plots') -> str:
    """Create and save a scatter plot showing correlation with regression line."""
    _validate_data_series(x)
    _validate_data_series(y)

    fig, ax = create_figure()
    sns.regplot(x=x, y=y, ax=ax)
    add_labels(ax, f"{test_results['test_type']} Results", x.name, y.name)
    ax.text(0.05, 0.95, f"Correlation: {test_results['correlation']:.4f}\np-value: {test_results['p_value']:.4f}", transform=ax.transAxes, verticalalignment='top')
    
    # Corrected: Accepts 'filename' and passes it to 'save_figure'
    return save_figure(fig, filename, output_dir)

def plot_chi_square(contingency_table: pd.DataFrame, output_dir: str = 'results/plots') -> str:
    """Create and save a heatmap for chi-square contingency table."""
    _validate_data_frame(contingency_table)

    fig, ax = create_figure(figsize=(10, 8))
    sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu', ax=ax)
    add_labels(ax, "Chi-square Contingency Table", "Variable 1", "Variable 2")
    return save_figure(fig, "chi_square_heatmap.png", output_dir)

def plot_metrics_comparison(metrics: Dict[str, Any], output_dir: str = 'results/plots') -> str:
    """
    Create and save a bar plot of evaluation metrics, handling nested structures appropriately.
    """
    # Only plot metrics that are scalar (int, float)
    flattened_metrics = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}

    if not flattened_metrics:
        raise ValueError("No plottable metrics found.")

    # Plot the scalar metrics
    fig, ax = create_figure()
    ax.bar(flattened_metrics.keys(), flattened_metrics.values())
    add_labels(ax, 'Clustering Evaluation Metrics', 'Metrics', 'Values')
    plt.xticks(rotation=45, ha='right')

    # Save the plot
    return save_figure(fig, 'metrics_comparison.png', output_dir)


def plot_elbow(n_clusters_range: List[int], metric_values: List[float], output_dir: str = 'results/plots') -> str:
    """Create and save an elbow plot for clustering analysis."""
    _validate_list(n_clusters_range, 'n_clusters_range')
    _validate_list(metric_values, 'metric_values')

    fig, ax = create_figure()
    ax.plot(n_clusters_range, metric_values, 'bo-')
    add_labels(ax, 'Elbow Method', 'Number of Clusters', 'Metric Value')
    return save_figure(fig, 'elbow_plot.png', output_dir)

def plot_silhouette(data: np.ndarray, labels: np.ndarray, output_dir: str = 'results/plots') -> str:
    """Create and save a silhouette plot."""
    _validate_ndarray(data, 'data')
    _validate_ndarray(labels, 'labels')

    n_clusters = len(np.unique(labels))
    silhouette_vals = silhouette_samples(data, labels)

    fig, ax = create_figure(figsize=(8, 6))
    y_lower, y_upper = 0, 0

    for i in range(n_clusters):
        cluster_silhouette_vals = silhouette_vals[labels == i]
        cluster_silhouette_vals.sort()
        y_upper += len(cluster_silhouette_vals)
        ax.fill_betweenx(np.arange(y_lower, y_upper), 0, cluster_silhouette_vals, facecolor='C'+str(i), edgecolor='none', alpha=0.7)
        y_lower += len(cluster_silhouette_vals)

    add_labels(ax, 'Silhouette Plot', 'Silhouette Coefficient', 'Cluster')
    ax.axvline(x=silhouette_vals.mean(), color="red", linestyle="--")
    return save_figure(fig, 'silhouette_plot.png', output_dir)

def plot_cluster_sizes(labels: np.ndarray, output_dir: str = 'results/plots') -> str:
    """Create and save a bar plot of cluster sizes."""
    _validate_ndarray(labels, 'labels')

    cluster_sizes = pd.Series(labels).value_counts().sort_index()
    fig, ax = create_figure()
    cluster_sizes.plot(kind='bar', ax=ax)
    add_labels(ax, 'Cluster Sizes', 'Cluster', 'Number of Samples')
    return save_figure(fig, 'cluster_sizes.png', output_dir)

def plot_feature_importance_heatmap(feature_importance: pd.DataFrame, output_dir: str = 'results/plots') -> str:
    """Create and save a heatmap of feature importance for each cluster."""
    _validate_data_frame(feature_importance)

    fig, ax = create_figure(figsize=(10, 8))
    sns.heatmap(feature_importance, annot=True, cmap='viridis', ax=ax)
    add_labels(ax, 'Feature Importance Heatmap', 'Features', 'Clusters')
    return save_figure(fig, 'feature_importance_heatmap.png', output_dir)

def plot_algorithm_comparison(algo_metrics: Dict[str, Dict[str, float]], output_dir: str = 'results/plots') -> str:
    """Create and save a bar plot comparing metrics across different algorithms."""
    if not algo_metrics:
        raise ValueError("Algorithm metrics must be a non-empty dictionary.")

    df = pd.DataFrame(algo_metrics).T
    fig, ax = create_figure(figsize=(10, 6))
    df.plot(kind='bar', ax=ax)
    add_labels(ax, 'Algorithm Comparison', 'Algorithms', 'Metric Values')
    ax.legend(title='Metrics')
    plt.xticks(rotation=45)
    return save_figure(fig, 'algorithm_comparison.png', output_dir)

# Helper validation functions
def _validate_data_series(data: pd.Series) -> None:
    """Validate that the data is a non-empty pandas Series."""
    if not isinstance(data, pd.Series) or data.empty:
        raise ValueError("Data must be a non-empty pandas Series.")

def _validate_data_frame(data: pd.DataFrame) -> None:
    """Validate that the data is a non-empty pandas DataFrame."""
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Data must be a non-empty pandas DataFrame.")

def _validate_ndarray(array: np.ndarray, name: str) -> None:
    """Validate that the input is a non-empty numpy ndarray."""
    if not isinstance(array, np.ndarray) or array.size == 0:
        raise ValueError(f"{name} must be a non-empty numpy ndarray.")

def _validate_list(lst: List[Any], name: str) -> None:
    """Validate that the input is a non-empty list."""
    if not isinstance(lst, list) or not lst:
        raise ValueError(f"{name} must be a non-empty list.")
    
def plot_correlation_heatmap(correlation_matrix: pd.DataFrame, filename: str, output_dir: str = 'results/plots') -> str:
    """
    Create and save a heatmap of correlations between features and the target variable.
    
    Args:
    correlation_matrix (pd.DataFrame): The correlation matrix to be plotted.
    filename (str): The name of the file to save the plot.
    output_dir (str): The directory where the plot will be saved.
    
    Returns:
    str: The path to the saved plot.
    """
    # Validate the input correlation matrix is a non-empty DataFrame
    _validate_data_frame(correlation_matrix)
    
    # Create figure and axis using the utility function
    fig, ax = create_figure(figsize=(10, 8))
    
    # Create the heatmap using seaborn
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
    
    # Add a colorbar using the utility function
    add_colorbar(fig, heatmap.get_children()[0], label='Correlation Coefficient')
    
    # Add labels to the plot
    add_labels(ax, 'Feature-Target Correlation Heatmap', 'Features', 'Target')

    # Set tick parameters for better visualization
    set_axis_tick_params(ax, axis='x', rotation=45, ha='right')
    set_axis_tick_params(ax, axis='y', rotation=0)
    
    # Save the figure using the utility function
    return save_figure(fig, filename, output_dir)

def plot_feature_importance_bar(importance_scores: pd.Series, filename: str, output_dir: str = 'results/plots') -> str:
    """
    Create and save a bar plot of feature importances or mutual information scores.
    
    Args:
    importance_scores (pd.Series): A pandas Series containing feature names as the index and their importance scores as the values.
    filename (str): The name of the file to save the plot.
    output_dir (str): The directory where the plot will be saved.
    
    Returns:
    str: The path to the saved plot.
    """
    # Validate the input importance scores
    if not isinstance(importance_scores, pd.Series) or importance_scores.empty:
        raise ValueError("importance_scores must be a non-empty pandas Series.")
    
    # Create figure and axis using the utility function
    fig, ax = create_figure(figsize=(10, 6))
    
    # Plot the bar chart in descending order
    importance_scores.sort_values(ascending=True).plot(kind='barh', ax=ax, color='skyblue', edgecolor='black')
    
    # Add labels to the plot
    add_labels(ax, 'Feature Importances or Mutual Information Scores', 'Importance', 'Features')

    # Set tick parameters for better visualization
    set_axis_tick_params(ax, axis='y', rotation=0)
    
    # Save the figure using the utility function
    return save_figure(fig, filename, output_dir)

def plot_pca_scree(explained_variance_ratio: pd.Series, filename: str, output_dir: str = 'results/plots') -> str:
    """
    Create and save a scree plot for PCA analysis.
    
    Args:
    explained_variance_ratio (pd.Series): A pandas Series containing the explained variance ratios for each Principal Component.
    filename (str): The name of the file to save the plot.
    output_dir (str): The directory where the plot will be saved.
    
    Returns:
    str: The path to the saved plot.
    """
    # Validate the input explained variance ratio
    if not isinstance(explained_variance_ratio, pd.Series) or explained_variance_ratio.empty:
        raise ValueError("explained_variance_ratio must be a non-empty pandas Series.")
    
    # Create figure and axis using the utility function
    fig, ax = create_figure(figsize=(10, 6))
    
    # Plot the scree plot
    ax.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='-')
    
    # Add labels to the plot
    add_labels(ax, 'PCA Scree Plot', 'Principal Component', 'Explained Variance Ratio')

    # Set tick parameters for better visualization
    set_axis_tick_params(ax, axis='x', rotation=0)
    
    # Save the figure using the utility function
    return save_figure(fig, filename, output_dir)

def plot_pca_biplot(pca_result: Dict[str, Any], filename: str, output_dir: str = 'results/plots') -> str:
    """
    Create and save a PCA biplot.
    
    Args:
    pca_result (Dict[str, Any]): A dictionary containing 'pca_df' (DataFrame of PCA results) 
                                 and 'explained_variance_ratio' (Series of explained variance ratios).
    filename (str): The name of the file to save the plot.
    output_dir (str): The directory where the plot will be saved.
    
    Returns:
    str: The path to the saved plot.
    """
    # Validate the input PCA results
    if 'pca_df' not in pca_result or 'explained_variance_ratio' not in pca_result:
        raise ValueError("pca_result must contain 'pca_df' and 'explained_variance_ratio'.")
    
    pca_df = pca_result['pca_df']
    explained_variance_ratio = pca_result['explained_variance_ratio']
    
    # Create figure and axis using the utility function
    fig, ax = create_figure(figsize=(10, 8))
    
    # Scatter plot of the first two principal components
    ax.scatter(pca_df.iloc[:, 0], pca_df.iloc[:, 1], alpha=0.5)
    
    # Add arrows for loadings (contributions of original features)
    for i, (pc1_loading, pc2_loading) in enumerate(zip(pca_df.iloc[:, 0], pca_df.iloc[:, 1])):
        ax.arrow(0, 0, pc1_loading, pc2_loading, color='r', alpha=0.5)
        ax.text(pc1_loading * 1.15, pc2_loading * 1.15, pca_df.columns[i], color='g', ha='center', va='center')
    
    # Add labels to the plot
    add_labels(ax, 'PCA Biplot', f'PC1 ({explained_variance_ratio[0]:.2%} variance)', 
               f'PC2 ({explained_variance_ratio[1]:.2%} variance)')
    
    # Set equal scaling for better visualization
    ax.set_aspect('equal', 'box')
    
    # Save the figure using the utility function
    return save_figure(fig, filename, output_dir)

def plot_vif_scores(vif_scores: pd.Series, filename: str, output_dir: str = 'results/plots') -> str:
    """
    Create and save a bar plot of Variance Inflation Factor (VIF) scores.
    
    Args:
    vif_scores (pd.Series): A pandas Series containing feature names as the index and their VIF scores as the values.
    filename (str): The name of the file to save the plot.
    output_dir (str): The directory where the plot will be saved.
    
    Returns:
    str: The path to the saved plot.
    """
    # Validate the input VIF scores
    if not isinstance(vif_scores, pd.Series) or vif_scores.empty:
        raise ValueError("vif_scores must be a non-empty pandas Series.")
    
    # Create figure and axis using the utility function
    fig, ax = create_figure(figsize=(10, 6))
    
    # Plot the bar chart
    vif_scores.sort_values(ascending=False).plot(kind='bar', ax=ax, color='salmon', edgecolor='black')
    
    # Add labels to the plot
    add_labels(ax, 'Variance Inflation Factor (VIF) Scores', 'Features', 'VIF Score')

    # Set tick parameters for better visualization
    set_axis_tick_params(ax, axis='x', rotation=45, ha='right')
    
    # Add a horizontal line indicating a common threshold for VIF (e.g., VIF > 10)
    ax.axhline(y=10, color='red', linestyle='--', label='VIF > 10')
    ax.legend()
    
    # Save the figure using the utility function
    return save_figure(fig, filename, output_dir)

def plot_boxplot(data: pd.Series, groups: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, output_dir: str = 'results/plots') -> str:
    """Create and save a box plot for visualizing distributions across categories."""
    _validate_data_series(data)
    _validate_data_series(groups)

    fig, ax = create_figure()
    sns.boxplot(x=groups, y=data, ax=ax)
    add_labels(ax, title, xlabel, ylabel)
    
    return save_figure(fig, filename, output_dir)

def plot_bar(data: pd.Series, title: str, xlabel: str, ylabel: str, filename: str, output_dir: str = 'results/plots') -> str:
    """Create and save a bar plot for visualizing counts or averages of categories."""
    _validate_data_series(data)
    
    fig, ax = create_figure()
    data.value_counts().plot(kind='bar', ax=ax)
    add_labels(ax, title, xlabel, ylabel)
    
    return save_figure(fig, filename, output_dir)
