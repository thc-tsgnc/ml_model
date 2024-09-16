# src/analysis/eda/binning_analysis.py

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
from typing import List, Dict, Any, Optional
import logging
from analysis.visualization.plot_functions import plot_histogram, plot_line
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_directory_exists(directory: str):
    """Ensure that a directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
                    
                    
def validate_metric_inputs(binned_data: np.ndarray, target: np.ndarray) -> None:
    """Validate input arrays for metric calculations."""
    if len(binned_data) != len(target):
        raise ValueError("binned_data and target must have the same length")
    if len(binned_data) == 0:
        raise ValueError("Input arrays cannot be empty")
    
def validate_binning_process_inputs(df: pd.DataFrame, column: str, target_column: str, min_bins: int, max_bins: int) -> None:
    """Validate input data and parameters for the binning process."""
    if column not in df.columns or target_column not in df.columns:
        raise ValueError(f"Columns '{column}' or '{target_column}' are not in the DataFrame")
    if df[column].nunique() < min_bins:
        raise ValueError(f"Column '{column}' must have at least {min_bins} unique values for binning")
    if not np.issubdtype(df[column].dtype, np.number):
        raise ValueError(f"Column '{column}' must be numeric for binning")
    if min_bins <= 1 or max_bins <= 1 or max_bins < min_bins:
        raise ValueError("Invalid range for min_bins and max_bins")

def calculate_bin_probabilities(binned_data: np.ndarray) -> np.ndarray:
    """
    Calculate bin probabilities.
    
    Args:
    binned_data (np.ndarray): Binned feature data.
    
    Returns:
    np.ndarray: Array of bin probabilities.
    """
    return np.bincount(binned_data) / len(binned_data)

def calculate_chi_square(binned_data: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calculate the Chi-Square statistic for the binned data and target variable.
    
    Args:
    binned_data (np.ndarray): Binned feature data.
    target (np.ndarray): Target variable data.
    
    Returns:
    Dict[str, float]: Dictionary containing Chi-Square statistic and p-value.
    """
    validate_metric_inputs(binned_data, target)
    contingency_table = pd.crosstab(binned_data, target)
    chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
    return {"chi_square": chi2, "chi_square_p_value": p_value}

def calculate_gini_index(binned_data: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calculate the Gini Index for the binned data.
    
    Args:
    binned_data (np.ndarray): Binned feature data.
    target (np.ndarray): Target variable data.
    
    Returns:
    Dict[str, float]: Dictionary containing Gini Index value.
    """
    validate_metric_inputs(binned_data, target)
    
    # Calculate probabilities of each bin
    bin_probabilities = calculate_bin_probabilities(binned_data)
    
    # Gini index is 1 minus the sum of squared probabilities
    gini = 1 - np.sum(bin_probabilities ** 2)
    
    return {"gini_index": gini}

def calculate_mse_between_bins(binned_data: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calculate the Mean Squared Error between bins.
    
    Args:
    binned_data (np.ndarray): Binned feature data.
    target (np.ndarray): Target variable data.
    
    Returns:
    Dict[str, float]: Dictionary containing MSE value.
    """
    validate_metric_inputs(binned_data, target)
    bin_means = pd.DataFrame({'bin': binned_data, 'target': target}).groupby('bin')['target'].mean()
    overall_mean = target.mean()
    mse = np.mean((bin_means - overall_mean) ** 2)  # Corrected the MSE calculation
    return {"mse_between_bins": mse}


def calculate_entropy(counts):
    """Calculate entropy using Numba for improved performance."""
    probabilities = counts / np.sum(counts)
    return -np.sum(probabilities * np.log2(probabilities + 1e-10))

def calculate_information_gain(binned_data: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calculate the Information Gain for the binned data.
    
    Args:
    binned_data (np.ndarray): Binned feature data.
    target (np.ndarray): Target variable data.
    
    Returns:
    Dict[str, float]: Dictionary containing Information Gain value.
    """
    validate_metric_inputs(binned_data, target)
    entropy_before = calculate_entropy(np.bincount(target))
    
    bin_entropies = []
    bin_probabilities = calculate_bin_probabilities(binned_data)
    for bin_value in np.unique(binned_data):
        bin_mask = binned_data == bin_value
        bin_target = target[bin_mask]
        bin_entropy = calculate_entropy(np.bincount(bin_target))
        bin_entropies.append(bin_entropy)
    
    entropy_after = np.sum(np.array(bin_entropies) * bin_probabilities)
    information_gain = entropy_before - entropy_after
    return {"information_gain": information_gain}

def calculate_ks_statistic(binned_data: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """
    Calculate the Kolmogorov-Smirnov statistic for the binned data.
    
    Args:
    binned_data (np.ndarray): Binned feature data.
    target (np.ndarray): Target variable data.
    
    Returns:
    Dict[str, float]: Dictionary containing KS statistic value.
    """
    validate_metric_inputs(binned_data, target)
    unique_bins = np.unique(binned_data)  # Corrected typo from "npned_data" to "np.unique(binned_data)"
    max_ks = 0
    for bin_value in unique_bins:
        bin_mask = binned_data == bin_value
        ks_stat, _ = stats.ks_2samp(target[bin_mask], target[~bin_mask])
        max_ks = max(max_ks, ks_stat)
    return {"ks_statistic": max_ks}

def find_optimal_bins(df: pd.DataFrame, column: str, target_column: str, min_bins: int, max_bins: int, strategy: str = 'quantile', min_sample_size_pct: float = 0.05) -> List[Dict[str, Any]]:
    """
    Find the optimal number of bins for a given column based on various metrics.
    """
    results = []
    X = df[column].values.reshape(-1, 1)
    y = df[target_column].values
    
    # Check for non-numeric or missing data
    if not np.issubdtype(X.dtype, np.number):
        raise ValueError(f"Column '{column}' must contain numeric data for binning.")
    
    for n_bins in range(min_bins, max_bins + 1):
        try:
            kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy=strategy)
            X_binned = kbd.fit_transform(X).ravel()

            # Ensure bins are correctly formed
            if np.issubdtype(X_binned.dtype, np.floating) and np.allclose(X_binned, np.round(X_binned)):
                X_binned = np.round(X_binned).astype(int)
            
            bin_counts = np.bincount(X_binned.astype(int))
            min_samples = df.shape[0] * min_sample_size_pct
            
            # Check minimum sample size
            meets_min_sample_size = all(count >= min_samples for count in bin_counts)
            
            if meets_min_sample_size:
                # Ensure valid metrics before appending
                try:
                    metrics = {
                        'n_bins': n_bins,
                        **calculate_chi_square(X_binned, y),
                        **calculate_gini_index(X_binned, y),
                        **calculate_mse_between_bins(X_binned, y),
                        **calculate_information_gain(X_binned, y),
                        **calculate_ks_statistic(X_binned, y),
                        'mutual_info': mutual_info_score(X_binned, y)
                    }
                    results.append(metrics)
                except Exception as metric_error:
                    logger.error(f"Error calculating metrics for {n_bins} bins: {metric_error}")
            else:
                # Log a warning when the bins do not meet the minimum sample size
                logger.warning(f"Skipping {n_bins} bins for column '{column}' due to insufficient sample size in some bins.")
        except ValueError as e:
            # Handle ValueError raised by KBinsDiscretizer
            logger.warning(f"Could not create {n_bins} bins for column '{column}' with strategy '{strategy}': {e}")
        except Exception as e:
            # Catch all other exceptions and log them
            logger.error(f"Error occurred for {n_bins} bins with strategy {strategy}: {str(e)}")
    
    if not results:
        # If no valid binning results were found, log an informative message
        logger.info(f"No valid binning could be determined for column '{column}' between {min_bins} and {max_bins} bins. Consider reviewing the data distribution or using different parameters.")
    
    return results



def generate_bin_report(results: List[Dict[str, Any]], output_path: str):
    """
    Generate a CSV report summarizing the binning evaluation results.
    """
    if not results:
        logger.error("No valid binning results available for report generation.")
        return None
    
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Create a DataFrame from results
        df_report = pd.DataFrame(results)
        df_report.to_csv(output_path, index=False)
        
        if 'mutual_info' in df_report.columns:
            optimal_bin_info = df_report.loc[df_report['mutual_info'].idxmax()]
            with open(output_path, 'a') as f:
                f.write("\nOptimal Bin Summary:\n")
                f.write(f"Optimal Bins: {optimal_bin_info['n_bins']}\n")
                f.write(f"Highest Mutual Information: {optimal_bin_info['mutual_info']}\n")
            logger.info(f"Binning report saved to: {output_path}")
        else:
            logger.error("Missing 'mutual_info' in binning results.")
        return output_path
    except Exception as e:
        logger.error(f"Failed to generate binning report: {str(e)}")
        return None

    

def generate_visualizations(df: pd.DataFrame, column: str, binning_results: List[Dict[str, Any]], 
                            output_dir: str = 'results/plots') -> List[str]:
    """Generate visualizations for binning results."""
    visualization_paths = []

    # 1. Histogram of the original data
    hist_path = plot_histogram(
        data=df[column],
        title=f"Distribution of {column}",
        xlabel=column,
        ylabel="Frequency",
        filename=f"{column}_distribution.png",
        output_dir=output_dir
    )
    visualization_paths.append(hist_path)

    # 2. Line plot of metrics vs number of bins
    metrics_to_plot = ['mutual_info', 'chi_square', 'information_gain', 'ks_statistic']
    for metric in metrics_to_plot:
        line_path = plot_line(
            x=[result['n_bins'] for result in binning_results],
            y=[result[metric] for result in binning_results],
            title=f"{metric.replace('_', ' ').title()} vs Number of Bins",
            xlabel="Number of Bins",
            ylabel=metric.replace('_', ' ').title(),
            filename=f"{column}_{metric}_vs_bins.png",
            output_dir=output_dir
        )
        visualization_paths.append(line_path)

    return visualization_paths

def perform_eda_binning(
    df: pd.DataFrame, 
    column: str, 
    target_column: str, 
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Perform EDA using a binning strategy and generate a report.
    
    Args:
    df (pd.DataFrame): The input DataFrame.
    column (str): The column to bin.
    target_column (str): The name of the target column.
    config (Optional[Dict[str, Any]]): Configuration dictionary for binning parameters.
    
    Returns:
    Dict[str, Any]: A dictionary containing binning results, optimal bin count, and visualization paths.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

    # Default configuration
    default_config = {
        'min_bins': 2,
        'max_bins': 20,
        'strategy': 'quantile',
        'min_sample_size_pct': 0.05,
        'output_path': os.path.join(project_root, f'results/example_run/binning/binning_report_{column}.csv'),
        'generate_plots': True,
        'plot_output_dir': os.path.join(project_root, 'results/example_run/binning/plots')
    }

    # Update default config with provided config
    if config:
        default_config.update(config)
    
    logger.info(f"Starting binning analysis for column: {column}")

    # High-level input validation
    try:
        validate_binning_process_inputs(
            df, column, target_column, 
            default_config['min_bins'], default_config['max_bins']
        )
    except ValueError as e:
        logger.error(f"Input validation error: {e}")
        return {'error': str(e)}

    try:
        # Perform the binning analysis
        binning_results = find_optimal_bins(
            df, column, target_column, 
            default_config['min_bins'], 
            default_config['max_bins'], 
            default_config['strategy'], 
            default_config['min_sample_size_pct']
        )

        # Generate the binning report
        generate_bin_report(binning_results, default_config['output_path'])
        
        # Determine optimal bins based on mutual information
        optimal_bins = max(binning_results, key=lambda x: x['mutual_info'])['n_bins']
        
        # Generate visualizations if needed
        visualization_paths = []
        if default_config['generate_plots']:
            visualization_paths = generate_visualizations(
                df, column, binning_results, default_config['plot_output_dir']
            )
        
        logger.info(f"Binning analysis completed for column: {column}")
        
        # Return results
        return {
            'binning_results': binning_results,
            'optimal_bins': optimal_bins,
            'visualization_paths': visualization_paths
        }

    except Exception as e:
        logger.error(f"Error in binning analysis for column {column}: {str(e)}")
        return {'error': str(e)}



# Example usage
if __name__ == "__main__":
    # This section is for testing purposes only
    df = pd.DataFrame({
        'feature': np.random.rand(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    result = perform_eda_binning(df, 'feature', 'target')
    print(result)