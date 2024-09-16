# src/task_workflows/tasks/eda_binning_analysis.py

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from task_workflows.task_registry import register_task
from analysis.eda.binning_analysis import (
    find_optimal_bins, generate_bin_report, generate_visualizations,
    validate_binning_process_inputs
)
import logging
import os

logger = logging.getLogger(__name__)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

@register_task("eda_binning_analysis")
def eda_binning_analysis(
    df: pd.DataFrame,
    target_column: str,
    columns: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Perform binning analysis as part of the EDA process.
    
    Args:
    df (pd.DataFrame): The input dataframe
    target_column (str): Name of the target column
    columns (Optional[List[str]]): List of column names to analyze. If None, all numeric columns will be analyzed.
    config (Optional[Dict[str, Any]]): Configuration dictionary for binning parameters
    **kwargs: Additional parameters for quick execution (e.g., min_bins, max_bins)
    
    Returns:
    Dict[str, Any]: A dictionary containing binning results and visualization paths for each analyzed column
    """

    output_dir = os.path.join(project_root, 'results', 'example_run', 'eda_binning')
    

    default_config = {
        'min_bins': 2,
        'max_bins': 20,
        'strategy': 'quantile',
        'min_sample_size_pct': 0.05,
        'output_dir': output_dir,  
        'generate_plots': True
    }
    
    # Merge configurations with priority to kwargs, then config, then default_config
    effective_config = {**default_config, **(config or {}), **kwargs}

    #columns = columns or df.select_dtypes(include=['number']).columns.tolist()
    columns = columns or [col for col in df.select_dtypes(include=['number']).columns.tolist() if col != target_column]

    results = {}
    for column in columns:
        logger.info(f"Starting analysis for column: {column}")
        try:
            results[column] = analyze_single_column(df, column, target_column, effective_config)
            logger.info(f"Analysis completed for column: {column}")
        except Exception as e:
            error_message = f"Error analyzing column {column}: {str(e)}"
            logger.error(error_message)
            results[column] = {
                'error': error_message,
                'error_type': type(e).__name__
            }

    return {
        'column_results': results,
        'summary': generate_summary(results),
        'config_used': effective_config
    }

def analyze_single_column(
    df: pd.DataFrame,
    column: str,
    target_column: str,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    logger.info(f"Starting analysis for column: {column}")
    
    # Validate inputs
    logger.info(f"Validating inputs for column: {column}")
    try:
        validate_binning_process_inputs(df, column, target_column, config['min_bins'], config['max_bins'])
    except ValueError as e:
        error_message = f"Input validation error for column {column}: {str(e)}"
        logger.error(error_message)
        return {
            'error': error_message,
            'error_type': type(e).__name__,
            'optimal_bins': None,  # Ensure key is present
        }

    # Attempt to find optimal bins
    logger.info(f"Finding optimal bins for column: {column}")
    try:
        binning_results = find_optimal_bins(
            df, column, target_column,
            config['min_bins'], config['max_bins'],
            config['strategy'], config['min_sample_size_pct']
        )
        
        # Ensure binning results are not empty
        if not binning_results:
            logger.info(f"No valid binning found for column '{column}'. This may be due to data issues or unsuitable binning parameters.")
            return {
                'error': f"No valid binning found for column '{column}'.",
                'suggestion': "Consider reviewing the data distribution or adjusting binning parameters.",
                'optimal_bins': None,  # Ensure key is present
            }
    except Exception as e:
        error_message = f"Error finding optimal bins for column {column}: {str(e)}"
        logger.error(error_message)
        return {
            'error': error_message,
            'error_type': type(e).__name__,
            'optimal_bins': None,  # Ensure key is present
        }
    
    # Generate bin report
    logger.info(f"Generating bin report for column: {column}")
    try:
        report_path = generate_bin_report(
            binning_results,
            os.path.join(config['output_dir'], f"{column}_binning_report.csv")
        )
        if report_path is None:
            raise ValueError("Failed to generate binning report due to an earlier error.")
    except Exception as e:
        error_message = f"Error generating report for column {column}: {str(e)}"
        logger.error(error_message)
        return {
            'error': error_message,
            'error_type': type(e).__name__,
            'optimal_bins': None,  # Ensure key is present
        }

    # Determine optimal bins based on mutual information
    try:
        if binning_results and 'mutual_info' in binning_results[0]:  # Check if the necessary key exists
            optimal_bins = max(binning_results, key=lambda x: x['mutual_info'])['n_bins']
        else:
            logger.info(f"Could not determine 'optimal_bins' for column '{column}' as 'mutual_info' is missing in binning results.")
            return {
                'error': f"Could not determine 'optimal_bins' for column '{column}'.",
                'suggestion': "Ensure that the dataset has sufficient variability and the binning parameters are appropriate.",
                'optimal_bins': None,  # Ensure key is present
            }
    except (KeyError, ValueError) as e:
        error_message = f"Error determining optimal bins for column {column}: {str(e)}"
        logger.error(error_message)
        return {
            'error': error_message,
            'error_type': type(e).__name__,
            'optimal_bins': None,  # Ensure key is present
        }

    # Generate visualizations if needed
    visualization_paths = []
    if config['generate_plots']:
        logger.info(f"Generating visualizations for column: {column}")
        try:
            visualization_paths = generate_visualizations(
                df, column, binning_results, 
                os.path.join(config['output_dir'], 'plots')
            )
        except Exception as e:
            logger.error(f"Error generating visualizations for column {column}: {str(e)}")

    logger.info(f"Analysis completed successfully for column: {column}")
    return {
        'optimal_bins': optimal_bins,  # Ensure key is present and correctly set
        'report_path': report_path,
        'visualization_paths': visualization_paths,
        'binning_results': binning_results
    }




def generate_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a summary of the binning analysis results."""
    # Filter out None values from optimal_bins before calculating the mean
    optimal_bins_list = [r['optimal_bins'] for r in results.values() if 'optimal_bins' in r and r['optimal_bins'] is not None]
    
    # Calculate the mean only if there are valid optimal bins
    average_optimal_bins = np.mean(optimal_bins_list) if optimal_bins_list else None

    summary = {
        'total_columns_analyzed': len(results),
        'successful_analyses': sum(1 for r in results.values() if 'error' not in r),
        'failed_analyses': sum(1 for r in results.values() if 'error' in r),
        'average_optimal_bins': average_optimal_bins
    }
    return summary

if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        'feature1': np.random.rand(1000),
        'feature2': np.random.rand(1000),
        'target': np.random.randint(0, 2, 1000)
    })
    result = eda_binning_analysis(df, 'target', min_bins=3, max_bins=15)
    print(result)