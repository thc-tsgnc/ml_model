# src/task_workflows/tasks/eda_statistical_analysis.py

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from task_workflows.task_registry import register_task
from analysis.eda.statistical_analysis import (
    run_statistical_analysis,
    DataType  # Import DataType enum to convert string to enum
)

@register_task("eda_statistical_analysis")
def eda_statistical_analysis(df: pd.DataFrame, 
                             columns: List[str], 
                             target_column: str, 
                             target_type: str,  # Accepts a string that describes the type
                             output_dir: str,
                             p_threshold: float = 0.05,
                             normality_test: str = 'shapiro',
                             homogeneity_test: str = 'levene') -> Dict[str, Any]:
    """
    Perform statistical analysis as part of the EDA process.
    
    Args:
    df (pd.DataFrame): The input dataframe
    columns (List[str]): List of column names to analyze
    target_column (str): Name of the target column
    target_type (str): The data type of the target column ('continuous', 'nominal', 'ordinal', etc.)
    output_dir (str): Directory to save output files
    p_threshold (float): P-value threshold for statistical significance
    normality_test (str): Test to use for normality check ('shapiro' or 'kstest')
    homogeneity_test (str): Test to use for homogeneity of variance check ('levene' or 'bartlett')
    
    Returns:
    Dict[str, Any]: A dictionary containing statistical analysis results and visualization paths
    """
    # Convert string target_type to DataType Enum
    try:
        target_type_enum = DataType(target_type.lower())  # Convert to DataType enum instance
    except ValueError:
        raise ValueError(f"Invalid target type '{target_type}'. Must be one of: {[e.value for e in DataType]}")

    # Run statistical analysis with the correct target type
    results = run_statistical_analysis(
        df=df,
        columns=columns,
        target=target_column,
        target_type=target_type_enum,  # Use target_type_enum here
        p_threshold=p_threshold,
        output_path=output_dir,
        data_playtype="eda_analysis",  # Example value for naming
        output_format='json',  # Output format can be 'json', 'csv', or 'excel'
        normality_test=normality_test,
        homogeneity_test=homogeneity_test
    )

    return results

# Example usage
if __name__ == "__main__":
    # Create a sample dataset
    np.random.seed(42)
    sample_size = 1000
    df = pd.DataFrame({
        'age': np.random.randint(18, 80, sample_size),
        'income': np.random.normal(50000, 15000, sample_size),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], sample_size),
        'satisfaction': np.random.choice(['Low', 'Medium', 'High'], sample_size),
        'performance': np.random.normal(70, 15, sample_size)
    })

    # Define parameters for the analysis
    columns_to_analyze = ['age', 'income', 'education', 'satisfaction']
    target_column = 'performance'
    target_type = 'continuous'  # This can be 'continuous', 'nominal', etc. based on your analysis needs
    output_directory = "results/eda_statistical_analysis"

    # Run the EDA statistical analysis
    results = eda_statistical_analysis(
        df=df,
        columns=columns_to_analyze,
        target_column=target_column,
        target_type=target_type,  # Provide target type here
        output_dir=output_directory,
        p_threshold=0.05,
        normality_test='shapiro',
        homogeneity_test='levene'
    )

    # Print a summary of the results
    print("\nStatistical Analysis Results Summary:")
    for column, result in results.items():
        print(f"\nColumn: {column}")
        print(f"Test Type: {result.get('test_type', 'N/A')}")
        print(f"P-value: {result.get('p_value', 'N/A')}")
        print(f"Interpretation: {result.get('interpretation', 'N/A')}")
        if 'visualizations' in result:
            print("Visualizations generated:")
            for vis_type, vis_path in result['visualizations'].items():
                print(f"  - {vis_type}: {vis_path}")
