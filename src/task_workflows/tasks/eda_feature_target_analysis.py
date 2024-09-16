# src/task_workflows/tasks/eda_feature_target_analysis.py

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from analysis.visualization.plot_functions import plot_correlation, plot_anova, plot_histogram
from analysis.eda.statistical_analysis import check_normality, assess_feature_correlation
from analysis.eda.feature_analysis import calculate_mutual_information, calculate_feature_target_correlation
from task_workflows.task_registry import register_task
import os
import json

@register_task("eda_feature_target_analysis")
def eda_feature_target_analysis(df: pd.DataFrame, features: list, target: str, output_dir: str):
    """
    Analyze the impact of multiple features on the target variable.
    
    Args:
    df (pd.DataFrame): The input dataframe
    features (list): List of feature names to analyze
    target (str): The name of the target column
    output_dir (str): Directory to save output files
    
    Returns:
    dict: A dictionary containing analysis results and visualization paths for each feature
    """
    
    if not all(feature in df.columns for feature in features):
        raise ValueError("Not all specified features exist in the DataFrame")
    if target not in df.columns:
        raise ValueError("Target column not found in the DataFrame")
    
    # Added directory creation
    os.makedirs(output_dir, exist_ok=True)
    
    overall_results = {}

    for feature in features:
        print(f"Analyzing feature: {feature}")
        results = analyze_single_feature(df, feature, target, output_dir)
        overall_results[feature] = results
        print(f"Completed analysis for feature: {feature}")

    # Save overall results
    save_results(overall_results, output_dir)
    print("Analysis completed. Results saved.")
    return overall_results

def analyze_single_feature(df: pd.DataFrame, feature: str, target: str, output_dir: str):
    print(f"Analyzing single feature: {feature}")
    # print(df[feature].head())
    results = {}
    
    # Determine feature type
    feature_type = 'continuous' if pd.api.types.is_numeric_dtype(df[feature]) else 'categorical'
    target_type = 'continuous' if pd.api.types.is_numeric_dtype(df[target]) else 'categorical'
    # print(f"Feature type: {feature_type}")
    # print(f"Target type: {target_type}")
    results['feature_type'] = feature_type
    results['target_type'] = target_type
    
    # Basic statistics
    results['basic_stats'] = df[feature].describe().to_dict()
    try:
        # Correlation analysis
        if feature_type == 'continuous' and target_type == 'continuous':
            correlation = calculate_feature_target_correlation(df[[feature]], df[target])
            correlation_value = correlation.loc['Correlation', feature]  # Access the scalar value
            results['correlation'] = correlation_value
            
            test_results = {
                'test_type': 'Pearson Correlation',  # Add 'test_type' key
                'correlation': float(correlation_value),  # Ensure it's a scalar float
                'p_value': 0.05  # Placeholder; replace with actual calculation result
            }

            vis_path = plot_correlation(df[feature], df[target], test_results, filename=f"{feature}_correlation.png", output_dir=output_dir)
            results['correlation_plot'] = vis_path
        
        # ANOVA for categorical variables
        elif feature_type == 'categorical' and target_type == 'continuous':
            df[feature] = df[feature].astype(str) 
            f_statistic, p_value = stats.f_oneway(*[group[target].values for name, group in df.groupby(feature)])
            results['anova'] = {'f_statistic': f_statistic, 'p_value': p_value}
            test_results = {
                'test_type': 'ANOVA',  # Add 'test_type' key
                'p_value': p_value
            }
            vis_path = plot_anova(df[target], df[feature], test_results, filename=f"{feature}_anova.png", output_dir=output_dir)
            results['anova_plot'] = vis_path
        
        # Chi-square test for categorical vs categorical
        elif feature_type == 'categorical' and target_type == 'categorical':
            chi2, p_value, dof, expected = stats.chi2_contingency(pd.crosstab(df[feature], df[target]))
            results['chi_square'] = {'chi2': chi2, 'p_value': p_value, 'dof': dof}
        
        # Mutual Information only if both are numeric
        if feature_type == 'continuous' and target_type == 'continuous':
            print(f"Analyzing mutual information for {feature}...")
            variance = df[feature].var()
            # print(f"Variance for {feature}: {variance}")
            mi = calculate_mutual_information(df[[feature]], df[target])
            # print(f"Mutual Information for {feature}: {mi}")
            if mi[feature] == 0:
                print(f"Mutual Information for {feature} is 0. No information gain with respect to the target.")
            
            results['mutual_information'] = mi[feature]
        
        # Normality test
        if feature_type == 'continuous':
            is_normal, normality_p_value = check_normality(df[feature])
            results['normality_test'] = {'is_normal': is_normal, 'p_value': normality_p_value}
            vis_path = plot_histogram(df[feature], f"Distribution of {feature}", feature, "Frequency", f"{feature}_distribution.png", output_dir)
            results['distribution_plot'] = vis_path
    
    except Exception as e:
        # Added error catching and logging
        results['error'] = str(e)
        print(f"Error analyzing feature {feature}: {str(e)}")
    
    return results


def save_results(results, output_dir):
    """Save results in a consistent CSV format and handle plot paths properly."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through each feature's results
    for feature, feature_results in results.items():
        for key, value in feature_results.items():
            # Convert dictionaries to DataFrame and save as CSV
            if isinstance(value, dict):
                pd.DataFrame([value]).to_csv(os.path.join(output_dir, f"{feature}_{key}.csv"), index=False)
            
            # Save Series results as CSV
            elif isinstance(value, pd.Series):
                value.to_frame(name=key).to_csv(os.path.join(output_dir, f"{feature}_{key}.csv"))
            
            # Save DataFrame results as CSV
            elif isinstance(value, pd.DataFrame):
                value.to_csv(os.path.join(output_dir, f"{feature}_{key}.csv"))
            
            # Save scalar (float) results as CSV
            elif isinstance(value, (float, int, np.number)):  # Handle scalar values
                pd.DataFrame([{key: value}]).to_csv(os.path.join(output_dir, f"{feature}_{key}.csv"), index=False)
            
            # Handle plot file paths by saving them in a text file
            elif isinstance(value, str) and value.endswith('.png'):
                plot_filename = os.path.basename(value).replace('.png', '')
                with open(os.path.join(output_dir, f"{feature}_{plot_filename}_path.txt"), 'w') as f:
                    f.write(value)
            
            # Handle strings like 'feature_type', 'target_type', 'error' as a simple CSV file
            elif isinstance(value, str):
                pd.DataFrame([{key: value}]).to_csv(os.path.join(output_dir, f"{feature}_{key}.csv"), index=False)
            
            # Skip unhandled types
            else:
                print(f"Skipping saving for unhandled result type: {feature} - {key}")



# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_classification
    import os

    # Create a synthetic dataset
    X, y = make_classification(n_samples=1000, n_features=5, n_informative=3, n_redundant=1, n_classes=2, random_state=42)
    feature_names = [f'feature_{i}' for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # Add a categorical feature
    df['categorical_feature'] = pd.cut(df['feature_0'], bins=3, labels=['low', 'medium', 'high'])

    # Set up output directory
    output_dir = 'test_results'
    
    # Run EDA feature target analysis
    print("Running EDA feature target analysis...")
    features_to_analyze = feature_names + ['categorical_feature']
    print("Features to analyze:", features_to_analyze)
    results = eda_feature_target_analysis(df, features_to_analyze, 'target', output_dir)

    # Print results
    print("\nAnalysis Results:")
    for feature, feature_results in results.items():
        print(f"\nResults for feature: {feature}")
        for key, value in feature_results.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            elif isinstance(value, float):  # Handle scalar values properly
                print(f"  {key}: {value:.4f}")  # Format float to 4 decimal places
            else:
                print(f"  {key}: {value}")

    # Check for generated files
    print("\nGenerated files:")
    for filename in os.listdir(output_dir):
        print(f"  {filename}")

    print("\nEDA feature target analysis completed successfully.")