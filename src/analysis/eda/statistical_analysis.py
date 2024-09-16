import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Any, List, Tuple
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn
import json
from pathlib import Path
from datetime import datetime
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import scikit_posthocs as sp
from analysis.visualization.plot_functions import (
    plot_anova, plot_correlation, plot_chi_square, 
    plot_normality, plot_homogeneity, plot_boxplot,
    plot_bar
)
from enum import Enum

class DataType(Enum):
    NOMINAL = 'nominal'
    ORDINAL = 'ordinal'
    CONTINUOUS = 'continuous'
    BINARY = 'binary'
    COUNT = 'count'

def set_target_type(target_column: str, target_type: DataType) -> Dict[str, DataType]:
    """
    Validate and set the data type for the target column.

    :param target_column: The name of the target column.
    :param target_type: The data type of the target column as a DataType Enum.
    :return: A dictionary mapping the target column to its validated data type.
    """
    # Validate target_column is a non-empty string
    if not isinstance(target_column, str) or not target_column.strip():
        raise ValueError("Invalid target column. Must be a non-empty string.")
    
    # Validate target_type is a valid DataType Enum instance
    if target_type is None or not isinstance(target_type, DataType):
        raise ValueError("Invalid target type. Must be an instance of DataType Enum and cannot be None.")
    
    # Return the validated target type mapping
    return {target_column: target_type}

def identify_column_types(df: pd.DataFrame, threshold: int = 10) -> Dict[str, DataType]:
    column_types = {}
    
    for column in df.columns:
        unique_values = df[column].nunique()
        
        if pd.api.types.is_numeric_dtype(df[column]):
            if unique_values <= threshold:
                column_types[column] = DataType.ORDINAL
            else:
                column_types[column] = DataType.CONTINUOUS
        else:
            column_types[column] = DataType.NOMINAL
    
    return column_types

def preprocess_categorical_variables(df: pd.DataFrame, column_types: Dict[str, DataType]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Preprocess categorical variables by encoding them appropriately based on identified types.
    
    :param df: DataFrame to be processed.
    :param column_types: A dictionary where keys are column names and values are DataType Enum.
    :return: A tuple with the processed DataFrame and a dictionary mapping original columns to their one-hot encoded counterparts.
    """
    # Create a copy of the DataFrame to avoid modifying the original DataFrame
    df_processed = df.copy()
    column_mapping = {}  # Store mapping from original column to new encoded columns

    for column, var_type in column_types.items():
        if var_type == DataType.NOMINAL:
            # One-Hot Encode Nominal Variables
            dummies = pd.get_dummies(df_processed[column], prefix=column)
            # Remove the original column and concatenate the new dummy columns
            df_processed = df_processed.drop(columns=[column]).join(dummies)
            # Store the new columns created by one-hot encoding
            new_columns = dummies.columns.tolist()
            column_mapping[column] = new_columns
            
            print(f"DEBUG: One-hot encoded '{column}' into columns: {new_columns}")
        
        elif var_type == DataType.ORDINAL:
            # Ordinal Encode Ordinal Variables
            df_processed[column] = pd.Categorical(df_processed[column], ordered=True).codes
            column_mapping[column] = [column]
            
            print(f"DEBUG: Ordinal encoded '{column}' with codes: {df_processed[column].unique()}")
        
        else:
            # Continuous variables are left unchanged
            column_mapping[column] = [column]

    print("DEBUG: Processed DataFrame columns after encoding:", df_processed.columns.tolist())
    print("DEBUG: Column mapping after preprocessing:", column_mapping)
    
    return df_processed, column_mapping


def generate_visualizations(df: pd.DataFrame, results: Dict[str, Any], column_mapping: Dict[str, List[str]]) -> Dict[str, str]:
    """Generate visualizations based on statistical analysis results."""
    visualizations = {}
    
    for column, result in results.items():
        if result.get("error"):
            continue
        
        test_type = result["test_type"]
        base_filename = f"{column}_{test_type.replace(' ', '_').lower()}.png"
        
        # Reconstruct the original categorical variable from dummy columns for visualization
        if result["column_type"] in ['nominal', 'ordinal'] and column in column_mapping:
            dummy_columns = column_mapping[column]
            reconstructed_column = df[dummy_columns].idxmax(axis=1).str.replace(r'.*_', '', regex=True)
            df['reconstructed_' + column] = reconstructed_column
        
        if test_type in ["One-way ANOVA", "Welch's ANOVA", "Kruskal-Wallis H"]:
            filename = f"{column}_anova_plot.png"
            if f'reconstructed_{column}' in df:
                vis_path = plot_anova(df[result["target"]], df[f'reconstructed_{column}'], result, filename)
                visualizations[f"{column}_anova"] = vis_path
            else:
                print(f"Column(s) for {column} not found in DataFrame: skipping plot generation.")
        
        elif test_type in ["Pearson correlation", "Spearman rank correlation"]:
            filename = f"{column}_correlation_plot.png"
            vis_path = plot_correlation(df[column], df[result["target"]], result, filename)
            visualizations[f"{column}_correlation"] = vis_path
        
        elif test_type == "Chi-square":
            contingency_table = pd.crosstab(df[column], df[result["target"]])
            filename = f"{column}_chi_square_plot.png"
            vis_path = plot_chi_square(contingency_table, filename)
            visualizations[f"{column}_chi_square"] = vis_path
        
        elif result["column_type"] in ['nominal', 'ordinal']:
            # Bar plot for counts or means of categories
            filename = f"{column}_barplot.png"
            if f'reconstructed_{column}' in df:
                vis_path = plot_bar(
                    df[f'reconstructed_{column}'],
                    f"Barplot of {column}",
                    column,
                    "Counts",
                    filename
                )
                visualizations[f"{column}_barplot"] = vis_path
            else:
                print(f"Column(s) for {column} not found in DataFrame: skipping plot generation.")
    
    return visualizations




def save_results(results: Dict[str, Any], output_path: str = "results/statistical_analysis", data_playtype: str = None, format: str = 'json') -> str:
    """Save the analysis results to a file."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts = ["statistical_analysis_results"]
    if data_playtype:
        filename_parts.append(data_playtype)
    filename_parts.append(timestamp)
    
    base_filename = "_".join(filename_parts)
    
    file_path = output_dir / f"{base_filename}.{format}"
    
    if format == 'json':
        with open(file_path, 'w') as f:
            json.dump(results, f, indent=4)
    elif format == 'csv':
        pd.DataFrame(results).T.to_csv(file_path)
    elif format == 'excel':
        pd.DataFrame(results).T.to_excel(file_path)
    else:
        raise ValueError("Unsupported format. Use 'json', 'csv', or 'excel'.")
    
    return str(file_path)

# Helper validation functions
def _validate_data_series(data: pd.Series) -> None:
    """Validate that the data is a non-empty pandas Series."""
    if not isinstance(data, pd.Series) or data.empty:
        raise ValueError("Data must be a non-empty pandas Series.")

def _validate_groups(groups: List[pd.Series]) -> None:
    """Validate that the input is a list of non-empty pandas Series."""
    if not all(isinstance(group, pd.Series) and not group.empty for group in groups):
        raise ValueError("Groups must be a non-empty list of pandas Series.")


def check_normality(data: pd.Series, test: str = 'shapiro', threshold: float = 0.05) -> Tuple[bool, float]:
    _validate_data_series(data)
    if test == 'shapiro':
        _, p_value = stats.shapiro(data)
    elif test == 'kstest':
        _, p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    else:
        raise ValueError("Invalid test specified. Use 'shapiro' or 'kstest'.")
    return p_value > threshold, p_value

def check_homogeneity_of_variances(groups: List[pd.Series], test: str = 'levene', threshold: float = 0.05) -> Tuple[bool, float]:
    _validate_groups(groups)
    if test == 'levene':
        _, p_value = stats.levene(*groups)
    elif test == 'bartlett':
        _, p_value = stats.bartlett(*groups)
    else:
        raise ValueError("Invalid test specified. Use 'levene' or 'bartlett'.")
    return p_value > threshold, p_value

def perform_anova(df: pd.DataFrame, column: str, target: str) -> Dict[str, Any]:
    formula = f"{target} ~ C({column})"
    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return {
        "test_type": "One-way ANOVA",
        "f_statistic": anova_table['F'].iloc[0],
        "p_value": anova_table['PR(>F)'].iloc[0],
        "df_between": anova_table['df'].iloc[0],
        "df_within": anova_table['df'].iloc[1]
    }

def perform_welch_anova(df: pd.DataFrame, column: str, target: str) -> Dict[str, Any]:
    formula = f"{target} ~ C({column})"
    model = ols(formula, data=df).fit()
    welch_anova_table = anova_lm(model, typ=2, robust='hc3')
    return {
        "test_type": "Welch's ANOVA",
        "f_statistic": welch_anova_table.iloc[0]['F'],
        "p_value": welch_anova_table.iloc[0]['PR(>F)']
    }

def perform_kruskal_wallis(df: pd.DataFrame, column: str, target: str) -> Dict[str, Any]:
    groups = [group[target] for _, group in df.groupby(column)]
    h_statistic, p_value = stats.kruskal(*groups)
    return {
        "test_type": "Kruskal-Wallis H",
        "h_statistic": h_statistic,
        "p_value": p_value
    }

def perform_post_hoc_tests(df: pd.DataFrame, column: str, target: str, test_type: str) -> Dict[str, Any]:
    if test_type == "One-way ANOVA":
        post_hoc_test = pairwise_tukeyhsd(df[target], df[column])
        post_hoc_result = {
            'post_hoc_test_name': "Tukey's HSD",
            'post_hoc_summary': post_hoc_test._results_table.data[1:]
        }
    elif test_type == "Welch's ANOVA":
        post_hoc_test = sp.posthoc_tamhane(df, val_col=target, group_col=column)
        post_hoc_result = {
            'post_hoc_test_name': "Tamhane's T2",
            'post_hoc_summary': post_hoc_test.reset_index().melt(id_vars='index', var_name='group2', value_name='p-adj').to_dict('records')
        }
    elif test_type == "Kruskal-Wallis H":
        post_hoc_test = posthoc_dunn(df, val_col=target, group_col=column, p_adjust='bonferroni')
        post_hoc_result = {
            'post_hoc_test_name': "Dunn's test",
            'post_hoc_summary': post_hoc_test.stack().reset_index().rename(columns={'level_0': 'group1', 'level_1': 'group2', 0: 'p-adj'}).to_dict('records')
        }
    else:
        raise ValueError(f"Unsupported test type for post-hoc analysis: {test_type}")
    
    post_hoc_result['post_hoc_test'] = post_hoc_test.to_dict() if isinstance(post_hoc_test, pd.DataFrame) else None
    return post_hoc_result

def test_group_differences(df: pd.DataFrame, dummy_columns: List[str], target: str, p_threshold: float,
                           normality_test: str = 'shapiro', homogeneity_test: str = 'levene') -> Dict[str, Any]:
    """
    Perform group difference tests for categorical variables represented by dummy columns.
    
    :param df: DataFrame containing the data.
    :param dummy_columns: List of one-hot encoded column names corresponding to the original categorical variable.
    :param target: Target variable for the statistical test.
    :param p_threshold: Significance threshold for tests.
    :param normality_test: Test to check normality (default: 'shapiro').
    :param homogeneity_test: Test to check homogeneity of variances (default: 'levene').
    :return: Dictionary containing test results.
    """
    # Debug print statements to check data at various stages
    print("DataFrame columns inside test_group_differences before grouping:", df.columns.tolist())
    print(f"Attempting to group by columns: {dummy_columns}")
    print(f"Unique values in target column '{target}':", df[target].unique())
    print(f"Dummy column unique values count: {df[dummy_columns].nunique()}")

    # Combine dummy columns back into a single categorical variable for plotting
    combined_column = df[dummy_columns].idxmax(axis=1).str.replace(r'.*_', '', regex=True)
    print("Combined column created from dummy columns:", combined_column.unique())

    # Perform the group differences test using the combined categorical variable
    groups = [df.loc[combined_column == category, target].dropna() for category in combined_column.unique()]
    print("Number of groups formed:", len(groups))
    print("Groups and their sizes:", [(category, len(group)) for category, group in zip(combined_column.unique(), groups)])

    # Ensure there are enough groups for analysis
    if len(groups) < 2:
        raise ValueError("Must enter at least two input sample vectors for group differences test.")

    # Check normality of each group
    all_normal = all(check_normality(group, test=normality_test, threshold=p_threshold)[0] for group in groups)
    print(f"All groups normal? {all_normal}")

    # Check homogeneity of variances
    homogeneous_variance, _ = check_homogeneity_of_variances(groups, test=homogeneity_test, threshold=p_threshold)
    print(f"Groups have homogeneous variances? {homogeneous_variance}")

    # Add the combined categorical variable back to the DataFrame for use in statistical tests and plotting
    df['combined_column'] = combined_column

    # Determine which ANOVA test to use based on normality and homogeneity of variances
    if all_normal and homogeneous_variance:
        print("Performing One-way ANOVA...")
        results = perform_anova(df, 'combined_column', target)
    elif all_normal and not homogeneous_variance:
        print("Performing Welch's ANOVA...")
        results = perform_welch_anova(df, 'combined_column', target)
    else:
        print("Performing Kruskal-Wallis H test...")
        results = perform_kruskal_wallis(df, 'combined_column', target)

    print("ANOVA/Kruskal-Wallis results:", results)

    # Perform post-hoc tests if necessary
    if len(groups) > 1 and any(group.nunique() > 1 for group in groups) and results['p_value'] < p_threshold:
        print("Performing post-hoc tests...")
        results.update(perform_post_hoc_tests(df, 'combined_column', target, results['test_type']))
    else:
        print("Skipping post-hoc tests. Not enough variability or insufficient groups for post-hoc analysis.")
        results.update({
            'post_hoc_test': None,
            'post_hoc_test_name': None,
            'post_hoc_summary': "Not enough variability or insufficient groups for post-hoc analysis."
        })

    print("Final results for group differences:", results)
    
    return results





def assess_feature_correlation(df: pd.DataFrame, column: str, target: str, p_threshold: float,
                              normality_test: str = 'shapiro') -> Dict[str, Any]:
    col_normal, _ = check_normality(df[column], test=normality_test, threshold=p_threshold)
    target_normal, _ = check_normality(df[target], test=normality_test, threshold=p_threshold)
    
    if col_normal and target_normal:
        correlation, p_value = stats.pearsonr(df[column], df[target])
        test_type = "Pearson correlation"
    else:
        correlation, p_value = stats.spearmanr(df[column], df[target])
        test_type = "Spearman rank correlation"
    
    return {
        "test_type": test_type,
        "correlation": correlation,
        "p_value": p_value
    }

def test_point_biserial(df: pd.DataFrame, column: str, target: str) -> Dict[str, Any]:
    if df[column].nunique() != 2:
        raise ValueError(f"Column '{column}' must be binary for point-biserial correlation.")
    
    correlation, p_value = stats.pointbiserialr(df[column], df[target])
    return {
        "test_type": "Point-biserial correlation",
        "correlation": correlation,
        "p_value": p_value
    }

def interpret_results(results: Dict[str, Any]) -> str:
    test_type = results["test_type"]
    p_value = results["p_value"]
    p_threshold = results["p_threshold"]
    
    if p_value < p_threshold:
        interpretation = f"Significant differences were found ({test_type}) with p-value = {p_value:.4f} (< {p_threshold}).\n"
    else:
        interpretation = f"No significant differences were found ({test_type}) with p-value = {p_value:.4f} (>= {p_threshold}).\n"
    
    if results.get('post_hoc_test') and results['post_hoc_test_name']:
        interpretation += f"\nPost-hoc analysis using {results['post_hoc_test_name']} was performed.\n"
        if results['post_hoc_summary']:
            significant_pairs = [f"{item['group1']} vs {item['group2']} (p-adj = {item['p-adj']:.4f})"
                                 for item in results['post_hoc_summary'] if item['p-adj'] < p_threshold]
            if significant_pairs:
                interpretation += "Significant differences were found between:\n" + "\n".join(significant_pairs)
            else:
                interpretation += "No significant differences were found between any specific groups."
        else:
            interpretation += "Post-hoc analysis results could not be generated."
    else:
        interpretation += "Post-hoc analysis was not required or could not be performed."
    
    return interpretation

def determine_variable_relationship(
    df: pd.DataFrame,
    column: str,
    target: str,
    column_types: Dict[str, str],
    column_mapping: Dict[str, List[str]],  # Added column_mapping parameter
    p_threshold: float,
    normality_test: str = 'shapiro',
    homogeneity_test: str = 'levene'
) -> Dict[str, Any]:
    """
    Determine the relationship between a column and a target variable, applying the appropriate statistical tests.
    """
    column_type = column_types[column]
    target_type = column_types[target]

    print(f"\nAnalyzing Column: {column}, Target: {target}")
    print(f"Column Type: {column_type.value}, Target Type: {target_type.value}")
    
    print("DataFrame columns before analyzing column:", df.columns.tolist())
    print(f"Column mapping for {column}: {column_mapping.get(column, 'No mapping found')}")
    
    results = {
        "column": column,
        "target": target,
        "column_type": column_type.value,
        "target_type": target_type.value,
        "p_threshold": p_threshold
    }
    
    if column_type in [DataType.NOMINAL, DataType.ORDINAL] and target_type == DataType.CONTINUOUS:
        print(f"Performing group differences test between {column} (categorical) and {target} (continuous).")
        dummy_columns = column_mapping[column]
        group_results = test_group_differences(df, dummy_columns, target, p_threshold, normality_test, homogeneity_test)
        results.update(group_results)
        results["visualizations"] = {f"{col}_visualization": path for col, path in group_results.items() if 'visualization' in col}

    elif column_type == DataType.CONTINUOUS and target_type == DataType.CONTINUOUS:
        print(f"Assessing feature correlation between {column} and {target} (both continuous).")
        results.update(assess_feature_correlation(df, column, target, p_threshold, normality_test))
        
    elif column_type in [DataType.NOMINAL, DataType.ORDINAL] and target_type in [DataType.NOMINAL, DataType.ORDINAL]:
        print(f"Performing Chi-square test between {column} and {target} (both categorical).")
        contingency_table = pd.crosstab(df[column], df[target])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        results.update({
            "test_type": "Chi-square",
            "chi2_statistic": chi2,
            "p_value": p_value,
            "degrees_of_freedom": dof
        })
        
    elif column_type == DataType.CONTINUOUS and target_type in [DataType.NOMINAL, DataType.ORDINAL, DataType.BINARY]:
        if target_type == DataType.BINARY:
            print(f"Performing Point-biserial correlation between {column} (continuous) and {target} (binary).")
            results.update(test_point_biserial(df, column, target))
        else:
            print(f"Performing one-way ANOVA with {column} (continuous) as dependent variable and {target} (categorical) as independent variable.")
            results.update(perform_anova(df, target, column))
    
    print(f"Results for {column}: {results}")
    results["interpretation"] = interpret_results(results)
    return results


def run_statistical_analysis(
    df: pd.DataFrame,
    columns: List[str],
    target: str,
    target_type: DataType,
    p_threshold: float = 0.05,
    output_path: str = "results/statistical_analysis",
    data_playtype: str = None,
    output_format: str = 'json',
    normality_test: str = 'shapiro',
    homogeneity_test: str = 'levene'
) -> Dict[str, Dict[str, Any]]:
    """
    Run statistical analysis on a DataFrame, given the columns to analyze and a target variable.
    """
    validated_target_type = set_target_type(target, target_type)
    
    column_types = identify_column_types(df)
    column_types.update(validated_target_type)
    
    df, column_mapping = preprocess_categorical_variables(df, column_types)
    
    print("DataFrame columns after preprocessing:", df.columns.tolist())
    
    results = {}
    visualizations = {}

    for column in columns:
        try:
            print(f"\nAnalyzing column: {column} with column_mapping: {column_mapping.get(column, 'No mapping found')}")
            
            results[column] = determine_variable_relationship(
                df, column, target, column_types, column_mapping, p_threshold, normality_test, homogeneity_test
            )
            
            column_visualizations = generate_visualizations(df, {column: results[column]}, column_mapping)
            visualizations.update(column_visualizations)
            
            column_visuals = {key: path for key, path in visualizations.items() if key.startswith(f"{column}_")}
            if column_visuals:
                results[column]["visualizations"] = column_visuals
                
            print(f"Updated results for {column}: {results[column]}")
        except Exception as e:
            results[column] = {"error": f"{type(e).__name__}: {str(e)}"}

    file_path = save_results(results, output_path, data_playtype, output_format)
    print(f"Results saved to: {file_path}")
    
    return results


if __name__ == "__main__":
    np.random.seed(42)
    
    sample_size = 1000
    df = pd.DataFrame({
        'feature1': np.random.rand(sample_size),
        'feature2': np.random.randint(0, 100, sample_size),
        'feature3': np.random.choice(['A', 'B', 'C'], sample_size),
        'feature4': np.random.choice(['Low', 'Medium', 'High'], sample_size),
        'target': np.random.randint(0, 3, sample_size)
    })

    results = run_statistical_analysis(
        df=df,
        columns=['feature1', 'feature2', 'feature3', 'feature4'],
        target='target',
        target_type=DataType.CONTINUOUS,
        p_threshold=0.05,
        output_path="results/example_run",
        data_playtype="sample_data",
        output_format='json',
        normality_test='shapiro',
        homogeneity_test='levene'
    )
    print("results: ", results)
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