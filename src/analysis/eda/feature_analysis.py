# src/analysis/eda/feature_analysis.py

import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.decomposition import PCA
from scipy.stats import pearsonr, spearmanr, kendalltau
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Dict, Any, Optional, List
from analysis.visualization.plot_functions import (
    plot_correlation_heatmap, plot_feature_importance_bar,
    plot_pca_scree, plot_pca_biplot, plot_vif_scores
)
import os 

def validate_input_data(X: pd.DataFrame, y: pd.Series) -> None:
    """Utility function to validate input data."""
    if X.isnull().values.any() or y.isnull().values.any():
        raise ValueError("Input data X and y should not have missing values.")
    if not np.issubdtype(X.dtypes.values[0], np.number) or not np.issubdtype(y.dtype, np.number):
        raise ValueError("Input data X and y must be numeric.")

def generate_visualizations(results: Dict[str, Any], output_dir: str = 'results/plots') -> Dict[str, str]:
    """
    Generate visualizations based on feature analysis results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    visualizations = {}

    plot_mapping = {
        'correlations': ('feature_target_correlations.png', plot_correlation_heatmap),
        'mutual_information': ('mutual_information_scores.png', plot_feature_importance_bar),
        'feature_importance': ('rf_feature_importance.png', plot_feature_importance_bar),
        'vif_scores': ('vif_scores_comparison.png', plot_vif_scores),
        'pca_scree': ('pca_scree_plot.png', plot_pca_scree),
        'pca_biplot': ('pca_biplot.png', plot_pca_biplot),
    }
    
    for key, (filename, plot_func) in plot_mapping.items():
        if key in results:
            vis_path = plot_func(results[key], filename=filename, output_dir=output_dir)
            visualizations[key] = vis_path
    
    return visualizations

def calculate_feature_target_correlation(X: pd.DataFrame, y: pd.Series, method: str = 'pearson') -> pd.Series:
    """Compute correlations between features and target variable."""
    validate_input_data(X, y)
    
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("Method must be 'pearson', 'spearman', or 'kendall'")
    validate_input_data(X, y)
    
    corr_func = {'pearson': pearsonr, 'spearman': spearmanr, 'kendall': kendalltau}[method]
    correlations = X.apply(lambda col: corr_func(col, y)[0])
    correlation_df = pd.DataFrame(correlations, columns=['Correlation']).T
    return correlation_df

def calculate_mutual_information(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calculate mutual information between features and target."""
    validate_input_data(X, y)
    
    # Ensure y is correctly handled as either numeric or categorical
    if pd.api.types.is_numeric_dtype(y):
        # For numeric targets, use regression
        mi_func = mutual_info_regression
        print("Using mutual_info_regression for numeric target")
    else:
        # For categorical targets, use classification
        mi_func = mutual_info_classif
        print("Using mutual_info_classif for categorical target")
    
    # Debugging: Print first few rows of X and check if target is numeric or categorical
    # print(f"Data type of target (y): {y.dtype}")
    # print("First few rows of X:")
    # print(X.head())

    # Calculate mutual information scores
    mi_scores = mi_func(X, y)

    # Return sorted mutual information scores as a pandas Series
    return pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)


def compute_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Compute feature importance using Random Forest."""
    validate_input_data(X, y)

    model = RandomForestRegressor(n_estimators=100, random_state=42) if pd.api.types.is_numeric_dtype(y) else RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    return pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

def reduce_multicollinearity(X: pd.DataFrame, threshold: float = 10) -> pd.DataFrame:
    """Reduce multicollinearity using Variance Inflation Factor (VIF)."""
    validate_input_data(X, X.iloc[:, 0])
    
    features = X.columns.tolist()
    while True:
        vif = pd.DataFrame({
            'feature': features,
            'VIF': [variance_inflation_factor(X[features].values, i) for i in range(len(features))]
        })
        max_vif = vif['VIF'].max()
        if max_vif > threshold:
            exclude_feature = vif.loc[vif['VIF'].idxmax(), 'feature']
            features.remove(exclude_feature)
            print(f"Dropped {exclude_feature} with VIF {max_vif:.2f}")
        else:
            break
    return X[features]

def perform_pca(X: pd.DataFrame, n_components: float = 0.95) -> Dict[str, Any]:
    """Perform PCA on the dataset."""
    validate_input_data(X, X.iloc[:, 0])

    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)
    columns = [f'PC{i+1}' for i in range(pca_result.shape[1])]
    pca_df = pd.DataFrame(pca_result, columns=columns)
    explained_variance_ratio = pd.Series(pca.explained_variance_ratio_, index=columns)

    return {'pca_df': pca_df, 'explained_variance_ratio': explained_variance_ratio}

def run_feature_analysis(X: pd.DataFrame, y: pd.Series, analyses: List[str] = None,
                         generate_vis: bool = False, output_dir: str = 'results/plots') -> Dict[str, Any]:
    """Run selected feature analysis steps and optionally generate visualizations."""
    
    validate_input_data(X, y)
    
    if analyses is None:
        analyses = ['correlation', 'mutual_info', 'importance', 'multicollinearity', 'pca']
    
    results = {}

    for analysis in analyses:
        # Added print statements for progress
        print(f"Starting {analysis} analysis...")
        if analysis == 'correlation':
            results['correlations'] = calculate_feature_target_correlation(X, y)
        elif analysis == 'mutual_info':
            results['mutual_information'] = calculate_mutual_information(X, y)
        elif analysis == 'importance':
            results['feature_importance'] = compute_feature_importance(X, y)
        elif analysis == 'multicollinearity':
            results['reduced_data'] = reduce_multicollinearity(X)
        elif analysis == 'pca':
            X_for_pca = results.get('reduced_data', X)
            pca_results = perform_pca(X_for_pca)
            results['pca_df'] = pca_results['pca_df']
            results['pca_explained_variance'] = pca_results['explained_variance_ratio']
        print(f"Completed {analysis} analysis.")

    if generate_vis:
        print("Generating visualizations...")
        results['visualizations'] = generate_visualizations(results, output_dir)
        print("Visualizations generated.")
    
    return results

# Example usage
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from sklearn.datasets import make_regression

    # Create a synthetic dataset
    X, y = make_regression(n_samples=1000, n_features=5, noise=0.1, random_state=42)
    feature_names = [f'feature_{i}' for i in range(5)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # Run feature analysis
    X = df[feature_names]
    y = df['target']
    
    print("Running feature analysis...")
    results = run_feature_analysis(X, y, analyses=['correlation', 'mutual_info', 'importance', 'pca'], generate_vis=True)

    # Print results
    print("\nAnalysis Results:")
    for analysis, result in results.items():
        if analysis == 'visualizations':
            print(f"\n{analysis.capitalize()}:")
            for vis_type, path in result.items():
                print(f"  {vis_type}: {path}")
        elif isinstance(result, pd.Series):
            print(f"\n{analysis.capitalize()}:")
            print(result)
        elif isinstance(result, dict):
            print(f"\n{analysis.capitalize()}:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        else:
            print(f"\n{analysis.capitalize()}: {result}")

    print("\nFeature analysis completed successfully.")
