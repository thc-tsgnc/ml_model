# src/data_processing/eda_process.py

import sys
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from typing import Dict, Tuple, List, Any

def preprocess_data(df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess the data for regression analysis."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def calculate_feature_variance(df: pd.DataFrame, target_column: str) -> float:
    """Calculate the average variance of features."""
    features = df.drop(columns=[target_column])
    return features.var().mean()

def run_regression(X: np.ndarray, y: np.ndarray, model_type: str, n_runs: int = 5, cv: int = 5) -> Dict[str, float]:
    """Run regression multiple times and return average performance metrics."""
    if model_type == 'linear':
        model = LinearRegression()
        scoring = 'r2'
    elif model_type == 'logistic':
        print("Running logistic regression")
        model = LogisticRegression()
        scoring = 'roc_auc'
    else:
        raise ValueError("Invalid model type. Choose 'linear' or 'logistic'.")

    all_scores = []
    for _ in range(n_runs):
        if model_type == 'logistic':
            model.set_params(random_state=np.random.randint(0, 10000))
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=np.random.randint(0, 10000))
        scores = cross_val_score(model, X, y, cv=cv_split, scoring=scoring)
        all_scores.extend(scores)

    return {
        f'mean_{scoring}': np.mean(all_scores),
        f'std_{scoring}': np.std(all_scores)
    }

def get_feature_importance(X: pd.DataFrame, y: pd.Series, model_type: str) -> pd.DataFrame:
    """Get feature importance based on regression coefficients."""
    if model_type == 'linear':
        model = LinearRegression()
    elif model_type == 'logistic':
        model = LogisticRegression(random_state=42)
    else:
        raise ValueError("Invalid model type. Choose 'linear' or 'logistic'.")
    
    model.fit(X, y)
    importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(model.coef_[0] if model_type == 'logistic' else model.coef_)
    })
    return importance.sort_values('importance', ascending=False)

def perform_eda_regression(df: pd.DataFrame, target_column: str, regression_type: str, n_runs: int = 5, cv: int = 5) -> Dict[str, Any]:
    """
    Perform EDA using the specified regression type.
    
    Args:
    df (pd.DataFrame): The input dataframe
    target_column (str): The name of the target column
    regression_type (str): The type of regression to perform ('linear' or 'logistic')
    n_runs (int): Number of times to run the regression (default: 5)
    cv (int): Number of cross-validation folds (default: 5)
    
    Returns:
    Dict[str, Any]: A dictionary containing regression results and feature importance
    """
    X, y = preprocess_data(df, target_column)
    print(f"regression_type", regression_type)
    
    if regression_type not in ['linear', 'logistic']:
        raise ValueError("Invalid regression_type. Choose 'linear' or 'logistic'.")
    
    results = run_regression(X, y, regression_type, n_runs, cv)
    feature_importance = get_feature_importance(df.drop(columns=[target_column]), y, regression_type)
    
    return {
        f'{regression_type}_regression': results,
        f'feature_importance_{regression_type}': feature_importance
    }

