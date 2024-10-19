import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple, Dict, Any

def apply_laplace_smoothing(value: float, count: float, smoothing_factor: float) -> float:
    return (value * count + smoothing_factor) / (count + 2 * smoothing_factor)

def normalize_probabilities(prob1: float, prob2: float) -> Tuple[float, float]:
    total = prob1 + prob2
    if total == 0:
        return 0, 0
    return prob1 / total, prob2 / total

def calculate_log5(prob1: float, prob2: float) -> Tuple[float, float]:
    log5_prob1 = (prob1 - prob1 * prob2) / (prob1 + prob2 - 2 * prob1 * prob2)
    return log5_prob1, 1 - log5_prob1

class ExtendedInputValidation:
    def _validate_input(self, X: pd.DataFrame, y=None) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        if not X.dtypes.apply(lambda dtype: np.issubdtype(dtype, np.number)).all():
            raise ValueError("X contains non-numeric data. All columns must be numeric.")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")

class LaplaceSmoothingTransformer(BaseEstimator, TransformerMixin, ExtendedInputValidation):
    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def fit(self, X: pd.DataFrame, y=None) -> 'LaplaceSmoothingTransformer':
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        X = X.copy()
        for col_pair, params in self.config.items():
            col1, col2 = col_pair
            count_col = params['count_column']
            smoothing_factor = params.get('smoothing_factor', 1)
            threshold = params.get('threshold')
            
            if threshold is not None:
                mask = X[count_col] < threshold
            else:
                mask = X[[col1, col2]].isin([0, 1]).any(axis=1)
            
            for col in [col1, col2]:
                X.loc[mask, col] = X.loc[mask, [col, count_col]].apply(
                    lambda row: apply_laplace_smoothing(row[col], row[count_col], smoothing_factor),
                    axis=1
                )
        
        return X

class ProbabilityNormalizationTransformer(BaseEstimator, TransformerMixin, ExtendedInputValidation):
    def __init__(self, column_pairs: List[Tuple[str, str]]):
        self.column_pairs = column_pairs

    def fit(self, X: pd.DataFrame, y=None) -> 'ProbabilityNormalizationTransformer':
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        X = X.copy()
        for col1, col2 in self.column_pairs:
            X[[f"{col1}_norm", f"{col2}_norm"]] = X[[col1, col2]].apply(
                lambda row: normalize_probabilities(row[col1], row[col2]),
                axis=1,
                result_type='expand'
            )
        return X

class Log5ProbabilityTransformer(BaseEstimator, TransformerMixin, ExtendedInputValidation):
    def __init__(self, column_pairs: List[Tuple[str, str]]):
        self.column_pairs = column_pairs

    def fit(self, X: pd.DataFrame, y=None) -> 'Log5ProbabilityTransformer':
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        X = X.copy()
        for col1, col2 in self.column_pairs:
            X[[f"{col1}_log5", f"{col2}_log5"]] = X[[col1, col2]].apply(
                lambda row: calculate_log5(row[col1], row[col2]),
                axis=1,
                result_type='expand'
            )
        return X

# Example usage and testing
if __name__ == "__main__":
    # Create sample data with multiple column pairs and edge cases
    data = {
        't1_player_o_rolling_tip_avg': [0.1, 0.2, 0.3, 0.4, 0.5, 0, 1, 0.5],
        't2_player_o_rolling_tip_avg': [0.9, 0.8, 0.7, 0.6, 0.5, 1, 0, 0.5],
        't1_o_rolling_fts_avg': [0.6, 0.7, 0.8, 0.9, 0.5, 0.2, 0.3, 0],
        't2_o_rolling_fts_avg': [0.4, 0.3, 0.2, 0.1, 0.5, 0.8, 0.7, 0],
        't1_player_h_rolling_tip_avg': [0.2, 0.3, 0.4, 0.5, 0.6, 0.1, 0.9, 0.5],
        't2_player_a_rolling_tip_avg': [0.8, 0.7, 0.6, 0.5, 0.4, 0.9, 0.1, 0.5],
        'count1': [5, 10, 15, 20, 25, 2, 30, 0],
        'count2': [8, 12, 18, 22, 28, 3, 35, 1]
    }
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    # LaplaceSmoothingTransformer
    laplace_config = {
        ('t1_player_o_rolling_tip_avg', 't2_player_o_rolling_tip_avg'): {'count_column': 'count1', 'threshold': 12, 'smoothing_factor': 1},
        ('t1_o_rolling_fts_avg', 't2_o_rolling_fts_avg'): {'count_column': 'count2', 'threshold': 15, 'smoothing_factor': 0.5},
        ('t1_player_h_rolling_tip_avg', 't2_player_a_rolling_tip_avg'): {'count_column': 'count1', 'smoothing_factor': 2}
    }
    laplace = LaplaceSmoothingTransformer(config=laplace_config)
    df_laplace = laplace.fit_transform(df)
    print("\nAfter Laplace Smoothing:")
    print(df_laplace)

    # ProbabilityNormalizationTransformer
    normalize = ProbabilityNormalizationTransformer(
        column_pairs=[
            ('t1_player_o_rolling_tip_avg', 't2_player_o_rolling_tip_avg'),
            ('t1_o_rolling_fts_avg', 't2_o_rolling_fts_avg'),
            ('t1_player_h_rolling_tip_avg', 't2_player_a_rolling_tip_avg')
        ]
    )
    df_normalized = normalize.fit_transform(df_laplace)
    print("\nAfter Normalization:")
    print(df_normalized)

    # Log5ProbabilityTransformer
    log5 = Log5ProbabilityTransformer(
        column_pairs=[
            ('t1_player_o_rolling_tip_avg', 't2_player_o_rolling_tip_avg'),
            ('t1_o_rolling_fts_avg', 't2_o_rolling_fts_avg'),
            ('t1_player_h_rolling_tip_avg', 't2_player_a_rolling_tip_avg')
        ]
    )
    df_log5 = log5.fit_transform(df_normalized)
    print("\nAfter Log5 Transformation:")
    print(df_log5)

    # Verify results
    print("\nVerification:")
    print("1. Check if Laplace smoothing was applied to low count or extreme probability cases")
    print("2. Verify that normalized probabilities sum to 1 for each pair")
    print("3. Confirm Log5 probabilities are calculated correctly")
    print("4. Ensure all edge cases (0, 1, and 0-0 pairs) are handled properly")

    # You can add more specific checks here if needed