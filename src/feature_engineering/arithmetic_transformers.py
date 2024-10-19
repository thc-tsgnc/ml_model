# src/feature_engineering/arithmetic_transformers.py

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from typing import List, Tuple

class InputValidationMixin:
    def _validate_input(self, X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        if X.empty:
            raise ValueError("Input DataFrame is empty")
        for col1, col2 in self.column_pairs:
            if col1 not in X.columns or col2 not in X.columns:
                raise ValueError(f"Columns {col1} or {col2} not found in the input DataFrame")
            if not np.issubdtype(X[col1].dtype, np.number) or not np.issubdtype(X[col2].dtype, np.number):
                raise ValueError(f"Columns {col1} and {col2} must contain numeric data")

class ArithmeticDifferenceTransformer(BaseEstimator, TransformerMixin, InputValidationMixin):
    def __init__(self, column_pairs: List[Tuple[str, str]]):
        self.column_pairs = column_pairs

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        diff_columns = {f"{col1}_{col2}_diff": X[col1] - X[col2] for col1, col2 in self.column_pairs}
        return X.assign(**diff_columns)

class ArithmeticRatioTransformer(BaseEstimator, TransformerMixin, InputValidationMixin):
    def __init__(self, column_pairs: List[Tuple[str, str]], fill_value: float = np.nan):
        self.column_pairs = column_pairs
        self.fill_value = fill_value

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_input(X)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        ratio_columns = {}
        for numerator, denominator in self.column_pairs:
            ratio = X[numerator] / X[denominator]
            ratio_columns[f"{numerator}_{denominator}_rto"] = ratio.fillna(self.fill_value)
        return X.assign(**ratio_columns)

# Test functions
def test_input_validation(df):
    try:
        invalid_transformer = ArithmeticDifferenceTransformer(
            column_pairs=[('value_a', 'non_numeric')]
        )
        invalid_transformer.fit(df)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert "must contain numeric data" in str(e)

    try:
        empty_df = pd.DataFrame()
        diff_transformer = ArithmeticDifferenceTransformer(column_pairs=[('value_a', 'value_b')])
        diff_transformer.fit(empty_df)
        assert False, "Expected ValueError was not raised"
    except ValueError as e:
        assert "Input DataFrame is empty" in str(e)

def test_nan_handling(df_diff, df_ratio):
    assert np.isnan(df_diff['value_a_value_b_diff'].iloc[4])
    assert df_ratio['value_a_value_b_rto'].iloc[4] == -999

def test_zero_division_handling(df_ratio):
    assert df_ratio['value_a_value_b_rto'].iloc[2] == -999
    assert df_ratio['value_a_value_b_rto'].iloc[3] == -999

def test_efficiency():
    import time
    large_df = pd.DataFrame(np.random.rand(100000, 2), columns=['A', 'B'])
    
    start_time = time.time()
    diff_transformer = ArithmeticDifferenceTransformer(column_pairs=[('A', 'B')])
    diff_transformer.fit_transform(large_df)
    diff_time = time.time() - start_time
    
    start_time = time.time()
    ratio_transformer = ArithmeticRatioTransformer(column_pairs=[('A', 'B')])
    ratio_transformer.fit_transform(large_df)
    ratio_time = time.time() - start_time
    
    print(f"Time taken for difference transformation: {diff_time:.4f} seconds")
    print(f"Time taken for ratio transformation: {ratio_time:.4f} seconds")
    
    assert diff_time < 1, "Difference transformation took too long"
    assert ratio_time < 1, "Ratio transformation took too long"

if __name__ == "__main__":
    # Create sample data
    data = {
        'value_a': [10, 20, 30, 40, np.nan],
        'value_b': [5, 10, 0, 0, 5],
        'value_c': [1, 2, 3, np.nan, 5],
        'non_numeric': ['a', 'b', 'c', 'd', 'e']
    }
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    # Test ArithmeticDifferenceTransformer
    diff_transformer = ArithmeticDifferenceTransformer(
        column_pairs=[('value_a', 'value_b'), ('value_b', 'value_c')]
    )
    df_diff = diff_transformer.fit_transform(df)
    print("\nAfter Difference Transformation:")
    print(df_diff)

    # Test ArithmeticRatioTransformer
    ratio_transformer = ArithmeticRatioTransformer(
        column_pairs=[('value_a', 'value_b'), ('value_b', 'value_c')],
        fill_value=-999  # Use -999 for divide-by-zero and NaN cases
    )
    df_ratio = ratio_transformer.fit_transform(df_diff)
    print("\nAfter Ratio Transformation:")
    print(df_ratio)

    # Run tests
    print("\nRunning tests...")
    try:
        test_input_validation(df)
        print("Input validation test passed")
        test_nan_handling(df_diff, df_ratio)
        print("NaN handling test passed")
        test_zero_division_handling(df_ratio)
        print("Zero division handling test passed")
        test_efficiency()
        print("Efficiency test passed")
        print("\nAll tests passed successfully!")
    except AssertionError as e:
        print(f"Test failed: {e}")