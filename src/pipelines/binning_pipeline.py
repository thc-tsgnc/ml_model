import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import KBinsDiscretizer
import warnings

class BinningTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, column, n_bins, strategy='quantile', random_state=42, add_suffix=True):
        self.column = column
        self.n_bins = n_bins
        self.strategy = strategy
        self.random_state = random_state
        self.add_suffix = add_suffix
        self.binner = None

    def fit(self, X, y=None):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in the input data")
        if not pd.api.types.is_numeric_dtype(X[self.column]):
            raise ValueError(f"Column '{self.column}' is not numeric")
        
        X_column = X[self.column]
        
        unique_values = X_column.nunique()

        # Skip the warning if strategy is 'quantile' because it can handle fewer unique values
        if unique_values < self.n_bins and self.strategy != 'quantile':
            warnings.warn(f"Column '{self.column}' has only {unique_values} unique values, reducing the number of bins to {unique_values}")
            self.n_bins = unique_values
        
        self.binner = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy=self.strategy, random_state=self.random_state)
        self.binner.fit(X_column.values.reshape(-1, 1))
        return self

    def transform(self, X):
        if self.column not in X.columns:
            raise ValueError(f"Column '{self.column}' not found in the input data")
        
        X_ = X.copy()
        # Note: Missing value handling is assumed to be done in the data cleaning pipeline
        # X_column = X_[self.column].fillna(X_[self.column].median())
        X_column = X_[self.column]
        
        new_col_name = f'{self.column}_binned' if self.add_suffix else self.column
        X_[new_col_name] = self.binner.transform(X_column.values.reshape(-1, 1)).ravel()
        return X_

def create_binning_transformer(column, n_bins, strategy='quantile', random_state=42, add_suffix=True):
    return BinningTransformer(column=column, n_bins=n_bins, strategy=strategy, random_state=random_state, add_suffix=add_suffix)

# Example usage and test cases
if __name__ == "__main__":
    import numpy as np

    # Create a sample dataset
    np.random.seed(42)
    df = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.randint(0, 10, 100),
        'feature3': [1, 1, 1, 2] * 25,
        'non_numeric': ['A', 'B', 'C'] * 33 + ['A']
    })

    # Test case 1: Normal scenario
    binning_transformer = create_binning_transformer(column='feature1', n_bins=5, strategy='quantile')
    df_binned = binning_transformer.fit_transform(df)
    assert 'feature1_binned' in df_binned.columns, "feature1 should be binned"
    assert df_binned['feature1_binned'].nunique() <= 5, "feature1 should have at most 5 bins"

    # Test case 2: Column with fewer unique values than bins for 'uniform' strategy
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # Ensure all warnings are caught
        binning_transformer = create_binning_transformer(column='feature3', n_bins=5, strategy='uniform')
        df_binned = binning_transformer.fit_transform(df)
        assert len(w) == 1, "Should raise a warning"
        assert "reducing the number of bins" in str(w[-1].message)
    assert df_binned['feature3_binned'].nunique() == 2, "feature3 should have only 2 bins"

    # Test case 3: Non-numeric column
    try:
        binning_transformer = create_binning_transformer(column='non_numeric', n_bins=3)
        binning_transformer.fit(df)
    except ValueError as e:
        assert "is not numeric" in str(e)

    # Test case 4: Column not in dataframe
    try:
        binning_transformer = create_binning_transformer(column='non_existent', n_bins=3)
        binning_transformer.fit(df)
    except ValueError as e:
        assert "not found in the input data" in str(e)

    # Test case 5: Overwrite original column
    binning_transformer = create_binning_transformer(column='feature2', n_bins=3, add_suffix=False)
    df_binned = binning_transformer.fit_transform(df)
    assert 'feature2_binned' not in df_binned.columns, "Should not create a new column"
    assert df_binned['feature2'].nunique() <= 3, "feature2 should be binned into at most 3 categories"

    print("All test cases passed successfully!")