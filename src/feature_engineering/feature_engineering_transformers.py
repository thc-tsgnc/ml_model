# src/feature_engineering/feature_engineering_transformers.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import PolynomialFeatures as SKPolynomialFeatures
from sklearn.feature_selection import VarianceThreshold
from boruta import BorutaPy


class ExtendedInputValidation:
    def _validate_input(self, X, y=None):
        if not isinstance(X, (pd.DataFrame, np.ndarray)):
            raise ValueError("X must be a pandas DataFrame or a NumPy array")
        if X.shape[0] == 0:
            raise ValueError("X cannot be empty")
        if isinstance(X, pd.DataFrame):
            # Apply np.issubdtype column-wise
            if not X.dtypes.apply(lambda dtype: np.issubdtype(dtype, np.number)).all():
                raise ValueError("X contains non-numeric data. All columns must be numeric.")
        else:
            if not np.issubdtype(X.dtype, np.number):
                raise ValueError("X must contain only numeric values.")
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same number of samples.")


class BorutaFeatureSelector(BaseEstimator, TransformerMixin, ExtendedInputValidation):
    def __init__(self, estimator=None, n_estimators=100, max_iter=100, random_state=None):
        self.estimator = estimator
        self.n_estimators = n_estimators
        self.max_iter = max_iter
        self.random_state = random_state
        self.selector = None
        self.feature_names_ = None

    def fit(self, X, y):
        self._validate_input(X, y)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        y_np = y.values if isinstance(y, pd.Series) else y
        
        # Instantiate estimator only if it is not provided
        if self.estimator is None:
            self.estimator = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        
        self.selector = BorutaPy(
            estimator=self.estimator,
            n_estimators='auto',
            max_iter=self.max_iter,
            random_state=self.random_state
        )
        
        self.selector.fit(X_np, y_np)
        self.feature_names_ = X.columns[self.selector.support_].tolist() if isinstance(X, pd.DataFrame) else None
        
        return self

    def transform(self, X):
        self._validate_input(X)
        X_np = X.values if isinstance(X, pd.DataFrame) else X
        X_selected = self.selector.transform(X_np)
        
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_selected, columns=self.feature_names_, index=X.index)
        return X_selected

class CustomPolynomialFeatures(BaseEstimator, TransformerMixin, ExtendedInputValidation):
    def __init__(self, degree=2, interaction_only=True, include_bias=False):
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias
        self.poly = SKPolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only, include_bias=self.include_bias)
        self.feature_names_ = None

    def fit(self, X, y=None):
        self._validate_input(X)
        self.poly.fit(X)
        if isinstance(X, pd.DataFrame):
            # Get feature names and replace spaces with underscores
            self.feature_names_ = [name.replace(" ", "_") for name in self.poly.get_feature_names_out(X.columns)]
        return self

    def transform(self, X):
        self._validate_input(X)
        X_poly = self.poly.transform(X)
        if isinstance(X, pd.DataFrame):
            # Replace spaces in feature names with underscores
            return pd.DataFrame(X_poly, columns=self.feature_names_, index=X.index)
        return X_poly

class CustomVarianceThreshold(BaseEstimator, TransformerMixin, ExtendedInputValidation):
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.var_selector = VarianceThreshold(threshold=self.threshold)
        self.feature_names_ = None

    def fit(self, X, y=None):
        self._validate_input(X)
        self.var_selector.fit(X)
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns[self.var_selector.get_support()].tolist()
        return self

    def transform(self, X):
        self._validate_input(X)
        X_transformed = self.var_selector.transform(X)
        if isinstance(X, pd.DataFrame):
            return pd.DataFrame(X_transformed, columns=self.feature_names_, index=X.index)
        return X_transformed

class FeatureImportanceReducer(BaseEstimator, TransformerMixin, ExtendedInputValidation):
    def __init__(self, estimator=None, threshold=0.01, top_n=None, random_state=None):
        self.estimator = estimator  # Delay initialization of estimator
        self.threshold = threshold
        self.top_n = top_n
        self.feature_names_ = None
        self.random_state = random_state  # Store the random_state for consistency

    def fit(self, X, y):
        self._validate_input(X, y)
        
        # Instantiate the estimator if not provided, passing the random_state
        if self.estimator is None:
            self.estimator = RandomForestClassifier(random_state=self.random_state)
        
        # Fit the estimator to the data
        self.estimator.fit(X, y)
        
        # Retrieve feature importances from the fitted estimator
        importances = self.estimator.feature_importances_
        
        # Select the top features based on importance or threshold
        if self.top_n:
            indices = np.argsort(importances)[::-1][:self.top_n]
        else:
            indices = np.where(importances >= self.threshold)[0]
        
        # Store the feature names if working with a DataFrame
        self.feature_names_ = X.columns[indices].tolist() if isinstance(X, pd.DataFrame) else indices
        return self

    def transform(self, X):
        self._validate_input(X)
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names_]
        return X[:, self.feature_names_]


if __name__ == "__main__":
    # Test code for the transformers
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, 
                               n_repeated=0, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y, name='target')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    # Test BorutaFeatureSelector
    print("\nTesting BorutaFeatureSelector...")
    bfs = BorutaFeatureSelector(max_iter=50, random_state=42)
    X_train_boruta = bfs.fit_transform(X_train, y_train)
    print(f"Original features: {X_train.shape[1]}, Selected features: {X_train_boruta.shape[1]}")
    print("Selected feature names:", X_train_boruta.columns.tolist())
    
    # Test CustomPolynomialFeatures
    print("Testing CustomPolynomialFeatures...")
    cpf = CustomPolynomialFeatures(degree=2, interaction_only=True)
    X_train_poly = cpf.fit_transform(X_train)
    print(f"Original features: {X_train.shape[1]}, Polynomial features: {X_train_poly.shape[1]}")
    print("First few polynomial feature names:", X_train_poly.columns[:5].tolist())
    # print("Polynomial feature names:", X_train_poly.columns.tolist())
    
    # Test FeatureImportanceReducer
    print("\nTesting FeatureImportanceReducer...")
    fir = FeatureImportanceReducer(estimator=RandomForestClassifier(n_estimators=100, random_state=42), top_n=15)
    X_train_reduced = fir.fit_transform(X_train_poly, y_train)
    print(f"Features after polynomial: {X_train_poly.shape[1]}, Selected features: {X_train_reduced.shape[1]}")
    print("Selected feature names:", X_train_reduced.columns.tolist())

    

    
    # Test CustomVarianceThreshold
    print("\nTesting CustomVarianceThreshold...")
    cvt = CustomVarianceThreshold(threshold=0.1)
    X_train_var = cvt.fit_transform(X_train_reduced)
    print(f"Features after importance reduction: {X_train_reduced.shape[1]}, Features after variance threshold: {X_train_var.shape[1]}")
    print("Final feature names:", X_train_var.columns.tolist())

