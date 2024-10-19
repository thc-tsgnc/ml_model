# src/modeling/clustering_transformers.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import NotFittedError
from sklearn.datasets import make_blobs

class BaseClusteringTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.is_fitted = False

    def check_is_fitted(self):
        if not self.is_fitted:
            raise NotFittedError(f"This {self.__class__.__name__} instance is not fitted yet. Call 'fit' before using this estimator.")

    def _validate_input(self, X):
        if not isinstance(X, (np.ndarray, pd.DataFrame)):
            raise ValueError("Input X must be a numpy array or pandas DataFrame.")
        if X.ndim != 2:
            raise ValueError("Input X must be a 2D array or DataFrame with features.")

class KMeansTransformer(BaseClusteringTransformer):
    def __init__(self, n_clusters=5, random_state=42):
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)

    def fit(self, X, y=None):
        self._validate_input(X)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X):
        self.check_is_fitted()
        self._validate_input(X)
        X_transformed = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        X_transformed['cluster_label'] = self.model.predict(X)
        return X_transformed

class DBSCANTransformer(BaseClusteringTransformer):
    def __init__(self, eps=0.5, min_samples=5):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)

    def fit(self, X, y=None):
        self._validate_input(X)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X):
        self.check_is_fitted()
        self._validate_input(X)
        X_transformed = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        X_transformed['cluster_label'] = self.model.fit_predict(X)  # Changed from self.model.labels_
        return X_transformed

class GMMTransformer(BaseClusteringTransformer):
    def __init__(self, n_components=3, random_state=42):
        super().__init__()
        self.n_components = n_components
        self.random_state = random_state
        self.model = GaussianMixture(n_components=self.n_components, random_state=self.random_state)

    def fit(self, X, y=None):
        self._validate_input(X)
        self.model.fit(X)
        self.is_fitted = True
        return self

    def transform(self, X):
        self.check_is_fitted()
        self._validate_input(X)
        X_transformed = pd.DataFrame(X) if isinstance(X, np.ndarray) else X.copy()
        X_transformed['cluster_label'] = self.model.predict(X)
        return X_transformed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

if __name__ == "__main__":
    # Generate sample data
    X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    X_df = pd.DataFrame(X, columns=['feature1', 'feature2'])

    # Test pipeline with KMeansTransformer
    print("Testing pipeline with KMeansTransformer:")
    kmeans_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Example preprocessing step
        ('kmeans', KMeansTransformer(n_clusters=4))
    ])
    X_kmeans_pipeline = kmeans_pipeline.fit_transform(X_df)
    print(X_kmeans_pipeline.head())
    print(f"Unique cluster labels: {X_kmeans_pipeline['cluster_label'].unique()}\n")

    # Test pipeline with DBSCANTransformer
    print("Testing pipeline with DBSCANTransformer:")
    dbscan_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Example preprocessing step
        ('dbscan', DBSCANTransformer(eps=0.3, min_samples=5))
    ])
    X_dbscan_pipeline = dbscan_pipeline.fit_transform(X_df)
    print(X_dbscan_pipeline.head())
    print(f"Unique cluster labels: {X_dbscan_pipeline['cluster_label'].unique()}\n")

    # Test pipeline with GMMTransformer
    print("Testing pipeline with GMMTransformer:")
    gmm_pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Example preprocessing step
        ('gmm', GMMTransformer(n_components=4))
    ])
    X_gmm_pipeline = gmm_pipeline.fit_transform(X_df)
    print(X_gmm_pipeline.head())
    print(f"Unique cluster labels: {X_gmm_pipeline['cluster_label'].unique()}\n")

    # Error Handling remains the same as before
    print("Testing error handling:")
    try:
        kmeans_transformer = KMeansTransformer()
        kmeans_transformer.transform(X_df)
    except NotFittedError as e:
        print(f"Caught expected error (not fitted): {e}")

    try:
        print("Testing with invalid input [1, 2, 3]:")
        kmeans_transformer.fit([1, 2, 3])
    except ValueError as e:
        print(f"Caught expected error (non-2D input): {e}")

    try:
        print("Testing with 1D NumPy array:")
        kmeans_transformer.fit(np.array([1, 2, 3]))
    except ValueError as e:
        print(f"Caught expected error (1D NumPy array): {e}")
