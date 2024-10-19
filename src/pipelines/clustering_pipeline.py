# src/pipelines/clustering_pipeline.py

import logging
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from pipelines.data_clean_pipeline import DataCleanPipeline
from modeling.clustering_transformers import KMeansTransformer, DBSCANTransformer, GMMTransformer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ClusteringPipeline:
    """
    A class to create and manage clustering pipelines.
    """

    def __init__(self, 
                 numeric_features, 
                 categorical_features, 
                 clustering_algorithm='kmeans',
                 advanced_cleaning=False,
                 numeric_impute_strategy='mean',
                 categorical_impute_strategy='most_frequent',
                 **clustering_params):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.clustering_algorithm = clustering_algorithm.lower()
        self.advanced_cleaning = advanced_cleaning
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.clustering_params = clustering_params

        self._validate_params()
        self.pipeline = self._create_pipeline()

    def _validate_params(self):
        valid_algorithms = ['kmeans', 'dbscan', 'gmm']
        if self.clustering_algorithm not in valid_algorithms:
            raise ValueError(f"Unsupported clustering algorithm: {self.clustering_algorithm}. "
                             f"Valid options are: {', '.join(valid_algorithms)}")

        if self.clustering_algorithm == 'kmeans' and 'n_clusters' not in self.clustering_params:
            raise ValueError("KMeans requires 'n_clusters' parameter.")
        
        if self.clustering_algorithm == 'gmm' and 'n_components' not in self.clustering_params:
            raise ValueError("GMM requires 'n_components' parameter.")
        
        if self.clustering_algorithm == 'dbscan':
            if 'eps' not in self.clustering_params or 'min_samples' not in self.clustering_params:
                raise ValueError("DBSCAN requires both 'eps' and 'min_samples' parameters.")

    def _create_pipeline(self):
        # Create the data cleaning pipeline
        data_cleaner = DataCleanPipeline(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            advanced=self.advanced_cleaning,
            numeric_impute_strategy=self.numeric_impute_strategy,
            categorical_impute_strategy=self.categorical_impute_strategy
        )

        # Create the clustering transformer based on the chosen algorithm
        if self.clustering_algorithm == 'kmeans':
            clusterer = KMeansTransformer(**self.clustering_params)
        elif self.clustering_algorithm == 'dbscan':
            clusterer = DBSCANTransformer(**self.clustering_params)
        elif self.clustering_algorithm == 'gmm':
            clusterer = GMMTransformer(**self.clustering_params)

        # Combine the data cleaning and clustering steps into a single pipeline
        pipeline = Pipeline([
            ('data_cleaning', data_cleaner.pipeline),
            ('clusterer', clusterer)
        ])

        return pipeline

    def fit_transform(self, X):
        """
        Fit the pipeline to the data and transform it.
        """
        logging.info("Fitting and transforming the data using the clustering pipeline.")
        transformed = self.pipeline.fit_transform(X)
        return pd.concat([X.reset_index(drop=True), transformed[['cluster_label']]], axis=1)

    def transform(self, X):
        """
        Transform the data using the fitted pipeline.
        """
        logging.info("Transforming the data using the fitted clustering pipeline.")
        transformed = self.pipeline.transform(X)
        return pd.concat([X.reset_index(drop=True), transformed[['cluster_label']]], axis=1)

def create_clustering_pipeline(numeric_features, 
                               categorical_features, 
                               clustering_algorithm='kmeans',
                               advanced_cleaning=False,
                               numeric_impute_strategy='mean',
                               categorical_impute_strategy='most_frequent',
                               **clustering_params):
    """
    Factory function to create a clustering pipeline.

    Parameters:
    - numeric_features (list): List of numeric feature column names
    - categorical_features (list): List of categorical feature column names
    - clustering_algorithm (str): The clustering algorithm to use ('kmeans', 'dbscan', or 'gmm')
    - advanced_cleaning (bool): Whether to use advanced cleaning (with scaling and encoding)
    - numeric_impute_strategy (str): Strategy for imputing numeric values
    - categorical_impute_strategy (str): Strategy for imputing categorical values
    - **clustering_params: Additional parameters for the chosen clustering algorithm

    Returns:
    - ClusteringPipeline: An instance of the ClusteringPipeline class
    """
    return ClusteringPipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        clustering_algorithm=clustering_algorithm,
        advanced_cleaning=advanced_cleaning,
        numeric_impute_strategy=numeric_impute_strategy,
        categorical_impute_strategy=categorical_impute_strategy,
        **clustering_params
    )

# Example usage
if __name__ == "__main__":
    # Create a sample dataset
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.randint(20000, 200000, 1000),
        'gender': np.random.choice(['M', 'F', None], 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', None], 1000)
    }
    df = pd.DataFrame(data)

    # Define feature lists
    numeric_features = ['age', 'income']
    categorical_features = ['gender', 'education']

    # Test KMeans clustering
    logging.info("\nTesting KMeans clustering:")
    kmeans_pipeline = create_clustering_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        clustering_algorithm='kmeans',
        advanced_cleaning=True,
        n_clusters=5
    )
    df_kmeans = kmeans_pipeline.fit_transform(df)
    logging.info(f"\nKMeans clustered data (first 5 rows):\n{df_kmeans.head()}")
    logging.info(f"Unique KMeans cluster labels: {df_kmeans['cluster_label'].unique()}")

    # Test DBSCAN clustering
    logging.info("\nTesting DBSCAN clustering:")
    dbscan_pipeline = create_clustering_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        clustering_algorithm='dbscan',
        advanced_cleaning=True,
        eps=0.5,
        min_samples=5
    )
    df_dbscan = dbscan_pipeline.fit_transform(df)
    logging.info(f"\nDBSCAN clustered data (first 5 rows):\n{df_dbscan.head()}")
    logging.info(f"Unique DBSCAN cluster labels: {df_dbscan['cluster_label'].unique()}")

    # Test GMM clustering
    logging.info("\nTesting GMM clustering:")
    gmm_pipeline = create_clustering_pipeline(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        clustering_algorithm='gmm',
        advanced_cleaning=True,
        n_components=5
    )
    df_gmm = gmm_pipeline.fit_transform(df)
    logging.info(f"\nGMM clustered data (first 5 rows):\n{df_gmm.head()}")
    logging.info(f"Unique GMM cluster labels: {df_gmm['cluster_label'].unique()}")

    # Test error handling
    logging.info("\nTesting error handling:")
    try:
        invalid_pipeline = create_clustering_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            clustering_algorithm='invalid_algo'
        )
    except ValueError as e:
        logging.info(f"Caught expected error: {e}")

    try:
        invalid_kmeans = create_clustering_pipeline(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            clustering_algorithm='kmeans'
        )
    except ValueError as e:
        logging.info(f"Caught expected error: {e}")