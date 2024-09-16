# src/feature_engineering/feature_engineering_manager.py

import pandas as pd
from typing import Dict, Any, Callable
from .cluster_features import generate_cluster_features

class FeatureEngineeringManager:
    def __init__(self):
        self.feature_methods = {}
        self.init_feature_pipeline()

    def init_feature_pipeline(self):
        """
        Initializes the feature engineering pipeline by setting up configurations and registering default methods.
        """
        # Register the cluster feature method by default
        self.register_feature_method('cluster', generate_cluster_features)
        
        # Additional setup can be added here (e.g., loading configurations, setting up logging)
        print("Feature engineering pipeline initialized.")

    def run_feature_creation(self, data: pd.DataFrame, method: str = 'cluster', params: Dict[str, Any] = {}) -> pd.DataFrame:
        """
        Main entry point for feature creation, directing the process based on the user's choice of method.

        Args:
        data (pd.DataFrame): The dataset that requires feature enhancement.
        method (str): A string specifying the feature creation method (default is 'cluster').
        params (Dict[str, Any]): A dictionary containing necessary parameters for the selected feature creation method.

        Returns:
        pd.DataFrame: The input DataFrame enhanced with new features.
        """
        if method not in self.feature_methods:
            raise ValueError(f"Unknown feature creation method: {method}")
        
        try:
            enhanced_data = self.feature_methods[method](data, **params)
            return enhanced_data
        except Exception as e:
            print(f"Error in run_feature_creation: {str(e)}")
            # In case of error, return the original dataset
            return data

    def register_feature_method(self, name: str, function: Callable):
        """
        Allows dynamic addition of new feature creation methods, supporting extensibility of the pipeline.

        Args:
        name (str): The name of the feature creation method.
        function (Callable): The function implementing the feature creation logic.
        """
        self.feature_methods[name] = function
        print(f"Feature method '{name}' registered successfully.")

# Create a global instance of the FeatureEngineeringManager
feature_manager = FeatureEngineeringManager()

# Expose the run_feature_creation method at the module level for ease of use
def run_feature_creation(data: pd.DataFrame, method: str = 'cluster', params: Dict[str, Any] = {}) -> pd.DataFrame:
    return feature_manager.run_feature_creation(data, method, params)