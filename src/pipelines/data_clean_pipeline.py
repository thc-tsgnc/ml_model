# src/pipelines/data_clean_pipeline.py

import logging
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_columns(X, expected_columns):
    """
    Ensures that all expected columns are present in the DataFrame.
    """
    missing_columns = [col for col in expected_columns if col not in X.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in the data: {missing_columns}")
    logging.info(f"All expected columns are present: {expected_columns}")

def validate_input_data(X):
    """
    Ensure that the input is a pandas DataFrame.
    """
    if not isinstance(X, pd.DataFrame):
        raise ValueError("Input data is not a pandas DataFrame. Please provide a DataFrame.")
    logging.info("Input data is a valid pandas DataFrame.")

def create_basic_cleaning_pipeline(numeric_features, categorical_features, 
                                   numeric_impute_strategy='mean', categorical_impute_strategy='most_frequent'):
    """
    Creates a basic data cleaning pipeline with:
    - Numeric imputation (default: 'mean')
    - Categorical imputation (default: 'most_frequent')
    
    Parameters:
    - numeric_features: List of numeric feature names
    - categorical_features: List of categorical feature names
    - numeric_impute_strategy: Strategy to impute numeric features ('mean', 'median', etc.)
    - categorical_impute_strategy: Strategy to impute categorical features ('most_frequent', 'constant', etc.)
    
    Returns:
    - A scikit-learn Pipeline object
    """
    
    logging.info("Creating a basic cleaning pipeline with numeric and categorical features.")
    
    # Create numeric transformer
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_impute_strategy))
    ])

    # Create categorical transformer
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_impute_strategy))
    ])

    # Combine numeric and categorical transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    logging.info("Basic cleaning pipeline created successfully.")
    
    # Return the assembled pipeline
    return Pipeline(steps=[('preprocessor', preprocessor)])


def create_advanced_cleaning_pipeline(numeric_features, categorical_features, 
                                      numeric_impute_strategy='mean', categorical_impute_strategy='constant',
                                      fill_value='missing'):
    """
    Creates an advanced data cleaning pipeline with:
    - Numeric imputation (default: 'mean') and scaling
    - Categorical imputation (default: 'constant') and one-hot encoding
    
    Parameters:
    - numeric_features: List of numeric feature names
    - categorical_features: List of categorical feature names
    - numeric_impute_strategy: Strategy to impute numeric features ('mean', 'median', etc.)
    - categorical_impute_strategy: Strategy to impute categorical features ('most_frequent', 'constant', etc.)
    - fill_value: Value to use for constant imputation for categorical features (default: 'missing')
    
    Returns:
    - A scikit-learn Pipeline object
    """
    
    logging.info("Creating an advanced cleaning pipeline with numeric and categorical features.")
    
    # Create numeric transformer (with imputation and scaling)
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=numeric_impute_strategy)),
        ('scaler', StandardScaler())
    ])

    # Create categorical transformer (with imputation and encoding)
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy=categorical_impute_strategy, fill_value=fill_value)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine numeric and categorical transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    logging.info("Advanced cleaning pipeline created successfully.")
    
    # Return the assembled pipeline
    return Pipeline(steps=[('preprocessor', preprocessor)])


class DataCleanPipeline:
    """
    A class to manage the validation, logging, and application of the data cleaning pipelines.
    """

    def __init__(self, numeric_features, categorical_features, advanced=False, 
                 numeric_impute_strategy='mean', categorical_impute_strategy='most_frequent', fill_value='missing'):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.advanced = advanced
        self.numeric_impute_strategy = numeric_impute_strategy
        self.categorical_impute_strategy = categorical_impute_strategy
        self.fill_value = fill_value
        
        # Decide whether to use basic or advanced cleaning
        if self.advanced:
            logging.info("Initializing advanced cleaning pipeline.")
            self.pipeline = create_advanced_cleaning_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                numeric_impute_strategy=numeric_impute_strategy,
                categorical_impute_strategy=categorical_impute_strategy,
                fill_value=fill_value
            )
        else:
            logging.info("Initializing basic cleaning pipeline.")
            self.pipeline = create_basic_cleaning_pipeline(
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                numeric_impute_strategy=numeric_impute_strategy,
                categorical_impute_strategy=categorical_impute_strategy
            )

    def fit_transform(self, X):
        """
        Fit and transform the data using the chosen cleaning pipeline.
        """
        logging.info("Starting data validation.")
        
        # Validate the input data
        validate_input_data(X)
        
        # Validate columns in the input data
        expected_columns = self.numeric_features + self.categorical_features
        validate_columns(X, expected_columns)
        
        logging.info("Fitting and transforming the data using the cleaning pipeline.")
        
        # Apply the pipeline to the data
        return self.pipeline.fit_transform(X)

def test_data_clean_pipeline():
    import pandas as pd
    import numpy as np

    # Create a sample dataset
    np.random.seed(42)
    data = {
        'age': np.random.randint(18, 80, 1000),
        'income': np.random.randint(20000, 200000, 1000),
        'gender': np.random.choice(['M', 'F', None], 1000),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD', None], 1000)
    }
    df = pd.DataFrame(data)

    # Introduce some missing values
    df.loc[np.random.choice(df.index, 100), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 100), 'income'] = np.nan

    # Define feature lists
    numeric_features = ['age', 'income']
    categorical_features = ['gender', 'education']

    # Test basic cleaning pipeline
    logging.info("Testing basic cleaning pipeline")
    basic_pipeline = DataCleanPipeline(numeric_features, categorical_features, advanced=False)
    df_basic_cleaned = basic_pipeline.fit_transform(df)
    logging.info(f"Basic cleaned data shape: {df_basic_cleaned.shape}")

    # Test advanced cleaning pipeline
    logging.info("\nTesting advanced cleaning pipeline")
    advanced_pipeline = DataCleanPipeline(numeric_features, categorical_features, advanced=True)
    df_advanced_cleaned = advanced_pipeline.fit_transform(df)
    logging.info(f"Advanced cleaned data shape: {df_advanced_cleaned.shape}")

    # Print sample results via logging
    logging.info("\nSample results:")
    logging.info("Original data:")
    logging.info(f"\n{df.head()}")
    logging.info("\nBasic cleaned data:")
    logging.info(f"\n{pd.DataFrame(df_basic_cleaned).head()}")
    logging.info("\nAdvanced cleaned data:")
    logging.info(f"\n{pd.DataFrame(df_advanced_cleaned).head()}")

if __name__ == '__main__':
    test_data_clean_pipeline()