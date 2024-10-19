from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from feature_engineering.feature_engineering_transformers import (
    BorutaFeatureSelector,
    CustomPolynomialFeatures,
    FeatureImportanceReducer,
    CustomVarianceThreshold
)

def create_feature_engineering_pipeline(
    use_boruta=False,
    use_polynomial_features=False,
    use_feature_importance=False,
    boruta_params=None,
    polynomial_params=None,
    feature_importance_params=None,
    variance_threshold=0.0
):
    steps = []
    
    if use_boruta:
        boruta = BorutaFeatureSelector(**(boruta_params or {}))
        steps.append(('boruta', boruta))
    
    if use_polynomial_features:
        poly = CustomPolynomialFeatures(**(polynomial_params or {}))
        steps.append(('polynomial', poly))
    
    if use_feature_importance:
        fir = FeatureImportanceReducer(**(feature_importance_params or {}))
        steps.append(('feature_importance', fir))
    
    # Always add CustomVarianceThreshold as the last step
    variance = CustomVarianceThreshold(threshold=variance_threshold)
    steps.append(('variance_threshold', variance))
    
    return Pipeline(steps)

# Example usage
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, 
                               n_repeated=0, n_classes=2, random_state=42)
    X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    y = pd.Series(y, name='target')

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with all transformers
    pipeline = create_feature_engineering_pipeline(
        use_boruta=True,
        use_polynomial_features=True,
        use_feature_importance=True,
        boruta_params={'max_iter': 50, 'random_state': 42},
        polynomial_params={'degree': 2, 'interaction_only': True},
        feature_importance_params={'estimator': RandomForestClassifier(n_estimators=100, random_state=42), 'top_n': 15},
        variance_threshold=0.01
    )

    # Fit and transform the data
    X_train_transformed = pipeline.fit_transform(X_train, y_train)
    X_test_transformed = pipeline.transform(X_test)

    print(f"Original number of features: {X_train.shape[1]}")
    print(f"Number of features after transformation: {X_train_transformed.shape[1]}")

    # Example of combining with another pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    combined_pipeline = Pipeline([
        ('feature_engineering', pipeline),
        ('scaler', StandardScaler()),
        ('classifier', LogisticRegression())
    ])

    # Fit the combined pipeline
    combined_pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = combined_pipeline.predict(X_test)

    # Print accuracy
    from sklearn.metrics import accuracy_score
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")