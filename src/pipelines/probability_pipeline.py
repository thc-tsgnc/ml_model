from sklearn.pipeline import Pipeline
from feature_engineering.probability_transformers import (
    LaplaceSmoothingTransformer,
    ProbabilityNormalizationTransformer,
    Log5ProbabilityTransformer
)

def create_probability_pipeline(
    laplace_config=None,
    normalization_column_pairs=None,
    use_log5=False,
    log5_column_pairs=None
):
    """
    Create a probability pipeline based on the provided configuration.

    Args:
    laplace_config (dict): Configuration for LaplaceSmoothingTransformer
    normalization_column_pairs (list): Column pairs for ProbabilityNormalizationTransformer
    use_log5 (bool): Whether to include Log5ProbabilityTransformer
    log5_column_pairs (list): Column pairs for Log5ProbabilityTransformer

    Returns:
    sklearn.pipeline.Pipeline: Configured probability pipeline
    """
    steps = []

    if laplace_config:
        steps.append(('laplace_smoothing', LaplaceSmoothingTransformer(config=laplace_config)))

    if normalization_column_pairs:
        steps.append(('normalization', ProbabilityNormalizationTransformer(column_pairs=normalization_column_pairs)))

    if use_log5 and log5_column_pairs:
        steps.append(('log5', Log5ProbabilityTransformer(column_pairs=log5_column_pairs)))

    return Pipeline(steps)

# Test code
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Create sample data
    np.random.seed(42)
    data = {
        't1_player_o_rolling_tip_avg': np.random.uniform(0, 1, 10),
        't2_player_o_rolling_tip_avg': np.random.uniform(0, 1, 10),
        't1_o_rolling_fts_avg': np.random.uniform(0, 1, 10),
        't2_o_rolling_fts_avg': np.random.uniform(0, 1, 10),
        'count1': np.random.randint(1, 30, 10),
        'count2': np.random.randint(1, 30, 10)
    }
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    # Define configurations
    laplace_config = {
        ('t1_player_o_rolling_tip_avg', 't2_player_o_rolling_tip_avg'): {'count_column': 'count1', 'threshold': 12, 'smoothing_factor': 1},
        ('t1_o_rolling_fts_avg', 't2_o_rolling_fts_avg'): {'count_column': 'count2', 'threshold': 15, 'smoothing_factor': 0.5}
    }
    normalization_column_pairs = [
        ('t1_player_o_rolling_tip_avg', 't2_player_o_rolling_tip_avg'),
        ('t1_o_rolling_fts_avg', 't2_o_rolling_fts_avg')
    ]
    log5_column_pairs = [
        ('t1_player_o_rolling_tip_avg', 't2_player_o_rolling_tip_avg'),
        ('t1_o_rolling_fts_avg', 't2_o_rolling_fts_avg')
    ]

    # Test pipeline without Log5
    pipeline_without_log5 = create_probability_pipeline(
        laplace_config=laplace_config,
        normalization_column_pairs=normalization_column_pairs,
        use_log5=False
    )
    df_transformed_without_log5 = pipeline_without_log5.fit_transform(df)

    print("\nTransformed DataFrame (without Log5):")
    print(df_transformed_without_log5)

    # Test pipeline with Log5
    pipeline_with_log5 = create_probability_pipeline(
        laplace_config=laplace_config,
        normalization_column_pairs=normalization_column_pairs,
        use_log5=True,
        log5_column_pairs=log5_column_pairs
    )
    df_transformed_with_log5 = pipeline_with_log5.fit_transform(df)

    print("\nTransformed DataFrame (with Log5):")
    print(df_transformed_with_log5)

    # Verify results
    print("\nVerification:")
    print("1. Check if Laplace smoothing was applied")
    print("2. Verify that normalized probabilities sum to 1 for each pair")
    print("3. Confirm Log5 probabilities are calculated correctly when included")
    print("4. Ensure all transformations are applied in the correct order")

    # Additional checks can be added here as needed