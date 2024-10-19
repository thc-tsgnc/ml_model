from sklearn.pipeline import Pipeline
from feature_engineering.arithmetic_transformers import ArithmeticDifferenceTransformer, ArithmeticRatioTransformer
from typing import List, Tuple, Optional

def create_arithmetic_pipeline(
    difference_column_pairs: Optional[List[Tuple[str, str]]] = None,
    ratio_column_pairs: Optional[List[Tuple[str, str]]] = None,
    ratio_fill_value: float = 0
):
    """
    Create an arithmetic pipeline based on the provided configuration.

    Args:
    difference_column_pairs (List[Tuple[str, str]], optional): Column pairs for ArithmeticDifferenceTransformer
    ratio_column_pairs (List[Tuple[str, str]], optional): Column pairs for ArithmeticRatioTransformer
    ratio_fill_value (float): Fill value for divide-by-zero cases in ArithmeticRatioTransformer

    Returns:
    sklearn.pipeline.Pipeline: Configured arithmetic pipeline
    """
    steps = []

    if difference_column_pairs:
        steps.append(('difference', ArithmeticDifferenceTransformer(column_pairs=difference_column_pairs)))

    if ratio_column_pairs:
        steps.append(('ratio', ArithmeticRatioTransformer(column_pairs=ratio_column_pairs, fill_value=ratio_fill_value)))

    return Pipeline(steps)

# Test code
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Create sample data
    np.random.seed(42)
    data = {
        'value_a': np.random.uniform(10, 50, 10),
        'value_b': np.random.uniform(5, 25, 10),
        'value_c': np.random.uniform(1, 10, 10)
    }
    df = pd.DataFrame(data)

    print("Original DataFrame:")
    print(df)

    # Define configurations
    difference_pairs = [('value_a', 'value_b'), ('value_b', 'value_c')]
    ratio_pairs = [('value_a', 'value_b'), ('value_b', 'value_c')]

    # Create and apply the pipeline
    arithmetic_pipeline = create_arithmetic_pipeline(
        difference_column_pairs=difference_pairs,
        ratio_column_pairs=ratio_pairs,
        ratio_fill_value=-1
    )
    df_transformed = arithmetic_pipeline.fit_transform(df)

    print("\nTransformed DataFrame:")
    print(df_transformed)

    # Verify results
    print("\nVerification:")
    print("1. Check if difference columns are created with '_diff' suffix")
    print("2. Verify that ratio columns are created with '_rto' suffix")
    print("3. Confirm that calculations are correct")
    print("4. Ensure divide-by-zero cases are handled properly in ratio calculations")

    # Additional checks can be added here as needed