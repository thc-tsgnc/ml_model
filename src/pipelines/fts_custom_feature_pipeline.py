from sklearn.pipeline import Pipeline
from typing import List, Dict, Any
import pandas as pd
import numpy as np
from feature_engineering.fts_custom_feature_transformers import ExpectedProbabilityTransformer, TrueProbabilityTransformer

def create_fts_feature_pipeline(feature_sets: List[Dict[str, Any]]) -> Pipeline:
    """
    Create a pipeline for FTS custom feature engineering that processes multiple feature sets in batch.

    This pipeline combines ExpectedProbabilityTransformer and TrueProbabilityTransformer in sequence.
    
    Each transformer modifies the data and passes it along to the next step in the pipeline.
    
    Args:
    - feature_sets: List of dictionaries, each containing 'exp_prob_config' and 'true_prob_config' for a feature set

    Returns:
    - A scikit-learn Pipeline object
    """
    steps = []

    for i, feature_set in enumerate(feature_sets):
        exp_prob_config = feature_set['exp_prob_config']
        true_prob_config = feature_set['true_prob_config']
        
        exp_prob_transformer = ExpectedProbabilityTransformer(exp_prob_config)
        true_prob_transformer = TrueProbabilityTransformer(**true_prob_config)
        
        steps.append((f'exp_prob_{i}', exp_prob_transformer))
        steps.append((f'true_prob_{i}', true_prob_transformer))

    pipeline = Pipeline(steps)
    
    return pipeline

# Test code
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_size = 1000
    
    sample_data = pd.DataFrame({
        'condition_win1': np.random.choice([0, 1], sample_size),
        'prob_win1': np.random.uniform(0, 1, sample_size),
        'condition_lose1': np.random.choice([0, 1], sample_size),
        'prob_lose1': np.random.uniform(0, 1, sample_size),
        'condition_win2': np.random.choice([0, 1], sample_size),
        'prob_win2': np.random.uniform(0, 1, sample_size),
        'condition_lose2': np.random.choice([0, 1], sample_size),
        'prob_lose2': np.random.uniform(0, 1, sample_size),
        'team1_2pa_poss': np.random.uniform(0, 1, sample_size),
        'team1_2p_pct': np.random.uniform(0.4, 0.6, sample_size),
        'team1_3pa_poss': np.random.uniform(0, 1, sample_size),
        'team1_3p_pct': np.random.uniform(0.3, 0.5, sample_size),
        'team1_ft_pct': np.random.uniform(0.7, 0.9, sample_size),
        'team2_2pa_poss': np.random.uniform(0, 1, sample_size),
        'team2_2p_pct': np.random.uniform(0.4, 0.6, sample_size),
        'team2_3pa_poss': np.random.uniform(0, 1, sample_size),
        'team2_3p_pct': np.random.uniform(0.3, 0.5, sample_size),
        'team2_ft_pct': np.random.uniform(0.7, 0.9, sample_size),
    })

    # Example configurations for multiple feature sets
    feature_sets = [
        {
            'exp_prob_config': {
                'team1_exp_prob': {
                    'condition_win': 'condition_win1',
                    'prob_win': 'prob_win1',
                    'condition_lose': 'condition_lose1',
                    'prob_lose': 'prob_lose1'
                },
                'team2_exp_prob': {
                    'condition_win': 'condition_win2',
                    'prob_win': 'prob_win2',
                    'condition_lose': 'condition_lose2',
                    'prob_lose': 'prob_lose2'
                }
            },
            'true_prob_config': {
                'team_columns': {
                    'team1': {
                        '2pa_poss': 'team1_2pa_poss',
                        '2p_pct': 'team1_2p_pct',
                        '3pa_poss': 'team1_3pa_poss',
                        '3p_pct': 'team1_3p_pct',
                        'ft_pct': 'team1_ft_pct'
                    },
                    'team2': {
                        '2pa_poss': 'team2_2pa_poss',
                        '2p_pct': 'team2_2p_pct',
                        '3pa_poss': 'team2_3pa_poss',
                        '3p_pct': 'team2_3p_pct',
                        'ft_pct': 'team2_ft_pct'
                    }
                },
                'output_columns': {
                    'team1': 'team1_true_prob',
                    'team2': 'team2_true_prob'
                },
                'iterations': {'team1': 5, 'team2': 6}
            }
        },
        {
            'exp_prob_config': {
                'combined_exp_prob': {
                    'condition_win': 'condition_win1',
                    'prob_win': 'prob_win2',
                    'condition_lose': 'condition_lose2',
                    'prob_lose': 'prob_lose1'
                }
            },
            'true_prob_config': {
                'team_columns': {
                    'combined': {
                        '2pa_poss': 'team1_2pa_poss',
                        '2p_pct': 'team2_2p_pct',
                        '3pa_poss': 'team1_3pa_poss',
                        '3p_pct': 'team2_3p_pct',
                        'ft_pct': 'team1_ft_pct'
                    }
                },
                'output_columns': {
                    'combined': 'combined_true_prob'
                },
                'iterations': 7
            }
        }
    ]

    # Create the pipeline
    fts_pipeline = create_fts_feature_pipeline(feature_sets)

    print("FTS Feature Pipeline created successfully.")
    print("Pipeline steps:", fts_pipeline.steps)

    # Apply the pipeline to the sample data
    X_transformed = fts_pipeline.fit_transform(sample_data)

    print("\nOriginal data shape:", sample_data.shape)
    print("Transformed data shape:", X_transformed.shape)

    # Print new column names
    new_columns = [col for col in X_transformed.columns if col not in sample_data.columns]
    print("\nNew columns added:")
    for col in new_columns:
        print(f"- {col}")

    # Print a sample of the new data
    print("\nSample of new data (first 5 rows):")
    print(X_transformed[new_columns].head())
