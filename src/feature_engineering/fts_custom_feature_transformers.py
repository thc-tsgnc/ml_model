import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Dict, List, Union, Callable

def validate_columns(X: Union[pd.DataFrame, np.ndarray], required_columns: List[str]) -> None:
    """
    Centralized validation logic for checking required columns.
    
    Args:
    - X: Input data (DataFrame or ndarray)
    - required_columns: List of required column names or indices
    """
    if isinstance(X, np.ndarray):
        if X.shape[1] < len(required_columns):
            raise ValueError("Input array does not have enough columns.")
    else:
        missing_columns = set(required_columns) - set(X.columns)
        if missing_columns:
            raise ValueError(f"Columns {missing_columns} not found in input DataFrame.")

def prepare_made_probabilities(X: Union[pd.DataFrame, np.ndarray], 
                               column_config: Dict[str, str], 
                               custom_logic: Callable = None) -> np.ndarray:
    """
    Prepares the made probabilities based on shooting percentages.
    
    Args:
    - X: Input data (DataFrame or ndarray)
    - column_config: Dictionary mapping required fields to column names
    - custom_logic: Optional custom function for probability calculation
    
    Returns:
    - made_prob: The calculated made probability
    """
    if custom_logic:
        return custom_logic(X, column_config)
    
    if isinstance(X, np.ndarray):
        return (
            X[:, column_config['2pa_poss']] * X[:, column_config['2p_pct']] +
            X[:, column_config['3pa_poss']] * X[:, column_config['3p_pct']] +
            0.14 * X[:, column_config['2pa_poss']] * X[:, column_config['ft_pct']] +
            0.009 * X[:, column_config['3pa_poss']] * X[:, column_config['ft_pct']]
        )
    else:
        return (
            X[column_config['2pa_poss']] * X[column_config['2p_pct']] +
            X[column_config['3pa_poss']] * X[column_config['3p_pct']] +
            0.14 * X[column_config['2pa_poss']] * X[column_config['ft_pct']] +
            0.009 * X[column_config['3pa_poss']] * X[column_config['ft_pct']]
        )

def calculate_recurring_probability(made_prob: np.ndarray, 
                                    miss_prob_self: np.ndarray, 
                                    miss_prob_opp: np.ndarray, 
                                    iterations: int = 5) -> np.ndarray:
    """
    Calculates the recurring probability over multiple iterations.

    Args:
    - made_prob: The base probability of making a shot
    - miss_prob_self: The miss probability of the same team
    - miss_prob_opp: The miss probability of the opponent
    - iterations: Number of iterations for the calculation

    Returns:
    - exp_made_prob: The final expected made probability
    """
    exp_made_prob = made_prob.copy()
    for _ in range(1, iterations):
        exp_made_prob += miss_prob_self * miss_prob_opp * made_prob
        miss_prob_self *= miss_prob_opp

    return exp_made_prob

class ExpectedProbabilityTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to calculate the expected probability from input columns.
    """
    def __init__(self, column_sets: Dict[str, Dict[str, str]]):
        """
        Initialize the transformer with column sets.

        Args:
        - column_sets: Dictionary where keys are output column names and values are
                       dictionaries specifying input column names.
        """
        self.column_sets = column_sets

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit method, performs validation on input data.
        """
        all_required_columns = set()
        for column_set in self.column_sets.values():
            all_required_columns.update(column_set.values())
        validate_columns(X, list(all_required_columns))
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply the expected probability calculation for each column set.
        """
        X_new = X.copy() if isinstance(X, pd.DataFrame) else np.copy(X)
        
        for output_col, input_cols in self.column_sets.items():
            if isinstance(X, np.ndarray):
                new_col = (
                    X[:, input_cols['condition_win']] * X[:, input_cols['prob_win']] + 
                    X[:, input_cols['condition_lose']] * X[:, input_cols['prob_lose']]
                )
                X_new = np.column_stack((X_new, new_col))
            else:
                X_new[f"{output_col}_exp_prob"] = (
                    X[input_cols['condition_win']] * X[input_cols['prob_win']] + 
                    X[input_cols['condition_lose']] * X[input_cols['prob_lose']]
                ).fillna(0)
        
        return X_new

class TrueProbabilityTransformer(BaseEstimator, TransformerMixin):
    """
    Transformer to calculate the true probability for teams.
    """
    def __init__(self, team_columns: Dict[str, Dict[str, str]], 
                 output_columns: Dict[str, str], 
                 iterations: Union[int, Dict[str, int]] = 5,
                 custom_probability_func: Callable = None):
        """
        Initialize the transformer with team-specific column names and output column names.

        Args:
        - team_columns: Dictionary of team names to column configurations
        - output_columns: Dictionary mapping team names to output column names
        - iterations: Number of iterations for recurring probability calculation (int or dict)
        - custom_probability_func: Optional custom function for probability calculation
        """
        self.team_columns = team_columns
        self.output_columns = output_columns
        self.iterations = iterations
        self.custom_probability_func = custom_probability_func

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit method, performs validation on input data.
        """
        all_required_columns = set()
        for team_config in self.team_columns.values():
            all_required_columns.update(team_config.values())
        validate_columns(X, list(all_required_columns))
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        """
        Perform the true probability calculation for each team.
        """
        X_new = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=[f"col_{i}" for i in range(X.shape[1])])
        
        made_probs = {}
        for team, columns in self.team_columns.items():
            made_probs[team] = prepare_made_probabilities(X, columns, self.custom_probability_func)
        
        miss_probs = {team: 1 - prob for team, prob in made_probs.items()}
        
        for team, output_col in self.output_columns.items():
            other_teams = [t for t in self.team_columns.keys() if t != team]
            iterations = self.iterations[team] if isinstance(self.iterations, dict) else self.iterations
            
            exp_made_prob = calculate_recurring_probability(
                made_probs[team], 
                miss_probs[team], 
                np.mean([miss_probs[t] for t in other_teams], axis=0),
                iterations
            )
            
            X_new[f"{output_col}_true_prob"] = exp_made_prob.fillna(0)
        
        return X_new

    
if __name__ == "__main__":
    # Create sample data
    np.random.seed(42)
    sample_data = pd.DataFrame({
        'team1_2pa_poss': np.random.uniform(0, 1, 100),
        'team1_2p_pct': np.random.uniform(0.4, 0.6, 100),
        'team1_3pa_poss': np.random.uniform(0, 1, 100),
        'team1_3p_pct': np.random.uniform(0.3, 0.5, 100),
        'team1_ft_pct': np.random.uniform(0.7, 0.9, 100),
        'team2_2pa_poss': np.random.uniform(0, 1, 100),
        'team2_2p_pct': np.random.uniform(0.4, 0.6, 100),
        'team2_3pa_poss': np.random.uniform(0, 1, 100),
        'team2_3p_pct': np.random.uniform(0.3, 0.5, 100),
        'team2_ft_pct': np.random.uniform(0.7, 0.9, 100),
        'condition_win1': np.random.choice([0, 1], 100),
        'prob_win1': np.random.uniform(0, 1, 100),
        'condition_lose1': np.random.choice([0, 1], 100),
        'prob_lose1': np.random.uniform(0, 1, 100),
        'condition_win2': np.random.choice([0, 1], 100),
        'prob_win2': np.random.uniform(0, 1, 100),
        'condition_lose2': np.random.choice([0, 1], 100),
        'prob_lose2': np.random.uniform(0, 1, 100),
    })

    # Initialize transformers
    exp_prob_transformer = ExpectedProbabilityTransformer({
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
    })

    true_prob_transformer = TrueProbabilityTransformer(
        team_columns={
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
        output_columns={
            'team1': 'team1_true_prob',
            'team2': 'team2_true_prob'
        },
        iterations={'team1': 5, 'team2': 6}
    )

    # Create and run pipeline
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ('exp_prob', exp_prob_transformer),
        ('true_prob', true_prob_transformer)
    ])

    # Fit and transform data
    result = pipeline.fit_transform(sample_data)

    # Print results
    print("Original data shape:", sample_data.shape)
    print("Transformed data shape:", result.shape)
    print("\nNew columns added:")
    new_columns = [col for col in result.columns if col not in sample_data.columns]
    for col in new_columns:
        print(f"- {col}")
    
    print("\nSample of new data:")
    print(result[new_columns].head())

    # Basic validation
    assert 'team1_exp_prob_exp_prob' in result.columns, "Expected probability column for team1 is missing"
    assert 'team2_exp_prob_exp_prob' in result.columns, "Expected probability column for team2 is missing"
    assert 'team1_true_prob_true_prob' in result.columns, "True probability column for team1 is missing"
    assert 'team2_true_prob_true_prob' in result.columns, "True probability column for team2 is missing"
    
    print("\nAll expected columns are present in the result.")
    print("Test completed successfully!")