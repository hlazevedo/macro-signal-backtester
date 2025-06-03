import pandas as pd
from typing import Any, List

def validate_data_alignment(data1: pd.DataFrame, data2: pd.DataFrame, 
                           name1: str = "data1", name2: str = "data2") -> None:
    """Validate that two DataFrames are properly aligned."""
    if not data1.index.equals(data2.index):
        common_dates = data1.index.intersection(data2.index)
        missing_in_1 = data2.index.difference(data1.index)
        missing_in_2 = data1.index.difference(data2.index)
        
        raise ValueError(
            f"Index mismatch between {name1} and {name2}. "
            f"Common dates: {len(common_dates)}, "
            f"Missing in {name1}: {len(missing_in_1)}, "
            f"Missing in {name2}: {len(missing_in_2)}"
        )

def validate_weights(weights: pd.Series, max_leverage: float = 1.0) -> None:
    """Validate portfolio weights."""
    total_weight = weights.abs().sum()
    
    if total_weight > max_leverage + 1e-6:
        raise ValueError(
            f"Total weight {total_weight:.4f} exceeds "
            f"maximum leverage {max_leverage}"
        )
    
    if weights.isnull().any():
        raise ValueError("Weights contain NaN values")