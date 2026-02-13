"""Data loading utilities for the Maestria_DS project."""

import pandas as pd
from pathlib import Path


def load_csv(filepath: str) -> pd.DataFrame:
    """Load a CSV file into a pandas DataFrame.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame with loaded data
    """
    return pd.read_csv(filepath)


def load_data(filepath: str) -> pd.DataFrame:
    """Load data from various formats.
    
    Args:
        filepath: Path to the data file
        
    Returns:
        DataFrame with loaded data
    """
    path = Path(filepath)
    
    if path.suffix == '.csv':
        return load_csv(filepath)
    elif path.suffix == '.xlsx':
        return pd.read_excel(filepath)
    elif path.suffix == '.json':
        return pd.read_json(filepath)
    else:
        raise ValueError(f'Unsupported file format: {path.suffix}')
