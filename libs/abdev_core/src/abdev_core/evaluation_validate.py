"""Validation functions for prediction format and schema."""

import pandas as pd
from pathlib import Path

from . import PROPERTY_LIST


def validate_prediction_format(pred_df: pd.DataFrame) -> tuple[bool, list[str]]:
    """Validate that a prediction DataFrame has the correct format.
    
    Args:
        pred_df: DataFrame to validate
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check for antibody_name column
    if "antibody_name" not in pred_df.columns:
        errors.append("Missing required column 'antibody_name'")
    
    # Check for at least one property column
    property_cols = [col for col in pred_df.columns if col in PROPERTY_LIST]
    if not property_cols:
        errors.append(
            f"No property columns found. Must have at least one of: {PROPERTY_LIST}"
        )
    
    # Check for NaN in antibody_name
    if "antibody_name" in pred_df.columns and pred_df["antibody_name"].isna().any():
        errors.append("antibody_name column contains NaN values")
    
    # Check for duplicate antibody names
    if "antibody_name" in pred_df.columns:
        duplicates = pred_df["antibody_name"].duplicated()
        if duplicates.any():
            n_dups = duplicates.sum()
            errors.append(f"Found {n_dups} duplicate antibody_name values")
    
    return len(errors) == 0, errors


def validate_prediction_file(pred_path: Path) -> tuple[bool, list[str]]:
    """Validate a prediction CSV file.
    
    Args:
        pred_path: Path to prediction CSV
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check file exists
    if not pred_path.exists():
        return False, [f"File not found: {pred_path}"]
    
    # Try to read the file
    try:
        df = pd.read_csv(pred_path)
    except Exception as e:
        return False, [f"Failed to read CSV: {str(e)}"]
    
    # Validate format
    is_valid, format_errors = validate_prediction_format(df)
    errors.extend(format_errors)
    
    return len(errors) == 0, errors

