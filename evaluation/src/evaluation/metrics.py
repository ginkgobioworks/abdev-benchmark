"""Evaluation metrics for antibody developability predictions."""

from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from abdev_core import PROPERTY_LIST, ASSAY_HIGHER_IS_BETTER


def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.1) -> float:
    """Calculate recall (TP)/(TP+FN) for top fraction of true values.

    A recall of 1 would mean that the top fraction of true values are also the top 
    fraction of predicted values. There is no penalty for ranking the top k differently.

    Args:
        y_true: true values with shape (num_data,)
        y_pred: predicted values with shape (num_data,)
        frac: fraction of data points to consider as the top. Defaults to 0.1.

    Returns:
        float: recall at top k of data
    """
    top_k = int(len(y_true) * frac)
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    true_top_k = np.argsort(y_true)[-1 * top_k :]
    predicted_top_k = np.argsort(y_pred)[-1 * top_k :]

    return (
        len(
            set(list(true_top_k.flatten())).intersection(
                set(list(predicted_top_k.flatten()))
            )
        )
        / top_k
    )


def evaluate(
    predictions_series: pd.Series, target_series: pd.Series, assay_col: str
) -> dict[str, float]:
    """Evaluate predictions for a single assay.
    
    Args:
        predictions_series: Predicted values
        target_series: True values
        assay_col: Name of the assay being evaluated
        
    Returns:
        Dictionary with evaluation metrics
    """
    results_dict = {
        "spearman": spearmanr(predictions_series, target_series, nan_policy="omit").correlation
    }
    # Top 10% recall
    y_true = target_series.values
    y_pred = predictions_series.values
    if not ASSAY_HIGHER_IS_BETTER[assay_col]:
        y_true = -1 * y_true
        y_pred = -1 * y_pred
    results_dict["top_10_recall"] = recall_at_k(y_true=y_true, y_pred=y_pred, frac=0.1)
    return results_dict


def evaluate_cross_validation(
    predictions_series: pd.Series,
    target_series: pd.Series,
    folds_series: pd.Series,
    assay_col: str,
    num_folds: int = 5,
) -> dict[str, float]:
    """Run evaluation in a cross-validation loop.
    
    Args:
        predictions_series: Predicted values
        target_series: True values
        folds_series: Fold assignments
        assay_col: Name of the assay being evaluated
        num_folds: Expected number of folds for validation
        
    Returns:
        Dictionary with mean evaluation metrics across folds
    """
    results_dict = defaultdict(list)
    actual_folds = folds_series.nunique()
    if actual_folds != num_folds:
        raise ValueError(f"Expected {num_folds} folds, got {actual_folds}")
    for fold in folds_series.unique():
        predictions_series_fold = predictions_series[folds_series == fold]
        target_series_fold = target_series[folds_series == fold]
        results = evaluate(predictions_series_fold, target_series_fold, assay_col)
        # Update the results_dict with the results for this fold
        for key, value in results.items():
            results_dict[key].append(value)
    # Calculate the mean of the results for each key
    for key, values in results_dict.items():
        results_dict[key] = np.mean(values)
    return results_dict


def evaluate_model(
    preds_path: Path,
    target_path: Path,
    model_name: str,
    dataset_name: str = None,
    fold_col: str = None,
    num_folds: int = 5,
) -> list[dict]:
    """Evaluate a single model on all properties.
    
    The predictions dataframe should have columns named by property (e.g., HIC, Tm2).
    
    Args:
        preds_path: Path to predictions CSV
        target_path: Path to ground truth CSV
        model_name: Name of the model
        dataset_name: Name of the dataset (e.g., "GDPa1", "GDPa1_cross_validation")
        fold_col: Column name for fold assignments (required for cross-validation)
        num_folds: Number of folds for validation (used in cross-validation)
        
    Returns:
        List of evaluation result dictionaries
    """
    predictions_df = pd.read_csv(preds_path)
    target_df = pd.read_csv(target_path)
    properties_in_preds = [col for col in predictions_df.columns if col in PROPERTY_LIST]
    
    # Determine which columns to include from target_df
    target_cols = ["antibody_name"] + PROPERTY_LIST
    if dataset_name == "GDPa1_cross_validation" and fold_col:
        if fold_col not in target_df.columns:
            raise ValueError(f"Fold column '{fold_col}' not found in target data")
        target_cols.insert(1, fold_col)
    
    df_merged = pd.merge(
        target_df[target_cols],
        predictions_df[["antibody_name"] + properties_in_preds],
        on="antibody_name",
        how="left",
        suffixes=("_true", "_pred"),
    )
    
    results_list = []
    for assay_col in properties_in_preds:
        if dataset_name == "GDPa1_cross_validation":
            if not fold_col:
                raise ValueError("fold_col is required for cross-validation evaluation")
            results = evaluate_cross_validation(
                df_merged[assay_col + "_pred"],
                df_merged[assay_col + "_true"],
                df_merged[fold_col],
                assay_col,
                num_folds=num_folds,
            )
        else:
            results = evaluate(
                df_merged[assay_col + "_pred"], df_merged[assay_col + "_true"], assay_col
            )
        results["dataset"] = dataset_name
        results["assay"] = assay_col
        results["model"] = model_name
        results_list.append(results)

    return results_list

