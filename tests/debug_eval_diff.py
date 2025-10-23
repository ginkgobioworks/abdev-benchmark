"""Debug the exact difference in evaluation between baseline and new predictions."""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abdev_core import PROPERTY_LIST, recall_at_k


def debug_eval_diff():
    """Debug where the evaluation difference comes from."""
    
    baseline_cv_pred = Path(__file__).parent / "baseline_results" / "predictions" / "GDPa1_cross_validation" / "TAP" / "TAP - linear regression.csv"
    new_cv_pred = project_root / "predictions" / "GDPa1_cross_validation" / "tap_linear" / "predictions.csv"
    train_data = project_root / "data" / "GDPa1_v1.2_20250814.csv"
    fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
    
    df_baseline_pred = pd.read_csv(baseline_cv_pred)
    df_new_pred = pd.read_csv(new_cv_pred)
    df_train = pd.read_csv(train_data)
    
    print("Baseline predictions columns:", list(df_baseline_pred.columns))
    print("New predictions columns:", list(df_new_pred.columns))
    print()
    
    # Test a specific property: HIC
    prop = "HIC"
    
    print("="*70)
    print(f"Analyzing {prop}")
    print("="*70)
    
    # Baseline merge
    target_cols_baseline = ["antibody_name", fold_col] + PROPERTY_LIST
    df_merged_baseline = pd.merge(
        df_train[target_cols_baseline],
        df_baseline_pred[["antibody_name"] + [prop]],
        on="antibody_name",
        how="left",
        suffixes=("_true", "_pred"),
    )
    
    print(f"\nBaseline merge result shape: {df_merged_baseline.shape}")
    print(f"Columns: {list(df_merged_baseline.columns)}")
    print(f"Sample of merged baseline data:")
    print(df_merged_baseline.head(3))
    
    # New merge (should be same)
    df_merged_new = pd.merge(
        df_train[target_cols_baseline],
        df_new_pred[["antibody_name"] + [prop]],
        on="antibody_name",
        how="left",
        suffixes=("_true", "_pred"),
    )
    
    print(f"\nNew merge result shape: {df_merged_new.shape}")
    print(f"Columns: {list(df_merged_new.columns)}")
    
    # Check if they're the same
    baseline_preds = df_merged_baseline[prop + "_pred"].values
    new_preds = df_merged_new[prop + "_pred"].values
    
    print(f"\nPrediction values match: {np.allclose(baseline_preds, new_preds)}")
    
    # Now check fold distribution
    print(f"\nFold distribution in baseline predictions:")
    print(df_merged_baseline[fold_col].value_counts().sort_index())
    
    print(f"\nFold distribution in new predictions:")
    print(df_merged_new[fold_col].value_counts().sort_index())
    
    # Check what happens in each fold
    print(f"\n{'Fold':<6} {'Baseline Top10':<20} {'New Top10':<20}")
    print("-" * 50)
    
    # Import evaluation code
    from abdev_core import evaluate, recall_at_k
    
    for fold in sorted(df_merged_baseline[fold_col].unique()):
        mask_baseline = df_merged_baseline[fold_col] == fold
        mask_new = df_merged_new[fold_col] == fold
        
        y_true_baseline = df_merged_baseline.loc[mask_baseline, prop + "_true"].values
        y_pred_baseline = df_merged_baseline.loc[mask_baseline, prop + "_pred"].values
        
        y_true_new = df_merged_new.loc[mask_new, prop + "_true"].values
        y_pred_new = df_merged_new.loc[mask_new, prop + "_pred"].values
        
        # Recall at k uses higher is better logic
        y_true_baseline_sign = -1 * y_true_baseline  # HIC is lower is better
        y_pred_baseline_sign = -1 * y_pred_baseline
        
        y_true_new_sign = -1 * y_true_new
        y_pred_new_sign = -1 * y_pred_new
        
        recall_baseline = recall_at_k(y_true_baseline_sign, y_pred_baseline_sign, frac=0.1)
        recall_new = recall_at_k(y_true_new_sign, y_pred_new_sign, frac=0.1)
        
        print(f"{fold:<6} {recall_baseline:<20.4f} {recall_new:<20.4f}")


if __name__ == "__main__":
    debug_eval_diff()
