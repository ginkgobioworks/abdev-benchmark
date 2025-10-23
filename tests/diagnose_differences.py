"""Diagnose differences between baseline and new CV predictions.

This script investigates:
1. Whether we're using the same fold assignments
2. Whether prediction values differ or are just reordered
3. Which baselines have precomputed values (should be identical)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abdev_core import PROPERTY_LIST, evaluate_model


def compare_cv_predictions():
    """Compare baseline vs new CV predictions."""
    
    baseline_results = Path(__file__).parent / "baseline_results"
    baseline_cv_pred = baseline_results / "predictions" / "GDPa1_cross_validation" / "TAP" / "TAP - linear regression.csv"
    new_cv_pred = project_root / "predictions" / "GDPa1_cross_validation" / "tap_linear" / "predictions.csv"
    train_data = project_root / "data" / "GDPa1_v1.2_20250814.csv"
    
    print("\n" + "="*70)
    print("Comparing TAP Linear CV Predictions")
    print("="*70)
    
    if not baseline_cv_pred.exists():
        print(f"❌ Baseline file not found: {baseline_cv_pred}")
        return
    
    if not new_cv_pred.exists():
        print(f"❌ New file not found: {new_cv_pred}")
        return
    
    # Load data
    df_baseline = pd.read_csv(baseline_cv_pred)
    df_new = pd.read_csv(new_cv_pred)
    df_train = pd.read_csv(train_data)
    
    print(f"\nBaseline predictions: {df_baseline.shape}")
    print(f"New predictions: {df_new.shape}")
    
    # Check what samples are in each
    baseline_samples = set(df_baseline['antibody_name'])
    new_samples = set(df_new['antibody_name'])
    train_samples = set(df_train['antibody_name'])
    
    print(f"\nBaseline sample count: {len(baseline_samples)}")
    print(f"New sample count: {len(new_samples)}")
    print(f"Training data samples: {len(train_samples)}")
    
    missing_in_new = baseline_samples - new_samples
    extra_in_new = new_samples - baseline_samples
    
    if missing_in_new:
        print(f"\n⚠️  Samples in baseline but missing in new: {len(missing_in_new)}")
        print(f"  First 5: {list(missing_in_new)[:5]}")
    
    if extra_in_new:
        print(f"\n⚠️  Samples in new but missing in baseline: {len(extra_in_new)}")
        print(f"  First 5: {list(extra_in_new)[:5]}")
    
    # Compare on common samples
    common_samples = baseline_samples & new_samples
    print(f"\n✓ Common samples: {len(common_samples)}")
    
    if len(common_samples) == 0:
        print("❌ No common samples found!")
        return
    
    df_baseline_common = df_baseline[df_baseline['antibody_name'].isin(common_samples)].set_index('antibody_name')
    df_new_common = df_new[df_new['antibody_name'].isin(common_samples)].set_index('antibody_name')
    
    # Compare property values
    properties = [col for col in df_baseline_common.columns if col in PROPERTY_LIST]
    
    print(f"\n{'Property':<15} {'All Match?':<12} {'Max Diff':<15} {'Mean Abs Diff'}")
    print("-" * 60)
    
    all_match = True
    for prop in properties:
        if prop not in df_new_common.columns:
            print(f"{prop:<15} {'❌ MISSING':<12}")
            all_match = False
            continue
        
        baseline_vals = df_baseline_common[prop].values
        new_vals = df_new_common.loc[df_baseline_common.index, prop].values
        
        # Filter out NaNs
        mask = ~(np.isnan(baseline_vals) | np.isnan(new_vals))
        if not mask.any():
            print(f"{prop:<15} {'(no data)':<12}")
            continue
        
        baseline_vals = baseline_vals[mask]
        new_vals = new_vals[mask]
        
        differences = np.abs(baseline_vals - new_vals)
        max_diff = differences.max()
        mean_diff = differences.mean()
        match = np.allclose(baseline_vals, new_vals, rtol=1e-5, atol=1e-8)
        
        status = "✓" if match else "❌"
        print(f"{prop:<15} {status:<12} {max_diff:<15.2e} {mean_diff:.2e}")
        
        if not match:
            all_match = False
            # Show which samples have largest differences
            if max_diff > 1e-5:
                worst_idx = np.argmax(differences)
                worst_sample = df_baseline_common.index[worst_idx]
                print(f"    Worst: {worst_sample} has diff {differences[worst_idx]:.2e}")
                print(f"      Baseline: {baseline_vals[worst_idx]}")
                print(f"      New:      {new_vals[worst_idx]}")
    
    if all_match:
        print("\n✓ All property values match!")
        print("→ Differences in evaluation are due to fold split differences")
    else:
        print("\n❌ Property values differ")
        print("→ This suggests actual model/prediction changes")


def check_fold_assignments():
    """Check which fold each sample belongs to in baseline CV predictions."""
    
    baseline_results = Path(__file__).parent / "baseline_results"
    baseline_cv_pred = baseline_results / "predictions" / "GDPa1_cross_validation" / "TAP" / "TAP - linear regression.csv"
    train_data = Path(__file__).parent.parent / "data" / "GDPa1_v1.2_20250814.csv"
    
    print("\n" + "="*70)
    print("Analyzing Fold Assignments in Baseline CV Predictions")
    print("="*70)
    
    df_preds = pd.read_csv(baseline_cv_pred)
    df_train = pd.read_csv(train_data)
    
    # Merge to get fold info
    df_merged = df_train[['antibody_name', 'hierarchical_cluster_IgG_isotype_stratified_fold']].merge(
        df_preds[['antibody_name']], on='antibody_name', how='right'
    )
    
    # Count by fold
    fold_counts = df_merged['hierarchical_cluster_IgG_isotype_stratified_fold'].value_counts().sort_index()
    
    print(f"\nFold distribution in baseline CV predictions:")
    for fold, count in fold_counts.items():
        print(f"  Fold {fold}: {count} samples")
    
    total_folds = df_merged['hierarchical_cluster_IgG_isotype_stratified_fold'].nunique()
    print(f"\nTotal unique folds: {total_folds}")
    print(f"Total samples: {len(df_merged)}")
    
    # Check if there are any NaN folds (should be 0)
    nan_count = df_merged['hierarchical_cluster_IgG_isotype_stratified_fold'].isna().sum()
    if nan_count > 0:
        print(f"\n⚠️  {nan_count} samples have no fold assignment")


if __name__ == "__main__":
    check_fold_assignments()
    compare_cv_predictions()
