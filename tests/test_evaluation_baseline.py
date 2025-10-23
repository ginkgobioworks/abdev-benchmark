"""Test that evaluation metrics match the known baseline results.

This script validates that the evaluation.metrics module produces identical results
to the baseline evaluation data stored in tests/baseline_results/results/.

This helps us confirm that:
1. The evaluation code is correct
2. Any differences in CV results are due to different fold assignments or predictions
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abdev_core import PROPERTY_LIST, evaluate_model


def test_tap_linear_cv_metrics():
    """Test that TAP linear CV evaluation produces baseline results."""
    
    # Paths
    baseline_results = Path(__file__).parent / "baseline_results"
    baseline_cv_pred_file = baseline_results / "predictions" / "GDPa1_cross_validation" / "TAP" / "TAP - linear regression.csv"
    baseline_cv_metrics = baseline_results / "results" / "GDPa1_cross_validation" / "TAP - linear regression.csv"
    train_data = project_root / "data" / "GDPa1_v1.2_20250814.csv"
    
    if not all([baseline_cv_pred_file.exists(), baseline_cv_metrics.exists(), train_data.exists()]):
        print("❌ Required test files not found")
        print(f"  Pred file exists: {baseline_cv_pred_file.exists()}")
        print(f"  Metrics file exists: {baseline_cv_metrics.exists()}")
        print(f"  Train data exists: {train_data.exists()}")
        return False
    
    # Load expected metrics
    expected_df = pd.read_csv(baseline_cv_metrics)
    expected_results = {}
    for _, row in expected_df.iterrows():
        expected_results[row['assay']] = {
            'spearman': row['spearman'],
            'top_10_recall': row['top_10_recall']
        }
    
    print("\n" + "="*70)
    print("Testing TAP Linear CV Evaluation")
    print("="*70)
    
    # Compute metrics using evaluation code
    try:
        computed_results = evaluate_model(
            preds_path=baseline_cv_pred_file,
            target_path=train_data,
            model_name="tap_linear_test",
            dataset_name="GDPa1_cross_validation",
            fold_col="hierarchical_cluster_IgG_isotype_stratified_fold",
            num_folds=5
        )
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Convert to dict for comparison
    computed_dict = {}
    for result in computed_results:
        computed_dict[result['assay']] = {
            'spearman': result['spearman'],
            'top_10_recall': result['top_10_recall']
        }
    
    # Compare
    all_match = True
    for assay in sorted(expected_results.keys()):
        if assay not in computed_dict:
            print(f"\n❌ {assay}: Missing from computed results")
            all_match = False
            continue
        
        exp_sp = expected_results[assay]['spearman']
        comp_sp = computed_dict[assay]['spearman']
        exp_recall = expected_results[assay]['top_10_recall']
        comp_recall = computed_dict[assay]['top_10_recall']
        
        sp_match = np.isclose(exp_sp, comp_sp, rtol=1e-5)
        recall_match = np.isclose(exp_recall, comp_recall, rtol=1e-5)
        
        status = "✓" if (sp_match and recall_match) else "❌"
        print(f"\n{status} {assay}:")
        print(f"    Spearman:      {exp_sp:.6f} → {comp_sp:.6f} {' ✓' if sp_match else ' ❌'}")
        print(f"    Top 10 Recall: {exp_recall:.6f} → {comp_recall:.6f} {' ✓' if recall_match else ' ❌'}")
        
        if not (sp_match and recall_match):
            all_match = False
    
    return all_match


def diagnose_cv_predictions():
    """Diagnose what's in the baseline CV predictions file."""
    
    baseline_results = Path(__file__).parent / "baseline_results"
    baseline_cv_pred_file = baseline_results / "predictions" / "GDPa1_cross_validation" / "TAP" / "TAP - linear regression.csv"
    
    if not baseline_cv_pred_file.exists():
        print(f"❌ File not found: {baseline_cv_pred_file}")
        return
    
    df = pd.read_csv(baseline_cv_pred_file)
    print("\n" + "="*70)
    print("Baseline CV Predictions Structure")
    print("="*70)
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nData types:")
    print(df.dtypes)
    print(f"\nProperty columns: {[c for c in df.columns if c in PROPERTY_LIST]}")
    

if __name__ == "__main__":
    diagnose_cv_predictions()
    
    success = test_tap_linear_cv_metrics()
    
    if success:
        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED - Evaluation code is correct!")
        print("="*70)
        sys.exit(0)
    else:
        print("\n" + "="*70)
        print("❌ TESTS FAILED - Evaluation metrics don't match")
        print("="*70)
        sys.exit(1)
