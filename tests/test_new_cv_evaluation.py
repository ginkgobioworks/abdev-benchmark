"""Test that new CV predictions produce the same metrics as baseline."""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from abdev_core import evaluate_model


def test_new_cv_metrics():
    """Test new CV predictions against training data."""
    
    new_cv_pred = project_root / "predictions" / "GDPa1_cross_validation" / "tap_linear" / "predictions.csv"
    train_data = project_root / "data" / "GDPa1_v1.2_20250814.csv"
    fold_col = "hierarchical_cluster_IgG_isotype_stratified_fold"
    
    print("\n" + "="*70)
    print("Testing New TAP Linear CV Predictions")
    print("="*70)
    
    # Test evaluation
    try:
        results = evaluate_model(
            preds_path=new_cv_pred,
            target_path=train_data,
            model_name="tap_linear",
            dataset_name="GDPa1_cross_validation",
            fold_col=fold_col,
            num_folds=5
        )
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Display results
    print("\nResults:")
    for result in results:
        print(f"\n{result['assay']}:")
        print(f"  Spearman:      {result['spearman']:.6f}")
        print(f"  Top 10 Recall: {result['top_10_recall']:.6f}")
    
    return True


if __name__ == "__main__":
    success = test_new_cv_metrics()
    sys.exit(0 if success else 1)
