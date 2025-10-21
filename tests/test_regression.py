"""Regression tests comparing new predictions against baseline results."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def compare_predictions(
    reference_path: Path, new_path: Path, tolerance: float = 1e-6
) -> tuple[bool, list[str]]:
    """Compare two prediction files for regression testing.
    
    Args:
        reference_path: Path to reference (original) predictions
        new_path: Path to new predictions
        tolerance: Numeric tolerance for floating point comparisons
        
    Returns:
        Tuple of (all_passed, list of issues)
    """
    issues = []
    
    # Check both files exist
    if not reference_path.exists():
        issues.append(f"Reference file not found: {reference_path}")
        return False, issues
    
    if not new_path.exists():
        issues.append(f"New predictions not found: {new_path}")
        return False, issues
    
    # Load data
    try:
        df_ref = pd.read_csv(reference_path)
        df_new = pd.read_csv(new_path)
    except Exception as e:
        issues.append(f"Failed to load CSVs: {str(e)}")
        return False, issues
    
    # Check antibody names match
    ref_names = set(df_ref["antibody_name"])
    new_names = set(df_new["antibody_name"])
    
    if ref_names != new_names:
        missing_in_new = ref_names - new_names
        extra_in_new = new_names - ref_names
        if missing_in_new:
            issues.append(f"Missing {len(missing_in_new)} antibodies in new predictions")
        if extra_in_new:
            issues.append(f"Extra {len(extra_in_new)} antibodies in new predictions")
    
    # Merge on antibody_name for comparison
    df_merged = df_ref.merge(
        df_new, on="antibody_name", how="inner", suffixes=("_ref", "_new")
    )
    
    if len(df_merged) == 0:
        issues.append("No matching antibodies found for comparison")
        return False, issues
    
    # Compare numeric columns
    ref_cols = [col for col in df_ref.columns if col != "antibody_name" and df_ref[col].dtype in [np.float64, np.int64]]
    
    for col in ref_cols:
        if col in ["vh_protein_sequence", "vl_protein_sequence"]:
            continue  # Skip sequence columns
        
        ref_col = f"{col}_ref"
        new_col = f"{col}_new"
        
        if new_col not in df_merged.columns:
            issues.append(f"Column {col} missing in new predictions")
            continue
        
        # Get valid (non-NaN) values for comparison
        valid_mask = df_merged[ref_col].notna() & df_merged[new_col].notna()
        n_valid = valid_mask.sum()
        
        if n_valid == 0:
            continue
        
        ref_values = df_merged.loc[valid_mask, ref_col]
        new_values = df_merged.loc[valid_mask, new_col]
        
        # Check if values are close
        if not np.allclose(ref_values, new_values, rtol=tolerance, atol=tolerance):
            max_diff = np.abs(ref_values - new_values).max()
            issues.append(
                f"Column {col}: values differ (max diff: {max_diff:.2e}, tolerance: {tolerance:.2e})"
            )
    
    return len(issues) == 0, issues


def main():
    """Run regression tests on predictions."""
    parser = argparse.ArgumentParser(description="Regression testing for predictions")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("tests/baseline_results/predictions"),
        help="Directory with baseline predictions",
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("predictions"),
        help="Directory with new predictions",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numeric tolerance for comparisons",
    )
    args = parser.parse_args()
    
    print("Running regression tests...")
    print(f"Baseline: {args.baseline_dir}")
    print(f"New: {args.new_dir}")
    print(f"Tolerance: {args.tolerance}")
    print()
    
    # Find all prediction files in baseline
    all_passed = True
    tested = 0
    
    for dataset_dir in args.baseline_dir.iterdir():
        if not dataset_dir.is_dir():
            continue
        
        dataset_name = dataset_dir.name
        
        for model_dir in dataset_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            
            for pred_file in model_dir.glob("*.csv"):
                tested += 1
                ref_path = pred_file
                new_path = args.new_dir / dataset_name / model_name / pred_file.name
                
                print(f"Testing: {dataset_name}/{model_name}/{pred_file.name}")
                passed, issues = compare_predictions(ref_path, new_path, args.tolerance)
                
                if passed:
                    print("  ✓ PASS")
                else:
                    print("  ✗ FAIL")
                    for issue in issues:
                        print(f"    - {issue}")
                    all_passed = False
                print()
    
    print(f"Tested {tested} prediction files")
    
    if all_passed:
        print("✓ All regression tests passed!")
        return 0
    else:
        print("✗ Some regression tests failed")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

