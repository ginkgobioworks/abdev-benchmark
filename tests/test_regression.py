"""Regression tests comparing new predictions against baseline results.

This script compares predictions from the current implementation against
reference predictions stored in tests/baseline_results/predictions/.

The baselines have been refactored to use Pixi multi-project architecture,
but they maintain the same output directory structure for compatibility.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import json


# Baseline configuration: maps baseline folder names to their output directories
# and the files they generate
BASELINE_CONFIG = {
    "tap_linear": {
        "output_dirs": {
            "GDPa1_cross_validation": "tap_linear",
            "heldout_test": "tap_linear"
        },
        "files": {
            "GDPa1_cross_validation": ["tap_linear.csv"],
            "heldout_test": ["tap_linear.csv"]
        },
        "reference_mapping": {
            # Maps new output to reference location
            "GDPa1_cross_validation": ("TAP", "TAP - linear regression.csv"),
            "heldout_test": ("TAP", "TAP - linear regression.csv")
        }
    },
    "tap_single_features": {
        "output_dirs": {
            "GDPa1": "TAP",
            "heldout_test": "TAP"
        },
        "files": {
            "GDPa1": [
                "TAP - CDR Length.csv",
                "TAP - PNC.csv",
                "TAP - PPC.csv",
                "TAP - SFvCSP.csv"
            ],
            "heldout_test": [
                "TAP - CDR Length.csv",
                "TAP - PNC.csv",
                "TAP - PPC.csv",
                "TAP - SFvCSP.csv"
            ]
        }
    },
    "aggrescan3d": {
        "output_dirs": {
            "GDPa1": "Aggrescan3D",
            "heldout_test": "Aggrescan3D"
        },
        "files": {
            "GDPa1": [
                "Aggrescan3D - aggrescan_90_score.csv",
                "Aggrescan3D - aggrescan_average_score.csv",
                "Aggrescan3D - aggrescan_cdrh3_average_score.csv",
                "Aggrescan3D - aggrescan_max_score.csv"
            ],
            "heldout_test": [
                "Aggrescan3D - aggrescan_90_score.csv",
                "Aggrescan3D - aggrescan_average_score.csv",
                "Aggrescan3D - aggrescan_cdrh3_average_score.csv",
                "Aggrescan3D - aggrescan_max_score.csv"
            ]
        }
    },
    "antifold": {
        "output_dirs": {
            "GDPa1": "AntiFold"
        },
        "files": {
            "GDPa1": ["AntiFold.csv"]
        }
    },
    "saprot_vh": {
        "output_dirs": {
            "GDPa1": "Saprot_VH"
        },
        "files": {
            "GDPa1": [
                "Saprot_VH - solubility_probability.csv",
                "Saprot_VH - stability_score.csv"
            ]
        }
    },
    "deepviscosity": {
        "output_dirs": {
            "GDPa1": "DeepViscosity"
        },
        "files": {
            "GDPa1": ["DeepViscosity.csv"]
        }
    }
}


def compare_predictions(
    reference_path: Path, new_path: Path, tolerance: float = 1e-6
) -> Tuple[bool, List[str]]:
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
    
    # Compare numeric columns (property predictions)
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
            mean_diff = np.abs(ref_values - new_values).mean()
            issues.append(
                f"Column {col}: values differ (max diff: {max_diff:.2e}, "
                f"mean diff: {mean_diff:.2e}, tolerance: {tolerance:.2e})"
            )
    
    return len(issues) == 0, issues


def test_baseline(
    baseline_name: str,
    baseline_dir: Path,
    new_dir: Path,
    tolerance: float,
    verbose: bool = False
) -> Tuple[int, int, List[str]]:
    """Test a single baseline against reference predictions.
    
    Args:
        baseline_name: Name of the baseline (e.g., 'tap_linear')
        baseline_dir: Path to baseline results directory
        new_dir: Path to new predictions directory
        tolerance: Numeric tolerance for comparisons
        verbose: Whether to print verbose output
        
    Returns:
        Tuple of (passed_count, total_count, list of failure messages)
    """
    if baseline_name not in BASELINE_CONFIG:
        return 0, 0, [f"Unknown baseline: {baseline_name}"]
    
    config = BASELINE_CONFIG[baseline_name]
    passed = 0
    total = 0
    failures = []
    
    # Check if we have reference_mapping for backward compatibility
    reference_mapping = config.get("reference_mapping", {})
    
    for dataset, output_dir in config["output_dirs"].items():
        if dataset not in config["files"]:
            continue
            
        for filename in config["files"][dataset]:
            total += 1
            
            # Use reference mapping if available for this dataset
            if dataset in reference_mapping:
                ref_output_dir, ref_filename = reference_mapping[dataset]
                ref_path = baseline_dir / dataset / ref_output_dir / ref_filename
            else:
                ref_path = baseline_dir / dataset / output_dir / filename
            
            new_path = new_dir / dataset / output_dir / filename
            
            test_name = f"{baseline_name}/{dataset}/{filename}"
            
            if verbose:
                print(f"  Testing: {test_name}")
                if dataset in reference_mapping:
                    print(f"    Reference: {ref_path.relative_to(baseline_dir)}")
                    print(f"    Current: {new_path.relative_to(new_dir)}")
            
            is_passed, issues = compare_predictions(ref_path, new_path, tolerance)
            
            if is_passed:
                passed += 1
                if verbose:
                    print("    ✓ PASS")
            else:
                failure_msg = f"{test_name}:"
                for issue in issues:
                    failure_msg += f"\n    - {issue}"
                failures.append(failure_msg)
                if verbose:
                    print("    ✗ FAIL")
                    for issue in issues:
                        print(f"      - {issue}")
    
    return passed, total, failures


def main():
    """Run regression tests on predictions."""
    parser = argparse.ArgumentParser(
        description="Regression testing for predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test all baselines
  python test_regression.py
  
  # Test specific baselines
  python test_regression.py --baselines tap_linear aggrescan3d
  
  # Use higher tolerance for numeric comparisons
  python test_regression.py --tolerance 1e-4
  
  # Show verbose output
  python test_regression.py --verbose
        """
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("tests/baseline_results/predictions"),
        help="Directory with baseline predictions (default: tests/baseline_results/predictions)",
    )
    parser.add_argument(
        "--new-dir",
        type=Path,
        default=Path("predictions"),
        help="Directory with new predictions (default: predictions)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-6,
        help="Numeric tolerance for comparisons (default: 1e-6)",
    )
    parser.add_argument(
        "--baselines",
        nargs="+",
        choices=list(BASELINE_CONFIG.keys()),
        default=list(BASELINE_CONFIG.keys()),
        help="Specific baselines to test (default: all)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show verbose output",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="List available baselines and exit",
    )
    args = parser.parse_args()
    
    # List baselines if requested
    if args.list_baselines:
        print("Available baselines:")
        for baseline in sorted(BASELINE_CONFIG.keys()):
            config = BASELINE_CONFIG[baseline]
            total_files = sum(len(files) for files in config["files"].values())
            print(f"  - {baseline} ({total_files} files)")
        return 0
    
    print("=" * 70)
    print("Antibody Developability Benchmark - Regression Tests")
    print("=" * 70)
    print(f"Reference: {args.baseline_dir}")
    print(f"Current:   {args.new_dir}")
    print(f"Tolerance: {args.tolerance}")
    print(f"Baselines: {', '.join(args.baselines)}")
    print("=" * 70)
    print()
    
    # Run tests for each baseline
    all_passed = True
    total_tested = 0
    total_passed = 0
    all_failures = []
    
    for baseline_name in args.baselines:
        print(f"Testing {baseline_name}...")
        passed, total, failures = test_baseline(
            baseline_name,
            args.baseline_dir,
            args.new_dir,
            args.tolerance,
            args.verbose
        )
        
        total_tested += total
        total_passed += passed
        
        if failures:
            all_passed = False
            all_failures.extend(failures)
        
        # Print summary for this baseline
        status = "✓ PASS" if not failures else "✗ FAIL"
        print(f"  {status} ({passed}/{total} files)")
        print()
    
    # Print overall summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Total files tested: {total_tested}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_tested - total_passed}")
    print()
    
    if all_passed:
        print("✓ All regression tests passed!")
        return 0
    else:
        print("✗ Some regression tests failed:")
        print()
        for failure in all_failures:
            print(failure)
            print()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())

