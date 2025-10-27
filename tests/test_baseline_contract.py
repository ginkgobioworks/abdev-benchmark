#!/usr/bin/env python3
"""Test script to validate all baselines satisfy the train/predict contract.

This script:
1. Discovers all baseline directories
2. For each baseline, runs train and predict commands
3. Validates that outputs are created correctly
4. Checks that predictions follow the expected format

Usage:
    python tests/test_baseline_contract.py
    python tests/test_baseline_contract.py --baseline tap_linear
    python tests/test_baseline_contract.py --skip-train  # Only test predict
"""

import argparse
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys
import pandas as pd

# Color codes for output
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def print_section(message: str):
    """Print a section header."""
    print(f"\n{BLUE}{'=' * 80}")
    print(f"{message}")
    print(f"{'=' * 80}{NC}\n")


def print_success(message: str):
    """Print a success message."""
    print(f"{GREEN}✓ {message}{NC}")


def print_error(message: str):
    """Print an error message."""
    print(f"{RED}✗ {message}{NC}")


def print_warning(message: str):
    """Print a warning message."""
    print(f"{YELLOW}⚠ {message}{NC}")


def discover_baselines(baselines_dir: Path) -> list[str]:
    """Discover all baseline directories.
    
    Args:
        baselines_dir: Path to baselines directory
        
    Returns:
        List of baseline names
    """
    baselines = []
    for item in baselines_dir.iterdir():
        if item.is_dir() and (item / "pixi.toml").exists():
            baselines.append(item.name)
    return sorted(baselines)


def test_baseline_train(baseline_name: str, baselines_dir: Path, data_path: Path, temp_dir: Path) -> tuple[bool, str]:
    """Test the train command for a baseline.
    
    Args:
        baseline_name: Name of the baseline
        baselines_dir: Path to baselines directory
        data_path: Path to training data
        temp_dir: Temporary directory for outputs
        
    Returns:
        Tuple of (success, message)
    """
    baseline_dir = baselines_dir / baseline_name
    run_dir = temp_dir / "runs" / baseline_name
    
    print(f"  Training {baseline_name}...")
    
    # Build command (use pixi run to activate baseline's environment)
    cmd = [
        "pixi", "run", "python", "-m", baseline_name.replace("-", "_"),
        "train",
        "--data", str(data_path),
        "--run-dir", str(run_dir),
        "--seed", "42"
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=baseline_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            return False, f"Train command failed with code {result.returncode}\nStderr: {result.stderr}"
        
        # Check that run_dir was created
        if not run_dir.exists():
            return False, f"Run directory not created: {run_dir}"
        
        # Check that at least one file was created
        files = list(run_dir.glob("*"))
        if not files:
            return False, "No artifacts saved to run_dir"
        
        return True, f"Trained successfully, created {len(files)} artifacts"
        
    except subprocess.TimeoutExpired:
        return False, "Train command timed out (>5 minutes)"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def test_baseline_predict(
    baseline_name: str,
    baselines_dir: Path,
    data_path: Path,
    run_dir: Path,
    out_dir: Path,
    is_training_data: bool = True
) -> tuple[bool, str]:
    """Test the predict command for a baseline.
    
    Args:
        baseline_name: Name of the baseline
        baselines_dir: Path to baselines directory
        data_path: Path to input data
        run_dir: Path to run directory with trained models
        out_dir: Directory for output predictions
        is_training_data: Whether this is training or heldout data
        
    Returns:
        Tuple of (success, message)
    """
    baseline_dir = baselines_dir / baseline_name
    dataset_name = "training" if is_training_data else "heldout"
    
    print(f"  Predicting on {dataset_name} data...")
    
    # Build command (use pixi run to activate baseline's environment)
    cmd = [
        "pixi", "run", "python", "-m", baseline_name.replace("-", "_"),
        "predict",
        "--data", str(data_path),
        "--run-dir", str(run_dir),
        "--out-dir", str(out_dir)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            cwd=baseline_dir,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        if result.returncode != 0:
            return False, f"Predict command failed with code {result.returncode}\nStderr: {result.stderr}"
        
        # Check that predictions.csv was created
        predictions_path = out_dir / "predictions.csv"
        if not predictions_path.exists():
            return False, f"Predictions file not created: {predictions_path}"
        
        # Validate predictions format
        try:
            df_pred = pd.read_csv(predictions_path)
        except Exception as e:
            return False, f"Failed to read predictions CSV: {str(e)}"
        
        # Check required columns
        required_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        missing_cols = [col for col in required_cols if col not in df_pred.columns]
        if missing_cols:
            return False, f"Missing required columns: {missing_cols}"
        
        # Check that at least one property column exists
        property_cols = [col for col in df_pred.columns if col not in required_cols]
        if not property_cols:
            print_warning("No property predictions found (only sequence columns)")
        
        # Load input data to check row count
        df_input = pd.read_csv(data_path)
        if len(df_pred) != len(df_input):
            return False, f"Row count mismatch: {len(df_pred)} predictions vs {len(df_input)} input samples"
        
        return True, f"Generated predictions for {len(df_pred)} samples with {len(property_cols)} properties"
        
    except subprocess.TimeoutExpired:
        return False, "Predict command timed out (>5 minutes)"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def test_baseline(
    baseline_name: str,
    baselines_dir: Path,
    data_dir: Path,
    temp_dir: Path,
    skip_train: bool = False
) -> dict:
    """Test a single baseline for train/predict contract compliance.
    
    Args:
        baseline_name: Name of the baseline
        baselines_dir: Path to baselines directory
        data_dir: Path to data directory
        temp_dir: Temporary directory for outputs
        skip_train: If True, skip training step
        
    Returns:
        Dictionary with test results
    """
    print_section(f"Testing: {baseline_name}")
    
    results = {
        "baseline": baseline_name,
        "train_passed": None,
        "train_message": None,
        "predict_train_passed": None,
        "predict_train_message": None,
        "predict_heldout_passed": None,
        "predict_heldout_message": None,
        "overall_passed": False
    }
    
    # Paths
    train_data_path = data_dir / "GDPa1_v1.2_20250814.csv"
    heldout_data_path = data_dir / "heldout-set-sequences.csv"
    run_dir = temp_dir / "runs" / baseline_name
    out_dir_train = temp_dir / "predictions" / baseline_name / "train"
    out_dir_heldout = temp_dir / "predictions" / baseline_name / "heldout"
    
    # Test 1: Train
    if not skip_train:
        success, message = test_baseline_train(baseline_name, baselines_dir, train_data_path, temp_dir)
        results["train_passed"] = success
        results["train_message"] = message
        
        if success:
            print_success(message)
        else:
            print_error(f"Train failed: {message}")
            return results
    else:
        print_warning("Skipping train (--skip-train flag set)")
        results["train_passed"] = True
        results["train_message"] = "Skipped"
    
    # Test 2: Predict on training data
    success, message = test_baseline_predict(
        baseline_name, baselines_dir, train_data_path, run_dir, out_dir_train, is_training_data=True
    )
    results["predict_train_passed"] = success
    results["predict_train_message"] = message
    
    if success:
        print_success(message)
    else:
        print_error(f"Predict (training data) failed: {message}")
        return results
    
    # Test 3: Predict on heldout data
    success, message = test_baseline_predict(
        baseline_name, baselines_dir, heldout_data_path, run_dir, out_dir_heldout, is_training_data=False
    )
    results["predict_heldout_passed"] = success
    results["predict_heldout_message"] = message
    
    if success:
        print_success(message)
    else:
        print_error(f"Predict (heldout data) failed: {message}")
        return results
    
    # All tests passed
    results["overall_passed"] = True
    print_success(f"{baseline_name} passed all tests!")
    
    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test all baselines for train/predict contract compliance"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        help="Test only this specific baseline"
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training step (assumes models already trained)"
    )
    parser.add_argument(
        "--baselines-dir",
        type=Path,
        default=Path(__file__).parent.parent / "baselines",
        help="Path to baselines directory"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data",
        help="Path to data directory"
    )
    args = parser.parse_args()
    
    # Validate paths
    if not args.baselines_dir.exists():
        print_error(f"Baselines directory not found: {args.baselines_dir}")
        sys.exit(1)
    
    if not args.data_dir.exists():
        print_error(f"Data directory not found: {args.data_dir}")
        sys.exit(1)
    
    # Discover baselines
    if args.baseline:
        baselines = [args.baseline]
    else:
        baselines = discover_baselines(args.baselines_dir)
    
    print_section(f"Found {len(baselines)} baseline(s) to test")
    for baseline in baselines:
        print(f"  - {baseline}")
    
    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory(prefix="baseline_test_") as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\nUsing temporary directory: {temp_path}")
        
        # Test each baseline
        all_results = []
        for baseline in baselines:
            results = test_baseline(
                baseline,
                args.baselines_dir,
                args.data_dir,
                temp_path,
                skip_train=args.skip_train
            )
            all_results.append(results)
        
        # Print summary
        print_section("Summary")
        
        passed = [r for r in all_results if r["overall_passed"]]
        failed = [r for r in all_results if not r["overall_passed"]]
        
        print(f"Total baselines tested: {len(all_results)}")
        print_success(f"Passed: {len(passed)}/{len(all_results)}")
        
        if passed:
            for r in passed:
                print(f"  {GREEN}✓{NC} {r['baseline']}")
        
        if failed:
            print_error(f"Failed: {len(failed)}/{len(all_results)}")
            for r in failed:
                print(f"  {RED}✗{NC} {r['baseline']}")
                if not r["train_passed"]:
                    print(f"      Train: {r['train_message']}")
                elif not r["predict_train_passed"]:
                    print(f"      Predict (train): {r['predict_train_message']}")
                elif not r["predict_heldout_passed"]:
                    print(f"      Predict (heldout): {r['predict_heldout_message']}")
        
        # Exit with appropriate code
        sys.exit(0 if len(failed) == 0 else 1)


if __name__ == "__main__":
    main()

