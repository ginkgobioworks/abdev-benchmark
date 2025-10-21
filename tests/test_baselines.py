"""Pytest-based regression tests for baseline predictions.

This module provides pytest-compatible tests for validating that the
refactored baselines produce results consistent with the reference implementation.

Usage:
    pytest tests/test_baselines.py -v
    pytest tests/test_baselines.py -v -k tap_linear
    pytest tests/test_baselines.py -v --tolerance 1e-4
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from typing import List

from test_regression import BASELINE_CONFIG, compare_predictions


# Test configuration
BASELINE_DIR = Path(__file__).parent / "baseline_results" / "predictions"
NEW_DIR = Path(__file__).parent.parent / "predictions"
DEFAULT_TOLERANCE = 1e-6


def pytest_addoption(parser):
    """Add custom command line options for pytest."""
    parser.addoption(
        "--tolerance",
        action="store",
        default=DEFAULT_TOLERANCE,
        type=float,
        help="Numeric tolerance for comparisons (default: 1e-6)",
    )
    parser.addoption(
        "--baseline-dir",
        action="store",
        default=str(BASELINE_DIR),
        help=f"Directory with baseline predictions (default: {BASELINE_DIR})",
    )
    parser.addoption(
        "--new-dir",
        action="store",
        default=str(NEW_DIR),
        help=f"Directory with new predictions (default: {NEW_DIR})",
    )


@pytest.fixture
def tolerance(request):
    """Get tolerance value from command line or use default."""
    return request.config.getoption("--tolerance")


@pytest.fixture
def baseline_dir(request):
    """Get baseline directory from command line or use default."""
    return Path(request.config.getoption("--baseline-dir"))


@pytest.fixture
def new_dir(request):
    """Get new predictions directory from command line or use default."""
    return Path(request.config.getoption("--new-dir"))


def generate_test_cases():
    """Generate test cases from baseline configuration.
    
    Returns:
        List of tuples: (baseline_name, dataset, output_dir, filename, test_id)
    """
    test_cases = []
    for baseline_name, config in BASELINE_CONFIG.items():
        for dataset, output_dir in config["output_dirs"].items():
            if dataset not in config["files"]:
                continue
            for filename in config["files"][dataset]:
                test_id = f"{baseline_name}_{dataset}_{filename.replace(' ', '_').replace('.csv', '')}"
                test_cases.append((baseline_name, dataset, output_dir, filename, test_id))
    return test_cases


# Parametrize tests based on configuration
@pytest.mark.parametrize(
    "baseline_name,dataset,output_dir,filename,test_id",
    generate_test_cases(),
    ids=lambda x: x[-1] if isinstance(x, tuple) else str(x)
)
def test_baseline_prediction(
    baseline_name, dataset, output_dir, filename, test_id,
    baseline_dir, new_dir, tolerance
):
    """Test that a baseline prediction matches the reference.
    
    This test compares predictions from the refactored baseline against
    the reference predictions stored in tests/baseline_results/.
    """
    ref_path = baseline_dir / dataset / output_dir / filename
    new_path = new_dir / dataset / output_dir / filename
    
    # Compare predictions
    passed, issues = compare_predictions(ref_path, new_path, tolerance)
    
    # Format error message
    if not passed:
        error_msg = f"\n{baseline_name}/{dataset}/{filename} failed:\n"
        for issue in issues:
            error_msg += f"  - {issue}\n"
        pytest.fail(error_msg)


class TestBaselineGroups:
    """Grouped tests for each baseline to provide better organization."""
    
    @pytest.mark.tap_linear
    def test_tap_linear_group(self, baseline_dir, new_dir, tolerance):
        """Test all TAP Linear baseline predictions."""
        _test_baseline_group("tap_linear", baseline_dir, new_dir, tolerance)
    
    @pytest.mark.tap_single_features
    def test_tap_single_features_group(self, baseline_dir, new_dir, tolerance):
        """Test all TAP Single Features baseline predictions."""
        _test_baseline_group("tap_single_features", baseline_dir, new_dir, tolerance)
    
    @pytest.mark.aggrescan3d
    def test_aggrescan3d_group(self, baseline_dir, new_dir, tolerance):
        """Test all Aggrescan3D baseline predictions."""
        _test_baseline_group("aggrescan3d", baseline_dir, new_dir, tolerance)
    
    @pytest.mark.antifold
    def test_antifold_group(self, baseline_dir, new_dir, tolerance):
        """Test all AntiFold baseline predictions."""
        _test_baseline_group("antifold", baseline_dir, new_dir, tolerance)
    
    @pytest.mark.saprot_vh
    def test_saprot_vh_group(self, baseline_dir, new_dir, tolerance):
        """Test all Saprot_VH baseline predictions."""
        _test_baseline_group("saprot_vh", baseline_dir, new_dir, tolerance)
    
    @pytest.mark.deepviscosity
    def test_deepviscosity_group(self, baseline_dir, new_dir, tolerance):
        """Test all DeepViscosity baseline predictions."""
        _test_baseline_group("deepviscosity", baseline_dir, new_dir, tolerance)


def _test_baseline_group(baseline_name: str, baseline_dir: Path, new_dir: Path, tolerance: float):
    """Helper function to test all files for a baseline.
    
    Args:
        baseline_name: Name of the baseline
        baseline_dir: Path to baseline results directory
        new_dir: Path to new predictions directory
        tolerance: Numeric tolerance for comparisons
    """
    config = BASELINE_CONFIG[baseline_name]
    all_passed = True
    all_issues = []
    
    for dataset, output_dir in config["output_dirs"].items():
        if dataset not in config["files"]:
            continue
            
        for filename in config["files"][dataset]:
            ref_path = baseline_dir / dataset / output_dir / filename
            new_path = new_dir / dataset / output_dir / filename
            
            passed, issues = compare_predictions(ref_path, new_path, tolerance)
            
            if not passed:
                all_passed = False
                all_issues.append(f"{dataset}/{filename}:")
                all_issues.extend(f"  - {issue}" for issue in issues)
    
    if not all_passed:
        error_msg = f"\n{baseline_name} failed:\n" + "\n".join(all_issues)
        pytest.fail(error_msg)


# Smoke tests to ensure basic structure is correct
class TestStructure:
    """Tests to ensure the prediction directory structure is correct."""
    
    def test_predictions_directory_exists(self, new_dir):
        """Test that the predictions directory exists."""
        assert new_dir.exists(), f"Predictions directory not found: {new_dir}"
    
    def test_baseline_results_directory_exists(self, baseline_dir):
        """Test that the baseline results directory exists."""
        assert baseline_dir.exists(), f"Baseline results directory not found: {baseline_dir}"
    
    @pytest.mark.parametrize("baseline_name", list(BASELINE_CONFIG.keys()))
    def test_baseline_output_dirs_exist(self, baseline_name, new_dir):
        """Test that output directories for each baseline exist."""
        config = BASELINE_CONFIG[baseline_name]
        for dataset, output_dir in config["output_dirs"].items():
            expected_dir = new_dir / dataset / output_dir
            assert expected_dir.exists(), (
                f"Output directory not found for {baseline_name}: {expected_dir}\n"
                f"Did you run the baseline? Run: cd baselines/{baseline_name} && pixi run predict"
            )
    
    @pytest.mark.parametrize("baseline_name", list(BASELINE_CONFIG.keys()))
    def test_baseline_output_files_exist(self, baseline_name, new_dir):
        """Test that output files for each baseline exist."""
        config = BASELINE_CONFIG[baseline_name]
        missing_files = []
        
        for dataset, output_dir in config["output_dirs"].items():
            if dataset not in config["files"]:
                continue
            for filename in config["files"][dataset]:
                expected_file = new_dir / dataset / output_dir / filename
                if not expected_file.exists():
                    missing_files.append(str(expected_file))
        
        if missing_files:
            pytest.fail(
                f"Missing output files for {baseline_name}:\n" +
                "\n".join(f"  - {f}" for f in missing_files) +
                f"\n\nRun: cd baselines/{baseline_name} && pixi run predict"
            )


# Statistical tests for data quality
class TestDataQuality:
    """Tests to ensure prediction data quality."""
    
    @pytest.mark.parametrize("baseline_name", list(BASELINE_CONFIG.keys()))
    def test_no_all_nan_columns(self, baseline_name, new_dir):
        """Test that predictions don't have all-NaN property columns."""
        config = BASELINE_CONFIG[baseline_name]
        
        for dataset, output_dir in config["output_dirs"].items():
            if dataset not in config["files"]:
                continue
            for filename in config["files"][dataset]:
                pred_file = new_dir / dataset / output_dir / filename
                if not pred_file.exists():
                    pytest.skip(f"File not found: {pred_file}")
                
                df = pd.read_csv(pred_file)
                
                # Check property columns (not antibody_name or sequences)
                property_cols = [
                    col for col in df.columns 
                    if col not in ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
                ]
                
                for col in property_cols:
                    assert not df[col].isna().all(), (
                        f"{baseline_name}/{dataset}/{filename}: "
                        f"Column '{col}' is all NaN"
                    )
    
    @pytest.mark.parametrize("baseline_name", list(BASELINE_CONFIG.keys()))
    def test_antibody_names_present(self, baseline_name, new_dir):
        """Test that prediction files have antibody_name column and no missing values."""
        config = BASELINE_CONFIG[baseline_name]
        
        for dataset, output_dir in config["output_dirs"].items():
            if dataset not in config["files"]:
                continue
            for filename in config["files"][dataset]:
                pred_file = new_dir / dataset / output_dir / filename
                if not pred_file.exists():
                    pytest.skip(f"File not found: {pred_file}")
                
                df = pd.read_csv(pred_file)
                
                assert "antibody_name" in df.columns, (
                    f"{baseline_name}/{dataset}/{filename}: "
                    f"Missing 'antibody_name' column"
                )
                
                assert not df["antibody_name"].isna().any(), (
                    f"{baseline_name}/{dataset}/{filename}: "
                    f"Found NaN values in 'antibody_name' column"
                )

