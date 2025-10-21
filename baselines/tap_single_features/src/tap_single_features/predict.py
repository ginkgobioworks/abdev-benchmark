"""Prediction module for TAP Single Features baseline."""

import argparse
from pathlib import Path
import pandas as pd


# Feature to property mappings from correlation analysis
# Format: feature_name: [(property, directionality), ...]
# Directionality: 1 = feature positively correlates, -1 = feature negatively correlates
FEATURE_MAPPINGS = {
    "PNC": [("AC-SINS_pH7.4", -1), ("PR_CHO", -1)],
    "SFvCSP": [("AC-SINS_pH7.4", 1), ("PR_CHO", 1), ("HIC", -1)],
    "PPC": [("AC-SINS_pH7.4", 1), ("Titer", 1)],
    "CDR Length": [("AC-SINS_pH7.4", -1), ("HIC", 1)],
}


def generate_predictions_for_feature(
    feature_name: str,
    df_sequences: pd.DataFrame,
    df_features: pd.DataFrame,
    output_dir: Path,
):
    """Generate predictions for a single TAP feature.
    
    Args:
        feature_name: Name of the TAP feature
        df_sequences: Dataframe with antibody sequences
        df_features: Dataframe with TAP features
        output_dir: Directory to write predictions
    """
    if feature_name not in FEATURE_MAPPINGS:
        print(f"  Skipping {feature_name} - no property mappings defined")
        return
    
    assay_mappings = FEATURE_MAPPINGS[feature_name]
    
    # Merge sequences with features
    df_merged = df_sequences.merge(
        df_features[["antibody_name", feature_name]], on="antibody_name", how="left"
    )
    
    # Apply directionality to create predictions
    assay_names = []
    for assay_name, directionality in assay_mappings:
        df_merged[assay_name] = df_merged[feature_name] * directionality
        assay_names.append(assay_name)
    
    # Select output columns
    output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"] + assay_names
    df_predictions = df_merged[output_cols]
    
    # Write output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"TAP - {feature_name}.csv"
    df_predictions.to_csv(output_file, index=False)
    print(f"  ✓ {output_file}")


def main():
    """Main entry point for predictions."""
    parser = argparse.ArgumentParser(description="TAP Single Features baseline predictions")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("../../data"),
        help="Path to data directory",
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("../../features/processed_features"),
        help="Path to features directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("../../predictions"),
        help="Path to output directory",
    )
    args = parser.parse_args()
    
    # Load sequence data
    print("Loading data...")
    df_gdpa1 = pd.read_csv(args.data_dir / "GDPa1_v1.2_20250814.csv")
    df_heldout = pd.read_csv(args.data_dir / "heldout-set-sequences.csv")
    
    # Load TAP features
    print("Loading TAP features...")
    df_tap_gdpa1 = pd.read_csv(args.features_dir / "GDPa1" / "TAP.csv")
    df_tap_heldout = pd.read_csv(args.features_dir / "heldout_test" / "TAP.csv")
    
    # Generate predictions for GDPa1 dataset
    print("Generating predictions for GDPa1...")
    for feature_name in FEATURE_MAPPINGS:
        generate_predictions_for_feature(
            feature_name,
            df_gdpa1,
            df_tap_gdpa1,
            args.output_dir / "GDPa1" / "TAP",
        )
    
    # Generate predictions for heldout test set
    print("Generating predictions for heldout_test...")
    for feature_name in FEATURE_MAPPINGS:
        generate_predictions_for_feature(
            feature_name,
            df_heldout,
            df_tap_heldout,
            args.output_dir / "heldout_test" / "TAP",
        )
    
    print("✓ All predictions complete")


if __name__ == "__main__":
    main()

