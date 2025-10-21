"""Prediction module for Saprot_VH baseline."""

import argparse
from pathlib import Path
import pandas as pd


# Feature to property mappings
FEATURE_MAPPINGS = {
    "solubility_probability": [("PR_CHO", 1)],
    "stability_score": [("Tm2", -1)],  # Note: Negative correlation observed
}


def generate_predictions_for_feature(
    feature_name: str,
    df_sequences: pd.DataFrame,
    df_features: pd.DataFrame,
    output_dir: Path,
):
    """Generate predictions for a single Saprot feature.
    
    Args:
        feature_name: Name of the Saprot feature
        df_sequences: Dataframe with antibody sequences
        df_features: Dataframe with Saprot features
        output_dir: Directory to write predictions
    """
    if feature_name not in df_features.columns:
        print(f"  Warning: {feature_name} not found in features")
        return
    
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
    output_file = output_dir / f"Saprot_VH - {feature_name}.csv"
    df_predictions.to_csv(output_file, index=False)
    print(f"  ✓ {output_file.name}")


def main():
    """Main entry point for predictions."""
    parser = argparse.ArgumentParser(description="Saprot_VH baseline predictions")
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
    # Note: heldout_test doesn't have Saprot features in the current dataset
    
    # Load Saprot features (only available for GDPa1)
    print("Loading Saprot_VH features...")
    df_saprot_gdpa1 = pd.read_csv(args.features_dir / "GDPa1" / "Saprot_VH.csv")
    
    # Generate predictions for GDPa1 dataset
    print("Generating predictions for GDPa1...")
    for feature_name in FEATURE_MAPPINGS:
        generate_predictions_for_feature(
            feature_name,
            df_gdpa1,
            df_saprot_gdpa1,
            args.output_dir / "GDPa1" / "Saprot_VH",
        )
    
    print("✓ All predictions complete")
    print("Note: Saprot_VH features only available for GDPa1 dataset")


if __name__ == "__main__":
    main()

