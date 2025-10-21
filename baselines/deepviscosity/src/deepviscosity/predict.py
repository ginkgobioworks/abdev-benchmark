"""Prediction module for DeepViscosity baseline."""

import argparse
from pathlib import Path
import pandas as pd


# Feature to property mappings
FEATURE_MAPPINGS = {
    "Viscosity": [("HIC", 1)],
}


def main():
    """Main entry point for predictions."""
    parser = argparse.ArgumentParser(description="DeepViscosity baseline predictions")
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
    
    # Load DeepViscosity features (only available for GDPa1)
    print("Loading DeepViscosity features...")
    df_viscosity_gdpa1 = pd.read_csv(args.features_dir / "GDPa1" / "DeepViscosity.csv")
    
    # Process GDPa1
    print("Generating predictions for GDPa1...")
    df_merged = df_gdpa1.merge(
        df_viscosity_gdpa1[["antibody_name", "Viscosity"]], on="antibody_name", how="left"
    )
    
    # Map viscosity to HIC
    df_merged["HIC"] = df_merged["Viscosity"] * 1  # Positive correlation
    
    output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence", "HIC"]
    df_pred = df_merged[output_cols]
    
    output_dir = args.output_dir / "GDPa1" / "DeepViscosity"
    output_dir.mkdir(parents=True, exist_ok=True)
    df_pred.to_csv(output_dir / "DeepViscosity.csv", index=False)
    print(f"  ✓ {output_dir / 'DeepViscosity.csv'}")
    
    print("✓ All predictions complete")
    print("Note: DeepViscosity features only available for GDPa1 dataset")


if __name__ == "__main__":
    main()

