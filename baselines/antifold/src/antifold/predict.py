"""Prediction module for AntiFold baseline."""

import argparse
from pathlib import Path
import pandas as pd


# Feature to property mappings
# Note: Negative correlations observed (marked in original code as "kinda strange")
FEATURE_MAPPINGS = {
    "Score": [("Tm2", -1), ("Titer", -1)],
}


def main():
    """Main entry point for predictions."""
    parser = argparse.ArgumentParser(description="AntiFold baseline predictions")
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
    
    # Load AntiFold features
    print("Loading AntiFold features...")
    df_antifold_gdpa1 = pd.read_csv(args.features_dir / "GDPa1" / "AntiFold.csv")
    df_antifold_heldout = pd.read_csv(args.features_dir / "heldout_test" / "AntiFold.csv")
    
    # Process GDPa1
    print("Generating predictions for GDPa1...")
    df_merged_gdpa1 = df_gdpa1.merge(
        df_antifold_gdpa1[["antibody_name", "Score"]], on="antibody_name", how="left"
    )
    
    for assay_name, directionality in FEATURE_MAPPINGS["Score"]:
        df_merged_gdpa1[assay_name] = df_merged_gdpa1["Score"] * directionality
    
    output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence", "Tm2", "Titer"]
    df_pred_gdpa1 = df_merged_gdpa1[output_cols]
    
    output_gdpa1_dir = args.output_dir / "GDPa1" / "AntiFold"
    output_gdpa1_dir.mkdir(parents=True, exist_ok=True)
    df_pred_gdpa1.to_csv(output_gdpa1_dir / "AntiFold.csv", index=False)
    print(f"  ✓ {output_gdpa1_dir / 'AntiFold.csv'}")
    
    # Process heldout test
    print("Generating predictions for heldout_test...")
    df_merged_heldout = df_heldout.merge(
        df_antifold_heldout[["antibody_name", "Score"]], on="antibody_name", how="left"
    )
    
    for assay_name, directionality in FEATURE_MAPPINGS["Score"]:
        df_merged_heldout[assay_name] = df_merged_heldout["Score"] * directionality
    
    df_pred_heldout = df_merged_heldout[output_cols]
    
    output_heldout_dir = args.output_dir / "heldout_test" / "AntiFold"
    output_heldout_dir.mkdir(parents=True, exist_ok=True)
    df_pred_heldout.to_csv(output_heldout_dir / "AntiFold.csv", index=False)
    print(f"  ✓ {output_heldout_dir / 'AntiFold.csv'}")
    
    print("✓ All predictions complete")


if __name__ == "__main__":
    main()

