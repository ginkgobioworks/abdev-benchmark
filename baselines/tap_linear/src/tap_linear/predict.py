"""Prediction module for TAP Linear baseline."""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import PredefinedSplit

from abdev_core import PROPERTY_LIST, FOLD_COL


# TAP feature names used for modeling
FEATURE_NAMES = ["SFvCSP", "PSH", "PPC", "PNC", "CDR Length"]


def get_cross_validation_predictions(
    df: pd.DataFrame, property_name: str, df_heldout: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Run cross-validation predictions for a single property.
    
    Args:
        df: Training dataframe with features and labels
        property_name: Name of the property to predict
        df_heldout: Held-out dataframe for final predictions
        
    Returns:
        Tuple of (cv_predictions, heldout_predictions)
    """
    model = Ridge()
    df_out = df.copy()
    splits = PredefinedSplit(test_fold=df[FOLD_COL])
    
    for i, (train_index, test_index) in enumerate(splits.split()):
        subset = df.iloc[train_index].dropna(subset=[property_name])
        model.fit(subset[FEATURE_NAMES], subset[property_name])
        df_out.loc[test_index, "pred_cv"] = model.predict(df.iloc[test_index][FEATURE_NAMES])
    
    # Re-fit whole model to get heldout predictions
    model.fit(df[FEATURE_NAMES], df[property_name])
    df_heldout["pred_heldout"] = model.predict(df_heldout[FEATURE_NAMES])
    
    return df_out["pred_cv"].values, df_heldout["pred_heldout"].values


def predict_all_properties(
    df: pd.DataFrame, df_heldout: pd.DataFrame, df_tap_train: pd.DataFrame, df_tap_heldout: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate predictions for all properties.
    
    Args:
        df: Ground truth dataframe
        df_heldout: Held-out sequences dataframe
        df_tap_train: TAP features for training set
        df_tap_heldout: TAP features for held-out set
        
    Returns:
        Tuple of (cv_predictions_df, heldout_predictions_df)
    """
    df_merged = df.merge(df_tap_train, on="antibody_name", how="left")
    df_merged_heldout = df_heldout.merge(df_tap_heldout, on="antibody_name", how="left")
    
    df_out_cv = df_merged.copy()
    
    for property_name in PROPERTY_LIST:
        not_na_mask = df_merged[property_name].notna()
        df_merged_subset = df_merged[not_na_mask].reset_index(drop=True)
        cv_preds, heldout_preds = get_cross_validation_predictions(
            df_merged_subset, property_name, df_merged_heldout
        )
        df_out_cv.loc[not_na_mask, property_name] = cv_preds
        df_merged_heldout[property_name] = heldout_preds
    
    # Return only required columns
    output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"] + PROPERTY_LIST
    return df_out_cv[output_cols], df_merged_heldout[output_cols]


def main():
    """Main entry point for predictions."""
    parser = argparse.ArgumentParser(description="TAP Linear baseline predictions")
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
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(args.data_dir / "GDPa1_v1.2_20250814.csv")
    df_heldout = pd.read_csv(args.data_dir / "heldout-set-sequences.csv")
    
    # Load TAP features
    print("Loading TAP features...")
    df_tap_train = pd.read_csv(args.features_dir / "GDPa1" / "TAP.csv")
    df_tap_heldout = pd.read_csv(args.features_dir / "heldout_test" / "TAP.csv")
    
    # Generate predictions
    print("Generating predictions...")
    df_predictions_cv, df_predictions_heldout = predict_all_properties(
        df, df_heldout, df_tap_train, df_tap_heldout
    )
    
    # Write outputs
    print("Writing predictions...")
    output_cv_dir = args.output_dir / "GDPa1_cross_validation" / "tap_linear"
    output_heldout_dir = args.output_dir / "heldout_test" / "tap_linear"
    
    output_cv_dir.mkdir(parents=True, exist_ok=True)
    output_heldout_dir.mkdir(parents=True, exist_ok=True)
    
    df_predictions_cv.to_csv(output_cv_dir / "tap_linear.csv", index=False)
    df_predictions_heldout.to_csv(output_heldout_dir / "tap_linear.csv", index=False)
    
    print("✓ Predictions complete")
    print(f"  CV predictions: {output_cv_dir / 'tap_linear.csv'}")
    print(f"  Heldout predictions: {output_heldout_dir / 'tap_linear.csv'}")


if __name__ == "__main__":
    main()

