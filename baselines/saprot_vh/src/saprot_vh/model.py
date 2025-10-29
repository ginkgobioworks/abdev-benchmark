"""Saprot_VH model implementation."""

from pathlib import Path
import json
import pandas as pd

from abdev_core import BaseModel, load_features


# Feature to property mappings
FEATURE_MAPPINGS = {
    "solubility_probability": [("PR_CHO", 1)],
    "stability_score": [("Tm2", -1)],  # Note: Negative correlation observed
}


class SaprotVhModel(BaseModel):
    """Saprot_VH baseline using protein language model features.

    This is a non-training baseline that directly maps Saprot features
    to predicted properties based on observed correlations.

    Features are loaded from the centralized feature store via abdev_core.
    """

    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """No-op training - this baseline uses pre-computed features.

        Saves configuration to run_dir for consistency.

        Args:
            df: Training dataframe (not used)
            run_dir: Directory to save configuration
            seed: Random seed (not used)
        """
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration for reference
        config = {
            "model_type": "saprot_vh",
            "feature_mappings": FEATURE_MAPPINGS,
            "note": "Non-training baseline using pre-computed Saprot_VH features",
        }

        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved configuration to {config_path}")
        print("Note: This is a non-training baseline using pre-computed features")

    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions using Saprot_VH features.

        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing configuration (not strictly needed)

        Returns:
            DataFrame with predictions for each property
        """
        # Load Saprot_VH features from centralized feature store (all datasets)
        try:
            saprot_features = load_features("Saprot_VH")
        except FileNotFoundError:
            print("Warning: Saprot_VH features not available")
            print("Generating empty predictions...")

            # Return empty predictions
            output_cols = [
                "antibody_name",
                "vh_protein_sequence",
                "vl_protein_sequence",
            ]
            df_output = df[output_cols].copy()
            return df_output

        # Merge sequences with features
        df_merged = df.copy()

        # Generate predictions for all mapped features
        all_properties = set()
        for feature_name, assay_mappings in FEATURE_MAPPINGS.items():
            if feature_name not in saprot_features.columns:
                print(f"Warning: {feature_name} not found in features, skipping")
                continue

            # Merge this feature
            df_feature = saprot_features[[feature_name]].reset_index()
            df_merged = df_merged.merge(df_feature, on="antibody_name", how="left")

            # Apply directionality to create predictions
            for assay_name, directionality in assay_mappings:
                df_merged[assay_name] = df_merged[feature_name] * directionality
                all_properties.add(assay_name)

        # Select output columns
        output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        output_cols.extend(sorted(all_properties))
        df_output = df_merged[output_cols]

        print(f"Generated predictions for {len(df_output)} samples")
        print(f"  Properties: {', '.join(sorted(all_properties))}")

        return df_output
