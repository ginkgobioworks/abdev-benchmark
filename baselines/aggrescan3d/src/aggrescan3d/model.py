"""Aggrescan3D model implementation."""

from pathlib import Path
import json
import pandas as pd

from abdev_core import BaseModel, load_features


# Feature to property mappings
# Format: feature_name: [(property, directionality), ...]
FEATURE_MAPPINGS = {
    "aggrescan_average_score": [("HIC", 1)],
    "aggrescan_max_score": [("HIC", 1), ("PR_CHO", -1)],
    "aggrescan_90_score": [("HIC", 1)],
    "aggrescan_cdrh3_average_score": [("HIC", 1)],
}


class Aggrescan3dModel(BaseModel):
    """Aggrescan3D baseline using pre-computed aggregation propensity features.

    This is a non-training baseline that directly maps Aggrescan3D features
    to predicted properties based on known correlations.

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
            "model_type": "aggrescan3d",
            "feature_mappings": FEATURE_MAPPINGS,
            "note": "Non-training baseline using pre-computed Aggrescan3D features",
        }

        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved configuration to {config_path}")
        print("Note: This is a non-training baseline using pre-computed features")

    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions using Aggrescan3D features.

        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing configuration (not strictly needed)

        Returns:
            DataFrame with predictions for each property
        """
        # Load Aggrescan3D features from centralized feature store (all datasets)
        aggrescan_features = load_features("Aggrescan3D")

        # Generate predictions for all mapped features
        all_predictions = []

        for feature_name, assay_mappings in FEATURE_MAPPINGS.items():
            if feature_name not in aggrescan_features.columns:
                print(f"Warning: {feature_name} not found in features, skipping")
                continue

            # Merge sequences with features
            df_merged = df.merge(
                aggrescan_features[[feature_name]].reset_index(),
                on="antibody_name",
                how="left",
            )

            # Apply directionality to create predictions
            for assay_name, directionality in assay_mappings:
                df_merged[f"{assay_name}_from_{feature_name}"] = (
                    df_merged[feature_name] * directionality
                )

            all_predictions.append(df_merged)

        # For now, we'll output the first mapping for each property
        # (This matches the original behavior where multiple features map to same properties)
        df_output = df[
            ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        ].copy()

        # Collect predictions from the first available feature for each property
        property_sources = {}
        for feature_name, assay_mappings in FEATURE_MAPPINGS.items():
            for assay_name, _ in assay_mappings:
                if assay_name not in property_sources:
                    property_sources[assay_name] = feature_name

        # Merge predictions
        for assay_name, feature_name in property_sources.items():
            col_name = f"{assay_name}_from_{feature_name}"
            for df_pred in all_predictions:
                if col_name in df_pred.columns:
                    df_output[assay_name] = df_pred[col_name]
                    break

        print(f"Generated predictions for {len(df_output)} samples")
        print(f"  Properties: {', '.join(property_sources.keys())}")

        return df_output
