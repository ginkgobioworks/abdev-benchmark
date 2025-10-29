"""AntiFold model implementation."""

from pathlib import Path
import json
import pandas as pd

from abdev_core import BaseModel, load_features


# Feature to property mappings
# Note: Negative correlations observed
FEATURE_MAPPINGS = {
    "Score": [("Tm2", -1), ("Titer", -1)],
}


class AntifoldModel(BaseModel):
    """AntiFold baseline using pre-computed antibody stability predictions.

    This is a non-training baseline that directly maps AntiFold scores
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
            "model_type": "antifold",
            "feature_mappings": FEATURE_MAPPINGS,
            "note": "Non-training baseline using pre-computed AntiFold stability scores",
        }

        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"Saved configuration to {config_path}")
        print("Note: This is a non-training baseline using pre-computed features")

    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions using AntiFold features.

        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing configuration (not strictly needed)

        Returns:
            DataFrame with predictions for each property
        """
        # Load AntiFold features from centralized feature store (all datasets)
        antifold_features = load_features("AntiFold")

        # Merge sequences with features
        df_merged = df.merge(
            antifold_features[["Score"]].reset_index(), on="antibody_name", how="left"
        )

        # Apply directionality to create predictions
        for assay_name, directionality in FEATURE_MAPPINGS["Score"]:
            df_merged[assay_name] = df_merged["Score"] * directionality

        # Select output columns
        output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        output_cols.extend([assay for assay, _ in FEATURE_MAPPINGS["Score"]])
        df_output = df_merged[output_cols]

        print(f"Generated predictions for {len(df_output)} samples")
        print(
            f"  Properties: {', '.join([assay for assay, _ in FEATURE_MAPPINGS['Score']])}"
        )

        return df_output
