"""TAP Single Features model implementation."""

from pathlib import Path
import json
import pandas as pd

from abdev_core import BaseModel, load_features


# Feature to property mappings from correlation analysis
# Format: feature_name: [(property, directionality), ...]
# Directionality: 1 = feature positively correlates, -1 = feature negatively correlates
FEATURE_MAPPINGS = {
    "PNC": [("AC-SINS_pH7.4", -1), ("PR_CHO", -1)],
    "SFvCSP": [("AC-SINS_pH7.4", 1), ("PR_CHO", 1), ("HIC", -1)],
    "PPC": [("AC-SINS_pH7.4", 1), ("Titer", 1)],
    "CDR Length": [("AC-SINS_pH7.4", -1), ("HIC", 1)],
}


class TapSingleFeaturesModel(BaseModel):
    """TAP Single Features baseline using individual TAP features as direct predictors.
    
    This is a non-training baseline that maps individual TAP features
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
            "model_type": "tap_single_features",
            "feature_mappings": FEATURE_MAPPINGS,
            "note": "Non-training baseline using individual TAP features"
        }
        
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved configuration to {config_path}")
        print("Note: This is a non-training baseline using pre-computed features")
    
    def predict(self, df: pd.DataFrame, run_dir: Path, out_dir: Path) -> None:
        """Generate predictions using TAP single features.
        
        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing configuration (not strictly needed)
            out_dir: Directory to write predictions.csv
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Load TAP features from centralized feature store
        dataset = "GDPa1"
        tap_features = load_features("TAP", dataset=dataset)
        
        # Merge sequences with features
        df_merged = df.copy()
        
        # Generate predictions for all mapped features
        all_properties = set()
        for feature_name, assay_mappings in FEATURE_MAPPINGS.items():
            if feature_name not in tap_features.columns:
                print(f"Warning: {feature_name} not found in TAP features, skipping")
                continue
            
            # Merge this feature
            df_feature = tap_features[[feature_name]].reset_index()
            df_merged = df_merged.merge(df_feature, on="antibody_name", how="left")
            
            # Apply directionality to create predictions
            # For each property, use the first feature that maps to it
            for assay_name, directionality in assay_mappings:
                # Only set if not already set by a previous feature
                if assay_name not in all_properties:
                    df_merged[assay_name] = df_merged[feature_name] * directionality
                    all_properties.add(assay_name)
        
        # Select output columns
        output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        output_cols.extend(sorted(all_properties))
        df_output = df_merged[output_cols]
        
        # Write predictions
        output_path = out_dir / "predictions.csv"
        df_output.to_csv(output_path, index=False)
        
        print(f"Generated predictions for {len(df_output)} samples")
        print(f"  Dataset: {dataset}")
        print(f"  Properties: {', '.join(sorted(all_properties))}")
        print(f"  Saved to: {output_path}")

