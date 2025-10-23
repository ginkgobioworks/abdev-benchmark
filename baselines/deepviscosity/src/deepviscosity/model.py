"""DeepViscosity model implementation."""

from pathlib import Path
import json
import pandas as pd

from abdev_core import BaseModel, load_features


# Feature to property mappings
FEATURE_MAPPINGS = {
    "Viscosity": [("HIC", 1)],
}


class DeepViscosityModel(BaseModel):
    """DeepViscosity baseline using pre-computed viscosity predictions.
    
    This is a non-training baseline that directly maps viscosity features
    to predicted properties based on observed correlations.
    
    Features are loaded from the centralized feature store via abdev_core.
    Note: DeepViscosity features are only available for GDPa1 dataset.
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
            "model_type": "deepviscosity",
            "feature_mappings": FEATURE_MAPPINGS,
            "note": "Non-training baseline using pre-computed DeepViscosity features (GDPa1 only)"
        }
        
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved configuration to {config_path}")
        print("Note: This is a non-training baseline using pre-computed features")
    
    def predict(self, df: pd.DataFrame, run_dir: Path, out_dir: Path) -> None:
        """Generate predictions using DeepViscosity features.
        
        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing configuration (not strictly needed)
            out_dir: Directory to write predictions.csv
        """
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if features are available for this dataset
        dataset = "GDPa1"
        try:
            viscosity_features = load_features("DeepViscosity", dataset=dataset)
        except FileNotFoundError:
            print(f"Warning: DeepViscosity features not available for {dataset}")
            print("Generating empty predictions...")
            
            # Return empty predictions
            output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
            df_output = df[output_cols].copy()
            
            output_path = out_dir / "predictions.csv"
            df_output.to_csv(output_path, index=False)
            print(f"  Saved to: {output_path}")
            return
        
        # Merge sequences with features
        df_merged = df.merge(
            viscosity_features[["Viscosity"]].reset_index(),
            on="antibody_name",
            how="left"
        )
        
        # Apply directionality to create predictions
        all_properties = []
        for assay_name, directionality in FEATURE_MAPPINGS["Viscosity"]:
            df_merged[assay_name] = df_merged["Viscosity"] * directionality
            all_properties.append(assay_name)
        
        # Select output columns
        output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        output_cols.extend(all_properties)
        df_output = df_merged[output_cols]
        
        # Write predictions
        output_path = out_dir / "predictions.csv"
        df_output.to_csv(output_path, index=False)
        
        print(f"Generated predictions for {len(df_output)} samples")
        print(f"  Dataset: {dataset}")
        print(f"  Properties: {', '.join(all_properties)}")
        print(f"  Saved to: {output_path}")

