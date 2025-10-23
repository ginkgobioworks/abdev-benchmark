"""TAP Linear model implementation."""

from pathlib import Path
import pickle
import pandas as pd
from sklearn.linear_model import Ridge

from abdev_core import BaseModel, PROPERTY_LIST, load_features


# TAP feature names used for modeling
FEATURE_NAMES = ["SFvCSP", "PSH", "PPC", "PNC", "CDR Length"]


class TapLinearModel(BaseModel):
    """Ridge regression model on TAP features.
    
    This model trains separate Ridge regression models for each property
    using TAP (Therapeutic Antibody Profiler) features.
    
    Features are loaded from the centralized feature store via abdev_core.
    """
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train Ridge regression models for each property on ALL provided data.
        
        Args:
            df: Training dataframe with labels
            run_dir: Directory to save trained models
            seed: Random seed (currently not used, sklearn Ridge doesn't need it)
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load TAP features from centralized feature store (all datasets)
        tap_features = load_features("TAP")
        
        # Merge features with data (reset index to join on antibody_name)
        df_merged = df.merge(tap_features.reset_index(), on="antibody_name", how="left")
        
        # Check for missing features
        if df_merged[FEATURE_NAMES].isna().any().any():
            print("Warning: Some samples are missing TAP features")
        
        # Train models for each property on ALL data
        models = {}
        
        for property_name in PROPERTY_LIST:
            # Filter to samples with non-null values for this property
            not_na_mask = df_merged[property_name].notna()
            df_property = df_merged[not_na_mask]
            
            if len(df_property) == 0:
                print(f"Warning: No training data for {property_name}, skipping")
                continue
            
            # Train model on all available data for this property
            X = df_property[FEATURE_NAMES]
            y = df_property[property_name]
            
            model = Ridge()
            model.fit(X, y)
            models[property_name] = model
            
            print(f"  Trained model for {property_name} on {len(df_property)} samples")
        
        # Save models
        models_path = run_dir / "models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(models, f)
        
        print(f"Saved {len(models)} models to {models_path}")
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions for ALL provided samples using trained models.
        
        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing trained models
            
        Returns:
            DataFrame with predictions for each property
        """
        # Load trained models
        models_path = run_dir / "models.pkl"
        if not models_path.exists():
            raise FileNotFoundError(f"Models not found: {models_path}")
        
        with open(models_path, "rb") as f:
            models = pickle.load(f)
        
        # Load TAP features from centralized feature store (all datasets)
        tap_features = load_features("TAP")
        df_merged = df.merge(tap_features.reset_index(), on="antibody_name", how="left")
        
        # Generate predictions for each property
        for property_name, model in models.items():
            X = df_merged[FEATURE_NAMES]
            predictions = model.predict(X)
            df_merged[property_name] = predictions
        
        # Select output columns
        output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        output_cols.extend([prop for prop in PROPERTY_LIST if prop in models])
        
        df_output = df_merged[output_cols]
        
        print(f"Generated predictions for {len(df_output)} samples")
        print(f"  Properties: {', '.join(models.keys())}")
        
        return df_output

