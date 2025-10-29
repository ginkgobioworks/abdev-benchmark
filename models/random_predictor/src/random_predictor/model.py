"""Random Predictor model implementation."""

from pathlib import Path
import json
import pandas as pd
import numpy as np

from abdev_core import BaseModel, PROPERTY_LIST


class RandomPredictorModel(BaseModel):
    """Random Predictor baseline that generates random predictions.
    
    This is a simple baseline for benchmarking and testing that generates
    random predictions uniformly distributed within reasonable ranges.
    
    Useful for:
    - Establishing a performance floor
    - Testing evaluation pipelines
    - Sanity checking that other models perform better than random
    """
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train (no-op) and save seed for reproducible random predictions.
        
        Args:
            df: Training dataframe
            run_dir: Directory to save configuration
            seed: Random seed for reproducibility
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate reasonable ranges from training data for each property
        property_ranges = {}
        for prop in PROPERTY_LIST:
            if prop in df.columns:
                values = df[prop].dropna()
                if len(values) > 0:
                    property_ranges[prop] = {
                        "min": float(values.min()),
                        "max": float(values.max()),
                        "mean": float(values.mean()),
                        "std": float(values.std())
                    }
        
        # Save configuration
        config = {
            "model_type": "random_predictor",
            "seed": seed,
            "property_ranges": property_ranges,
            "note": "Generates random predictions uniformly within observed ranges"
        }
        
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print(f"Saved configuration to {config_path}")
        print(f"Property ranges computed from {len(df)} training samples:")
        for prop, ranges in property_ranges.items():
            print(f"  {prop}: [{ranges['min']:.3f}, {ranges['max']:.3f}]")
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate random predictions using saved configuration.
        
        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing configuration
            
        Returns:
            DataFrame with random predictions for each property
        """
        # Load configuration
        config_path = run_dir / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"Configuration not found: {config_path}. "
                "Run train() first to generate configuration."
            )
        
        with open(config_path, "r") as f:
            config = json.load(f)
        
        seed = config["seed"]
        property_ranges = config["property_ranges"]
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create output dataframe
        df_output = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
        
        # Generate random predictions for each property
        n_samples = len(df)
        for prop, ranges in property_ranges.items():
            # Generate uniform random values within observed range
            df_output[prop] = np.random.uniform(
                low=ranges["min"],
                high=ranges["max"],
                size=n_samples
            )
        
        print(f"Generated random predictions for {len(df_output)} samples")
        print(f"  Seed: {seed}")
        print(f"  Properties: {', '.join(property_ranges.keys())}")
        
        return df_output

