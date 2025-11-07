"""DeepSP Ridge model implementation."""

from pathlib import Path
import pickle
import pandas as pd
from sklearn.linear_model import Ridge

from abdev_core import BaseModel, PROPERTY_LIST
from .feature_generator import DeepSPFeatureGenerator


class DeepSPRidgeModel(BaseModel):
    """Ridge regression model on DeepSP spatial features.
    
    This model computes DeepSP spatial properties on-the-fly from VH/VL sequences,
    then trains separate Ridge regression models for each property.
    
    DeepSP generates 30 spatial features per antibody:
    - SAP_pos: Spatial Aggregation Propensity (positive charges) for 10 regions
    - SCM_neg: Spatial Charge Map (negative charges) for 10 regions  
    - SCM_pos: Spatial Charge Map (positive charges) for 10 regions
    
    Regions: CDRH1, CDRH2, CDRH3, CDRL1, CDRL2, CDRL3, CDR, Hv, Lv, Fv
    """
    
    ALPHA = 1.0  # Ridge regression regularization parameter
    
    def __init__(self) -> None:
        """Initialize model (lazy load feature generator on first use)."""
        self.feature_generator = None
    
    def _initialize_feature_generator(self) -> None:
        """Lazy initialize the DeepSP feature generator."""
        if self.feature_generator is not None:
            return
        
        # Model weights are in model_weights/ directory relative to this file
        model_weights_dir = Path(__file__).parent.parent.parent / "model_weights"
        self.feature_generator = DeepSPFeatureGenerator(model_weights_dir)
    
    def _generate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate DeepSP features for all sequences in dataframe.
        
        Args:
            df: DataFrame with antibody_name, vh_protein_sequence, vl_protein_sequence
            
        Returns:
            DataFrame with antibody_name and 30 DeepSP feature columns
        """
        self._initialize_feature_generator()
        
        names = df["antibody_name"].tolist()
        vh_seqs = df["vh_protein_sequence"].tolist()
        vl_seqs = df["vl_protein_sequence"].tolist()
        
        features = self.feature_generator.generate_features(names, vh_seqs, vl_seqs)
        return features
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train Ridge regression models on DeepSP features for each property.
        
        Args:
            df: Training dataframe with VH/VL sequences and property labels
            run_dir: Directory to save trained models
            seed: Random seed for reproducibility
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate DeepSP features for all training samples
        features = self._generate_features(df)
        
        # Merge features with training data
        df_merged = df.merge(features, on="antibody_name", how="left")
        
        # Get feature column names
        feature_cols = DeepSPFeatureGenerator.FEATURE_NAMES
        
        # Train Ridge regression models for each property
        models = {}
        
        for property_name in PROPERTY_LIST:
            # Skip if property not in training data
            if property_name not in df.columns:
                continue
            
            # Filter to samples with non-null values
            not_na_mask = df_merged[property_name].notna()
            df_property = df_merged[not_na_mask]
            
            if len(df_property) == 0:
                continue
            
            # Get features and labels
            X = df_property[feature_cols].values
            y = df_property[property_name].values
            
            # Train Ridge regression with seeded random state
            model = Ridge(alpha=self.ALPHA, random_state=seed)
            model.fit(X, y)
            models[property_name] = model
        
        # Save models
        models_path = run_dir / "models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(models, f)
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions for all samples using trained models.
        
        Args:
            df: Input dataframe with VH/VL sequences
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
        
        # Generate DeepSP features for input data
        features = self._generate_features(df)
        
        # Merge with input data
        df_merged = df.merge(features, on="antibody_name", how="left")
        
        # Get feature column names
        feature_cols = DeepSPFeatureGenerator.FEATURE_NAMES
        
        # Generate predictions for each property
        df_output = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
        
        for property_name, model in models.items():
            X = df_merged[feature_cols].values
            predictions = model.predict(X)
            df_output[property_name] = predictions
        
        return df_output

