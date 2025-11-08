"""MOE molecular descriptors baseline model implementation."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import warnings

from abdev_core import BaseModel, PROPERTY_LIST, load_features

warnings.filterwarnings('ignore')


# Best model configurations from training notebook
BEST_CONFIGS = {
    'HIC': {
        'model_type': 'Ridge',
        'features': [
            'patch_cdr_hyd', 'ens_charge', 'cdr_len', 'patch_pos', 'coeff_280',
            'zdipole', 'amphipathicity', 'ASPmax', 'BSA_LC_HC', 'hyd_idx', 'hyd_strength_cdr'
        ],
        'alpha': 79.06043210907701
    },
    'PR_CHO': {  # Mapped from CHO
        'model_type': 'Ridge',
        'features': [
            'patch_cdr_pos', 'patch_cdr_neg', 'asa_hyd', 'coeff_280', 'sed_const',
            'zeta', 'zquadrupole', 'strand', 'E', 'hyd_idx'
        ],
        'alpha': 59.636233165946365
    },
    'AC-SINS_pH7.4': {  # Mapped from AC-SINS
        'model_type': 'Ridge',
        'features': [
            'patch_cdr_hyd', 'ens_charge', 'Fv_chml', 'patch_cdr_pos', 'patch_cdr_neg',
            'pI_3D', 'patch_pos', 'patch_neg', 'r_gyr', 'dipole_moment', 'hyd_moment',
            'coeff_280', 'sed_const', 'eccen', 'zquadrupole', 'affinity_VL_VH',
            'amphipathicity', 'DRT', 'HI', 'hyd_idx_cdr', 'hyd_strength', 'hyd_strength_cdr',
            'Packing Score'
        ],
        'alpha': 1.5264179671752334
    },
    'Titer': {
        'model_type': 'Ridge',
        'features': ['r_gyr', 'hyd_moment', 'zquadrupole', 'amphipathicity'],
        'alpha': 59.636233165946365
    },
    'Tm2': {
        'model_type': 'MLP',
        'features': [
            'hyd_moment', 'zquadrupole', 'helicity', 'strand', 'E',
            'amphipathicity', 'hyd_idx_cdr', 'Packing Score'
        ],
        'mlp_params': {
            'hidden_layer_sizes': (64,),
            'activation': 'tanh',
            'learning_rate_init': 0.007851328233611145,
            'alpha': 0.056870032781999154,
            'batch_size': 16,
            'max_iter': 500,
            'early_stopping': True,
            'random_state': 42
        }
    }
}


class MoeBaselineModel(BaseModel):
    """MOE molecular descriptors baseline using optimized regression models.
    
    This model trains separate regression models for each biophysical property
    using MOE (Molecular Operating Environment) molecular descriptors. Each
    property uses its own optimized model type and feature set as determined
    through extensive cross-validation experiments.
    
    Model configurations:
    - HIC: Ridge regression (11 features, alpha=79.1)
    - PR_CHO: Ridge regression (10 features, alpha=59.6)
    - AC-SINS_pH7.4: Ridge regression (23 features, alpha=1.5)
    - Titer: Ridge regression (4 features, alpha=59.6)
    - Tm2: MLP neural network (8 features, 1 hidden layer)
    
    Features are loaded from the centralized feature store via abdev_core.
    """
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train optimized regression models for each property on ALL provided data.
        
        Args:
            df: Training dataframe with property labels
            run_dir: Directory to save trained models
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Load MOE features from centralized feature store
        moe_features = load_features("MOE_properties")
        
        # Merge features with training data
        df_merged = df.merge(moe_features.reset_index(), on="antibody_name", how="left")
        
        # Train separate model for each property
        models = {}
        scalers = {}
        
        for property_name in PROPERTY_LIST:
            if property_name not in df.columns:
                continue
            
            if property_name not in BEST_CONFIGS:
                continue
            
            config = BEST_CONFIGS[property_name]
            features = config['features']
            
            # Filter to samples with non-null values for this property
            not_na_mask = df_merged[property_name].notna()
            df_property = df_merged[not_na_mask]
            
            if len(df_property) == 0:
                continue
            
            # Check if all required features are available
            missing_feats = [f for f in features if f not in df_property.columns]
            if missing_feats:
                continue
            
            # Prepare data
            X = df_property[features].values
            y = df_property[property_name].values
            
            # Train model based on type
            if config['model_type'] == 'Ridge':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = Ridge(alpha=config['alpha'], random_state=seed)
                model.fit(X_scaled, y)
                
                scalers[property_name] = scaler
                models[property_name] = model
                
            elif config['model_type'] == 'MLP':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                model = MLPRegressor(**config['mlp_params'])
                model.fit(X_scaled, y)
                
                scalers[property_name] = scaler
                models[property_name] = model
        
        # Save models, scalers, and configs
        artifacts = {
            'models': models,
            'scalers': scalers,
            'configs': BEST_CONFIGS
        }
        
        artifacts_path = run_dir / "model_artifacts.pkl"
        with open(artifacts_path, "wb") as f:
            pickle.dump(artifacts, f)
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions for ALL provided samples using trained models.
        
        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing trained models
            
        Returns:
            DataFrame with predictions for each property
        """
        # Load trained models and scalers
        artifacts_path = run_dir / "model_artifacts.pkl"
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Model artifacts not found: {artifacts_path}")
        
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
        
        models = artifacts['models']
        scalers = artifacts['scalers']
        configs = artifacts['configs']
        
        # Load MOE features from centralized feature store
        moe_features = load_features("MOE_properties")
        df_merged = df.merge(moe_features.reset_index(), on="antibody_name", how="left")
        
        # Create output dataframe with required columns
        # Include antibody_id if present in input data
        output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        if "antibody_id" in df.columns:
            output_cols.insert(0, "antibody_id")
        df_output = df[output_cols].copy()
        
        # Generate predictions for each trained property
        for property_name, model in models.items():
            config = configs[property_name]
            features = config['features']
            
            # Check if features are available
            missing_feats = [f for f in features if f not in df_merged.columns]
            if missing_feats:
                continue
            
            X = df_merged[features].values
            
            # Apply scaling and predict
            if property_name in scalers:
                X_scaled = scalers[property_name].transform(X)
                predictions = model.predict(X_scaled)
            else:
                predictions = model.predict(X)
            
            df_output[property_name] = predictions
        
        return df_output
