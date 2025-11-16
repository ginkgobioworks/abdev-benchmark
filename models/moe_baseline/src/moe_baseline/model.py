"""MOE molecular descriptors baseline model implementation with nested CV."""

from pathlib import Path
import pickle
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import lightgbm as lgb
import warnings

from abdev_core import BaseModel, PROPERTY_LIST, load_features

warnings.filterwarnings('ignore')


# Best model configurations from nested CV (excluding Stabl)
# Features are loaded per-fold from JSON files to prevent data leakage
BEST_CONFIGS = {
    'HIC': {
        'model_type': 'Ridge',
        'feature_set': 'Consensus',  # LASSO ∪ XGBoost
        'cv_target': 'HIC'  # Target name in CV results
    },
    'PR_CHO': {  # Mapped from CHO in dataset
        'model_type': 'LightGBM',
        'feature_set': 'Consensus',
        'cv_target': 'CHO'
    },
    'AC-SINS_pH7.4': {  # Mapped from AC-SINS in dataset
        'model_type': 'Ridge',
        'feature_set': 'LASSO',
        'cv_target': 'AC-SINS'
    },
    'Titer': {
        'model_type': 'MLP',
        'feature_set': 'All_MOE',
        'cv_target': 'Titer'
    },
    'Tm2': {
        'model_type': 'LightGBM',
        'feature_set': 'Consensus',
        'cv_target': 'Tm2'
    }
}


class MoeBaselineModel(BaseModel):
    """MOE baseline with nested CV to prevent data leakage.
    
    Uses pre-computed per-fold features from nested cross-validation experiments.
    Each fold's features were selected using only that fold's training data,
    preventing the optimistic bias from naive feature selection.
    
    Best models (excluding Stabl):
    - HIC: Ridge + Consensus features (Spearman: 0.655 ± 0.078)
    - PR_CHO: LightGBM + Consensus features (Spearman: 0.383 ± 0.133)
    - AC-SINS_pH7.4: Ridge + LASSO features (Spearman: 0.460 ± 0.065)
    - Titer: MLP + All MOE features (Spearman: 0.202 ± 0.109)
    - Tm2: LightGBM + Consensus features (Spearman: 0.135 ± 0.089)
    
    Features are loaded from the centralized feature store via abdev_core.
    """
    
    def _get_fold_features(self, cv_target: str, feature_set: str, fold_id: int) -> list:
        """Load pre-computed features for a specific fold and target.
        
        Args:
            cv_target: Target name in CV results (HIC, CHO, AC-SINS, Titer, Tm2)
            feature_set: Feature set name (Consensus, LASSO, All_MOE, etc.)
            fold_id: Fold number (0-4)
            
        Returns:
            List of feature names for this fold
        """
        # Load fold features JSON
        json_path = Path(__file__).parent.parent.parent / f"{cv_target}_fold_features_updated_feature_selection.json"
        
        if not json_path.exists():
            raise FileNotFoundError(f"Fold features not found: {json_path}")
        
        with open(json_path, 'r') as f:
            fold_features = json.load(f)
        
        fold_key = f"fold_{fold_id}"
        if fold_key not in fold_features:
            raise ValueError(f"Fold {fold_id} not found in {cv_target} features")
        
        if feature_set not in fold_features[fold_key]:
            raise ValueError(f"Feature set '{feature_set}' not found for {cv_target} fold {fold_id}")
        
        return fold_features[fold_key][feature_set]
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train models using per-fold features from nested CV.
        
        The orchestrator calls this once per fold. We determine which fold this is
        and load the corresponding pre-computed features.
        
        Args:
            df: Training dataframe with property labels (80% of data for this fold)
            run_dir: Directory to save trained models
            seed: Random seed for reproducibility
        """
        np.random.seed(seed)
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine which fold this is by checking the fold column in the data
        # The orchestrator provides the training data for ONE fold at a time
        fold_col = 'hierarchical_cluster_IgG_isotype_stratified_fold'
        if fold_col in df.columns:
            # This is cross-validation - determine fold ID from what's MISSING
            # The test fold is the one NOT in this training data
            all_folds = set(range(5))  # Folds 0-4
            present_folds = set(df[fold_col].unique())
            missing_folds = all_folds - present_folds
            if len(missing_folds) == 1:
                fold_id = list(missing_folds)[0]
            else:
                # Fallback: use fold 0 if we can't determine (e.g., heldout test)
                fold_id = 0
        else:
            # No fold column (e.g., heldout test set) - use fold 0 features
            fold_id = 0
        
        # Load MOE features from centralized feature store
        moe_features = load_features("MOE_properties")
        
        # Merge features with training data
        df_merged = df.merge(moe_features.reset_index(), on="antibody_name", how="left")
        
        # Train separate model for each property
        models = {}
        scalers = {}
        selected_features = {}
        
        for property_name in PROPERTY_LIST:
            if property_name not in df.columns:
                continue
            
            if property_name not in BEST_CONFIGS:
                continue
            
            config = BEST_CONFIGS[property_name]
            cv_target = config['cv_target']
            feature_set = config['feature_set']
            
            # Load per-fold features
            features = self._get_fold_features(cv_target, feature_set, fold_id)
            selected_features[property_name] = features
            
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
                
                # Use RidgeCV to select best alpha
                model = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=5)
                model.fit(X_scaled, y)
                
                scalers[property_name] = scaler
                models[property_name] = model
                
            elif config['model_type'] == 'LightGBM':
                # LightGBM with hyperparameter search
                param_dist = {
                    'num_leaves': randint(20, 150),
                    'learning_rate': uniform(0.01, 0.29),
                    'n_estimators': randint(50, 200),
                    'subsample': uniform(0.6, 0.4),
                    'colsample_bytree': uniform(0.6, 0.4),
                    'min_child_samples': randint(5, 50)
                }
                
                base_model = lgb.LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1)
                search = RandomizedSearchCV(
                    base_model, param_dist, n_iter=25, cv=5,
                    scoring='neg_mean_squared_error', random_state=seed, n_jobs=-1
                )
                search.fit(X, y)
                models[property_name] = search.best_estimator_
                
            elif config['model_type'] == 'MLP':
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # MLP with hyperparameter search
                param_dist = {
                    'hidden_layer_sizes': [(32,), (64,), (128,), (32, 16), (64, 32), (128, 64)],
                    'learning_rate_init': uniform(0.0001, 0.01),
                    'alpha': uniform(0.0001, 0.1),
                    'batch_size': [16, 32, 64],
                    'activation': ['relu', 'tanh']
                }
                
                base_model = MLPRegressor(max_iter=500, early_stopping=True, random_state=seed)
                search = RandomizedSearchCV(
                    base_model, param_dist, n_iter=25, cv=5,
                    scoring='neg_mean_squared_error', random_state=seed, n_jobs=-1
                )
                search.fit(X_scaled, y)
                
                scalers[property_name] = scaler
                models[property_name] = search.best_estimator_
        
        # Save models, scalers, selected features (for this fold), and configs
        artifacts = {
            'models': models,
            'scalers': scalers,
            'selected_features': selected_features,
            'fold_id': fold_id,
            'configs': BEST_CONFIGS
        }
        
        artifacts_path = run_dir / "model_artifacts.pkl"
        with open(artifacts_path, "wb") as f:
            pickle.dump(artifacts, f)
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions using trained models and their fold-specific features.
        
        Args:
            df: Input dataframe with sequences
            run_dir: Directory containing trained models
            
        Returns:
            DataFrame with predictions for each property
        """
        # Load trained models, scalers, and selected features
        artifacts_path = run_dir / "model_artifacts.pkl"
        if not artifacts_path.exists():
            raise FileNotFoundError(f"Model artifacts not found: {artifacts_path}")
        
        with open(artifacts_path, "rb") as f:
            artifacts = pickle.load(f)
        
        models = artifacts['models']
        scalers = artifacts['scalers']
        selected_features = artifacts['selected_features']
        
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
            features = selected_features[property_name]
            
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
