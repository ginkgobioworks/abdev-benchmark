"""MOE molecular descriptors with Stabl feature selection."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint
import lightgbm as lgb
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
import warnings

from abdev_core import BaseModel, PROPERTY_LIST, load_features

warnings.filterwarnings('ignore')

PER_TARGET_MODEL_CONFIG = {
    "HIC": {
        "model": RidgeCV(alphas=np.logspace(-3, 3, 10)),
        "preprocess": Pipeline([("var", VarianceThreshold()), ("scaler", StandardScaler())]),
        "params": {}
    },
    "PR_CHO": {
        "model": XGBRegressor(objective='reg:squarederror', eval_metric='rmse', use_label_encoder=False),
        "preprocess": Pipeline([("var", VarianceThreshold()), ("scaler", StandardScaler())]),
        "params": {
            'n_estimators': randint(50, 200),
            'max_depth': randint(3, 10),
            'learning_rate': uniform(0.01, 0.29),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_weight': randint(1, 10)
        }
    },
    "AC-SINS_pH7.4": {
        "model": lgb.LGBMRegressor(),
        "preprocess": Pipeline([("var", VarianceThreshold()), ("scaler", StandardScaler())]),
        "params": {
            'num_leaves': randint(20, 150),
            'learning_rate': uniform(0.01, 0.29),
            'n_estimators': randint(50, 200),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_samples': randint(5, 50)
        }
    },
    "Tm2": {
        "model": MLPRegressor(max_iter=500, early_stopping=True),
        "preprocess": Pipeline([("var", VarianceThreshold()), ("scaler", StandardScaler())]),
        "params": {
                    'hidden_layer_sizes': [(16,), (32,), (64,), (16, 16), (32, 16), (64, 32)],
                    'learning_rate_init': uniform(0.0001, 0.01),
                    'alpha': uniform(0.0001, 0.1),
                    'batch_size': [16, 32, 64],
                    'activation': ['relu', 'tanh']
                }
    },
    "Titer":{
        "model": lgb.LGBMRegressor(),
        "preprocess": Pipeline([("var", VarianceThreshold()), ("scaler", StandardScaler())]),
        "params": {
            'num_leaves': randint(20, 150),
            'learning_rate': uniform(0.01, 0.29),
            'n_estimators': randint(50, 200),
            'subsample': uniform(0.6, 0.4),
            'colsample_bytree': uniform(0.6, 0.4),
            'min_child_samples': randint(5, 50)
        }
    }
}

class MoeStablBaselineModel(BaseModel):
    """ MOE molecular descriptors with Stabl feature selection."""
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train MOE Stabl Baseline model."""
        print("Training MOE Stabl Baseline model")
        np.random.seed(seed)
        run_dir.mkdir(parents=True, exist_ok=True)

        models = {}
        preprocessors = {}
        selected_features = {}
        fdp_infos = {}
        merged_freqs_all = {}

        # does_stabl_selection_exist = "stabl_selection.pkl" in run_dir.iterdir()
        stabl_selection_file = Path(__file__).parent.parent.parent / "stabl_feature_selection_results.pkl"
        does_stabl_selection_exist = stabl_selection_file.exists()
        if not does_stabl_selection_exist:
            # raise error
            print("STABL selection file not found")
            raise FileNotFoundError("STABL selection file not found")
        else:
            with open(stabl_selection_file, "rb") as f:
                stabl_data = pickle.load(f)

        fold_col = 'hierarchical_cluster_IgG_isotype_stratified_fold'
        if fold_col in df.columns:
            all_folds = set(range(5))  # Folds 0-4
            present_folds = set(df[fold_col].unique())
            missing_folds = all_folds - present_folds
            if len(missing_folds) == 1:
                fold_id = list(missing_folds)[0]
            else:
                fold_id = "all"
        else:
            fold_id = "all"
        
        moe_features = load_features("MOE_properties")
        df_target = df[PROPERTY_LIST]
        df_merged = df.merge(moe_features.reset_index(), on="antibody_name", how="left")
        ab_cols = [col for col in df_merged.columns if 'antibody_id' in col] + ["mseq"]
        df_merged = df_merged.drop(columns=ab_cols, errors="ignore")
        moe_features_intersect_merged = [col for col in moe_features.columns if col in df_merged.columns]
        df_merged = df_merged[moe_features_intersect_merged]

        print("Training data shape after filtering MOE features:", df_merged.shape)
        print()
        for target in PROPERTY_LIST:
            if target not in df.columns:
                print(f"Skipping target {target}, not in dataframe")
                continue
            
            # Filter samples with non-null target values
            not_na_mask = df_target[target].notna()
            y_train = df_target.loc[not_na_mask, target].values
            df_non_na = df_merged.loc[not_na_mask].copy()


            if len(df_non_na) < 10:
                print(f"Skipping target {target}, not enough data")
                continue
            
            print(f"Loading Stabl selection from file for target: {target}")
            sel_features = stabl_data[target][fold_id]["selected_features"]
            if target in ["Titer"]:
                # for Titer, keep all features as the optimal threshold is close to 1.0
                sel_features = [f for f in sel_features if f in df_non_na.columns]
            merged_freq = stabl_data[target][fold_id]["merged_freq"]
            fdp_info = stabl_data[target][fold_id]["fdp_info"]

            selected_features[target] = sel_features
            merged_freqs_all[target] = merged_freq
            fdp_infos[target] = fdp_info

            if len(sel_features) == 0:
                print("No features selected in this fold. Skipping model training.")
                continue

            df_tr_sel = df_non_na[sel_features]
            base_model_config = PER_TARGET_MODEL_CONFIG[target]
            base_model = base_model_config["model"]
            param_dist = base_model_config["params"]
            preproc = base_model_config["preprocess"]
            X_tr_sel = preproc.fit_transform(df_tr_sel)
                  
            if param_dist:
                search = RandomizedSearchCV(
                    base_model,
                    param_distributions=param_dist,
                    n_iter=25,
                    scoring='neg_mean_squared_error',
                    cv=5,
                    random_state=seed,
                    n_jobs=-1
                    )
                search.fit(X_tr_sel, y_train)
                best_model = search.best_estimator_
            else:
                best_model = base_model
                best_model.fit(X_tr_sel, y_train)
            models[target] = best_model
            preprocessors[target] = preproc

            print("Summary for target:", target)
           
            print("Number of selected features:", len(sel_features))
            print("Selected features:", [f for f in sel_features if f in df_non_na.columns])
            print("Model: ", type(best_model).__name__)
            print()
            
        artifacts = {
            "selected_features": selected_features,
            "models": models,
            "preprocessors": preprocessors,
            "fdp_infos": fdp_infos,
            "merged_freqs_all": merged_freqs_all
        }

        artifacts_path = run_dir / "model_artifacts.pkl"
        with open(artifacts_path, "wb") as f:
            pickle.dump(artifacts, f)

    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Predict using trained MOE Stabl Baseline model."""
        print("Predicting with MOE Stabl Baseline model")
        with open(run_dir / "model_artifacts.pkl", "rb") as f:
            artifacts = pickle.load(f)

        selected_features = artifacts["selected_features"]
        models = artifacts["models"]
        preprocessors = artifacts["preprocessors"]

        moe_features = load_features("MOE_properties")
    
        df_merged = df.merge(moe_features.reset_index(), on="antibody_name", how="left")

        output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        if "antibody_id" in df.columns:
            output_cols.insert(0, "antibody_id")
        df_output = df[output_cols].copy()

        for target, model in models.items():
            preproc = preprocessors[target]
            
            sel_features = selected_features[target]

            X_proc = preproc.transform(df_merged[sel_features])

            preds = model.predict(X_proc)
            df_output[target] = preds
        
        return df_output