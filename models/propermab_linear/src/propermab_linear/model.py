from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from scipy.stats import spearmanr

from abdev_core import BaseModel, PROPERTY_LIST

# top 5 propermab features
FEATURE_NAMES = ["hyd_patch_area_cdr", "pos_patch_area", "dipole_moment",
                 "aromatic_asa", "exposed_net_charge"]

# all propermab features
FEATURE_CSV_PATH = "feature_store_top5.csv"


class TapLinearModel(BaseModel):

    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """
        Train Ridge models and compute 5-fold CV Spearman correlation.
        Saves:
        models.pkl          – trained full models
        cv_spearman.csv     – average Spearman ρ per property
        """
        run_dir.mkdir(parents=True, exist_ok=True)

        # propermab features
        feature_df = pd.read_csv(FEATURE_CSV_PATH)

        # merge features into training df
        df_merged = df.merge(feature_df, on="antibody_name", how="left")

        cv_results = []
        models = {}

        for property_name in PROPERTY_LIST:

            mask = df_merged[property_name].notna()
            df_prop = df_merged[mask]

            if len(df_prop) == 0:
                print(f"Warning: no data for {property_name}")
                continue

            X = df_prop[FEATURE_NAMES].values
            y = df_prop[property_name].values

            # ----- 5-fold cross-validation -----
            kf = KFold(n_splits=5, shuffle=True, random_state=seed)
            spearman_scores = []

            for train_idx, val_idx in kf.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = Ridge()
                model.fit(X_train, y_train)

                preds = model.predict(X_val)
                rho, _ = spearmanr(y_val, preds, nan_policy='omit')
                spearman_scores.append(rho)

            avg_rho = np.nanmean(spearman_scores)

            cv_results.append({
                "property": property_name,
                "spearman_rho": avg_rho
            })

            print(f"{property_name}: CV Spearman ρ = {avg_rho:.4f}")

            # train final model on all available data
            final_model = Ridge()
            final_model.fit(X, y)
            models[property_name] = final_model

        # save trained models
        models_path = run_dir / "models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(models, f)
        print(f"Saved models to {models_path}")

        # save CV Spearman results
        df_cv = pd.DataFrame(cv_results)
        cv_path = run_dir / "cv_spearman.csv"
        df_cv.to_csv(cv_path, index=False)
        print(f"Saved CV Spearman results to {cv_path}")


    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions for all provided samples using trained models."""
        models_path = run_dir / "models.pkl"
        if not models_path.exists():
            raise FileNotFoundError(f"Models not found: {models_path}")

        with open(models_path, "rb") as f:
            models = pickle.load(f)

        # merge features into prediction df
        feature_df = pd.read_csv(FEATURE_CSV_PATH)
        df_merged = df.merge(feature_df, on="antibody_name", how="left")

        # generate predictions
        for property_name, model in models.items():
            X = df_merged[FEATURE_NAMES].values
            df_merged[property_name] = model.predict(X)

        # output columns
        output_cols = ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        output_cols.extend([prop for prop in PROPERTY_LIST if prop in models])
        df_output = df_merged[output_cols]

        print(f"Generated predictions for {len(df_output)} samples")
        print(f"  Properties: {', '.join(models.keys())}")

        return df_output
