"""ESM2 + TAP XGBoost model implementation with PCA dimensionality reduction."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import xgboost as xgb
import torch
from transformers import AutoTokenizer, AutoModel

from abdev_core import BaseModel, PROPERTY_LIST, load_features


# TAP feature names
TAP_FEATURE_NAMES = ["SFvCSP", "PSH", "PPC", "PNC", "CDR Length"]


class ESM2TapXGBModel(BaseModel):
    """XGBoost model on PCA-reduced ESM2 embeddings + TAP + subtype features.


    Key anti-overfitting strategies :
    - PCA dimensionality reduction (640 → 50)
    - Moderate boosting rounds (50)
    - Moderate tree depth (max_depth=3, balanced complexity)
    - Moderate learning rate (0.075, balanced learning speed)
    - Subsample training data (75%, moderate randomness)
    - Subsample features (75%, moderate randomness)
    - Strong L2 regularization (reg_lambda=30)
    - Early stopping with validation split (prevents overfitting)

    Feature breakdown:
    - ESM2 (PCA-reduced): 50 dimensions
    - TAP features: 5 dimensions
    - hc_subtype (one-hot): 3 dimensions (IgG1, IgG2, IgG4)
    - lc_subtype (one-hot): 2 dimensions (Kappa, Lambda)

    """

    # ESM2 configuration
    ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    PCA_COMPONENTS = 50  # Reduce 640D → 50D (optimal balance, ~94% variance)

    # XGBoost hyperparameters (Version 3 - Balanced anti-overfitting)
    XGB_N_ESTIMATORS = 50           # Moderate boosting rounds (sweet spot between 30 and 100)
    XGB_MAX_DEPTH = 3               # Moderate tree depth (balanced between 2 and 4)
    XGB_LEARNING_RATE = 0.075       # Moderate learning rate (balanced between 0.05 and 0.1)
    XGB_MIN_CHILD_WEIGHT = 7        # Moderate sample requirement (balanced between 5 and 10)
    XGB_SUBSAMPLE = 0.75            # Subsample 75% of training data (balanced)
    XGB_COLSAMPLE_BYTREE = 0.75     # Subsample 75% of features (balanced)
    XGB_REG_LAMBDA = 30             # Strong L2 regularization (balanced between 10 and 50)
    XGB_EARLY_STOPPING_ROUNDS = 10  # Stop if no validation improvement for 10 rounds

    def __init__(self) -> None:
        """Initialize model (lazy load transformers on first use)."""
        self.tokenizer = None
        self.esm2_model = None
        self.device = None

    def _initialize_esm2(self) -> None:
        """Lazy initialize the ESM2 transformer model and tokenizer."""
        if self.esm2_model is not None:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(f"Loading ESM2 model on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.ESM2_MODEL_NAME)
        self.esm2_model = AutoModel.from_pretrained(self.ESM2_MODEL_NAME).to(self.device)
        self.esm2_model.eval()

    def _embed_sequence(self, sequence: str) -> np.ndarray:
        """Generate mean-pooled ESM2 embedding for a single sequence.

        Args:
            sequence: Protein sequence (amino acid string)

        Returns:
            1D array of shape (320,) with mean-pooled representation
        """
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.esm2_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1]

        # Mean pool over sequence length, excluding padding
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        mean_pooled = sum_embeddings / sum_mask

        embedding = mean_pooled.detach().cpu().numpy().squeeze(0)
        return embedding

    def _generate_esm2_embeddings(
        self, vh_sequences: list[str], vl_sequences: list[str]
    ) -> np.ndarray:
        """Generate concatenated VH+VL ESM2 embeddings.

        Returns:
            Array of shape (n_sequences, 640) with concatenated embeddings
        """
        self._initialize_esm2()

        embeddings_list = []
        for vh_seq, vl_seq in zip(vh_sequences, vl_sequences):
            vh_embedding = self._embed_sequence(vh_seq)
            vl_embedding = self._embed_sequence(vl_seq)
            combined_embedding = np.concatenate([vh_embedding, vl_embedding])
            embeddings_list.append(combined_embedding)

        embeddings = np.stack(embeddings_list)
        # print(f"Generated ESM2 embeddings: {embeddings.shape}")
        return embeddings

    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train XGBoost models on PCA-reduced ESM2 + TAP features.

        Args:
            df: Training dataframe with VH/VL sequences and property labels
            run_dir: Directory to save trained models
            seed: Random seed for reproducibility
        """
        # Set random seeds
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate ESM2 embeddings (640D)
        # print("Generating ESM2 embeddings...")
        esm2_embeddings = self._generate_esm2_embeddings(
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
        )

        # 2. Apply PCA to reduce dimensionality (640D → 50D)
        # print(f"Applying PCA: {esm2_embeddings.shape[1]}D → {self.PCA_COMPONENTS}D")
        pca = PCA(n_components=self.PCA_COMPONENTS, random_state=seed)
        esm2_reduced = pca.fit_transform(esm2_embeddings)
        # print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        # 3. Load TAP features (5D)
        # print("Loading TAP features...")
        tap_features = load_features("TAP")
        df_merged = df.merge(tap_features.reset_index(), on="antibody_name", how="left")
        tap_array = df_merged[TAP_FEATURE_NAMES].values

        # 4. One-hot encode antibody subtypes
        # print("Encoding antibody subtypes...")
        hc_dummies = pd.get_dummies(df["hc_subtype"], prefix="hc").astype(int)
        lc_dummies = pd.get_dummies(df["lc_subtype"], prefix="lc").astype(int)
        subtype_array = np.concatenate([hc_dummies.values, lc_dummies.values], axis=1)
        subtype_columns = list(hc_dummies.columns) + list(lc_dummies.columns)
        # print(f"Subtype features: {subtype_array.shape[1]} ({', '.join(subtype_columns)})")

        # 5. Combine PCA-reduced ESM2 + TAP + Subtypes (60D total)
        X_combined = np.concatenate([esm2_reduced, tap_array, subtype_array], axis=1)
        # print(f"Combined features: {X_combined.shape} (ESM2-PCA: {self.PCA_COMPONENTS}, TAP: {len(TAP_FEATURE_NAMES)}, Subtypes: {len(subtype_columns)})")

        # 6. Train XGBoost models for each property
        models = {}

        for property_name in PROPERTY_LIST:
            if property_name not in df.columns:
                continue

            # Filter to samples with non-null values
            not_na_mask = df[property_name].notna()
            if not_na_mask.sum() == 0:
                # print(f"  Skipping {property_name}: no training data")
                continue

            X = X_combined[not_na_mask]
            y = df[property_name][not_na_mask].values

            # Split into train/validation for early stopping
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )

            # Train XGBoost with anti-overfitting hyperparameters and early stopping
            xgb_model = xgb.XGBRegressor(
                n_estimators=self.XGB_N_ESTIMATORS,
                max_depth=self.XGB_MAX_DEPTH,
                learning_rate=self.XGB_LEARNING_RATE,
                min_child_weight=self.XGB_MIN_CHILD_WEIGHT,
                subsample=self.XGB_SUBSAMPLE,
                colsample_bytree=self.XGB_COLSAMPLE_BYTREE,
                reg_lambda=self.XGB_REG_LAMBDA,
                random_state=seed,
                n_jobs=-1,  # Use all CPU cores
                tree_method="hist",  # Faster histogram-based method
                early_stopping_rounds=self.XGB_EARLY_STOPPING_ROUNDS,
            )
            xgb_model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
            models[property_name] = xgb_model

            # Get actual number of trees used (may be less due to early stopping)
            # best_iteration = xgb_model.best_iteration if hasattr(xgb_model, 'best_iteration') else self.XGB_N_ESTIMATORS

            # print(f"  Trained XGBoost for {property_name} on {len(y_train)} samples (val: {len(y_val)})")
            # print(f"    Hyperparams: n_estimators={self.XGB_N_ESTIMATORS} (used: {best_iteration}), max_depth={self.XGB_MAX_DEPTH}, "
            #       f"learning_rate={self.XGB_LEARNING_RATE}, reg_lambda={self.XGB_REG_LAMBDA}")

        # 7. Save models, PCA transformer, subtype columns, and embeddings
        models_path = run_dir / "models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(models, f)

        pca_path = run_dir / "pca.pkl"
        with open(pca_path, "wb") as f:
            pickle.dump(pca, f)

        subtype_columns_path = run_dir / "subtype_columns.pkl"
        with open(subtype_columns_path, "wb") as f:
            pickle.dump(subtype_columns, f)

        esm2_embeddings_path = run_dir / "esm2_embeddings.npy"
        np.save(esm2_embeddings_path, esm2_embeddings)

        # print(f"\nSaved {len(models)} models to {models_path}")
        # print(f"Saved PCA transformer to {pca_path}")
        # print(f"Saved subtype columns to {subtype_columns_path}")
        # print(f"Saved ESM2 embeddings to {esm2_embeddings_path}")

    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions using trained XGBoost models.

        Args:
            df: Input dataframe with VH/VL sequences
            run_dir: Directory containing trained models and PCA transformer

        Returns:
            DataFrame with predictions for each property
        """
        # Load trained models, PCA, and subtype columns
        models_path = run_dir / "models.pkl"
        pca_path = run_dir / "pca.pkl"
        subtype_columns_path = run_dir / "subtype_columns.pkl"

        if not models_path.exists():
            raise FileNotFoundError(f"Models not found: {models_path}")
        if not pca_path.exists():
            raise FileNotFoundError(f"PCA transformer not found: {pca_path}")
        if not subtype_columns_path.exists():
            raise FileNotFoundError(f"Subtype columns not found: {subtype_columns_path}")

        with open(models_path, "rb") as f:
            models = pickle.load(f)
        with open(pca_path, "rb") as f:
            pca = pickle.load(f)
        with open(subtype_columns_path, "rb") as f:
            subtype_columns = pickle.load(f)

        # 1. Generate ESM2 embeddings
        # print(f"Generating ESM2 embeddings for {len(df)} samples...")
        esm2_embeddings = self._generate_esm2_embeddings(
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
        )

        # 2. Apply PCA transformation
        # print(f"Applying PCA transformation: {esm2_embeddings.shape[1]}D → {self.PCA_COMPONENTS}D")
        esm2_reduced = pca.transform(esm2_embeddings)

        # 3. Load TAP features
        tap_features = load_features("TAP")
        df_merged = df.merge(tap_features.reset_index(), on="antibody_name", how="left")
        tap_array = df_merged[TAP_FEATURE_NAMES].values

        # 4. Encode antibody subtypes (same as training)
        hc_dummies = pd.get_dummies(df["hc_subtype"], prefix="hc").astype(int)
        lc_dummies = pd.get_dummies(df["lc_subtype"], prefix="lc").astype(int)

        # Ensure same columns as training (handle missing categories)
        all_dummies = pd.concat([hc_dummies, lc_dummies], axis=1)
        for col in subtype_columns:
            if col not in all_dummies.columns:
                all_dummies[col] = 0
        subtype_array = all_dummies[subtype_columns].values

        # 5. Combine features
        X_combined = np.concatenate([esm2_reduced, tap_array, subtype_array], axis=1)

        # 6. Generate predictions
        df_output = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()

        for property_name, model in models.items():
            predictions = model.predict(X_combined)
            df_output[property_name] = predictions

        # print(f"Generated predictions for {len(df_output)} samples")
        # print(f"  Properties: {', '.join(models.keys())}")

        return df_output
