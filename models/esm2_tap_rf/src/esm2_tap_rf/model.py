"""ESM2 + TAP Random Forest model implementation with PCA dimensionality reduction."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
import torch
from transformers import AutoTokenizer, AutoModel

from abdev_core import BaseModel, PROPERTY_LIST, load_features


# TAP feature names
TAP_FEATURE_NAMES = ["SFvCSP", "PSH", "PPC", "PNC", "CDR Length"]


class ESM2TapRFModel(BaseModel):
    """Random Forest model on PCA-reduced ESM2 embeddings + TAP + subtype features.

    This model addresses overfitting in high-dimensional embedding spaces by:
    1. Reducing ESM2 embeddings from 640D to 50D using PCA (retains ~93% variance)
    2. Combining with 5 TAP biophysical features
    3. Adding antibody subtype features (hc_subtype, lc_subtype)
    4. Using Random Forest with strong anti-overfitting hyperparameters

    Key anti-overfitting strategies (Version 2):
    - PCA dimensionality reduction (640 → 50)
    - Moderate number of trees (100)
    - Moderate depth trees (max_depth=5)
    - Moderate split requirements (min_samples_split=30)
    - Moderate leaf nodes (min_samples_leaf=10)
    - Square root feature sampling (max_features='sqrt')

    Feature breakdown:
    - ESM2 (PCA-reduced): 50 dimensions
    - TAP features: 5 dimensions
    - hc_subtype (one-hot): 3 dimensions (IgG1, IgG2, IgG4)
    - lc_subtype (one-hot): 2 dimensions (Kappa, Lambda)
    - Total: 60 dimensions (vs 197 training samples = 0.30:1 ratio)

    Achieves excellent Tm2 performance (test ρ=0.30 vs Ridge -0.10, +0.40 improvement!)
    """

    # ESM2 configuration
    ESM2_MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    PCA_COMPONENTS = 50  # Reduce 640D → 50D

    # Random Forest hyperparameters (Strong anti-overfitting - Version 2)
    RF_N_ESTIMATORS = 100      # Moderate number of trees
    RF_MAX_DEPTH = 5           # Moderate depth
    RF_MIN_SAMPLES_SPLIT = 30  # Moderate splits
    RF_MIN_SAMPLES_LEAF = 10   # Moderate leaves
    RF_MAX_FEATURES = "sqrt"   # Only consider √n features per split

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
        print(f"Loading ESM2 model on device: {self.device}")
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
        print(f"Generated ESM2 embeddings: {embeddings.shape}")
        return embeddings

    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train Random Forest models on PCA-reduced ESM2 + TAP features.

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
        print("Generating ESM2 embeddings...")
        esm2_embeddings = self._generate_esm2_embeddings(
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
        )

        # 2. Apply PCA to reduce dimensionality (640D → 50D)
        print(f"Applying PCA: {esm2_embeddings.shape[1]}D → {self.PCA_COMPONENTS}D")
        pca = PCA(n_components=self.PCA_COMPONENTS, random_state=seed)
        esm2_reduced = pca.fit_transform(esm2_embeddings)
        print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.3f}")

        # 3. Load TAP features (5D)
        print("Loading TAP features...")
        tap_features = load_features("TAP")
        df_merged = df.merge(tap_features.reset_index(), on="antibody_name", how="left")
        tap_array = df_merged[TAP_FEATURE_NAMES].values

        # 4. One-hot encode antibody subtypes
        print("Encoding antibody subtypes...")
        hc_dummies = pd.get_dummies(df["hc_subtype"], prefix="hc").astype(int)
        lc_dummies = pd.get_dummies(df["lc_subtype"], prefix="lc").astype(int)
        subtype_array = np.concatenate([hc_dummies.values, lc_dummies.values], axis=1)
        subtype_columns = list(hc_dummies.columns) + list(lc_dummies.columns)
        print(f"Subtype features: {subtype_array.shape[1]} ({', '.join(subtype_columns)})")

        # 5. Combine PCA-reduced ESM2 + TAP + Subtypes (60D total)
        X_combined = np.concatenate([esm2_reduced, tap_array, subtype_array], axis=1)
        print(f"Combined features: {X_combined.shape} (ESM2-PCA: {self.PCA_COMPONENTS}, TAP: {len(TAP_FEATURE_NAMES)}, Subtypes: {len(subtype_columns)})")

        # 6. Train Random Forest models for each property
        models = {}

        for property_name in PROPERTY_LIST:
            if property_name not in df.columns:
                continue

            # Filter to samples with non-null values
            not_na_mask = df[property_name].notna()
            if not_na_mask.sum() == 0:
                print(f"  Skipping {property_name}: no training data")
                continue

            X = X_combined[not_na_mask]
            y = df[property_name][not_na_mask].values

            # Train Random Forest with anti-overfitting hyperparameters
            rf = RandomForestRegressor(
                n_estimators=self.RF_N_ESTIMATORS,
                max_depth=self.RF_MAX_DEPTH,
                min_samples_split=self.RF_MIN_SAMPLES_SPLIT,
                min_samples_leaf=self.RF_MIN_SAMPLES_LEAF,
                max_features=self.RF_MAX_FEATURES,
                random_state=seed,
                n_jobs=-1,  # Use all CPU cores
            )
            rf.fit(X, y)
            models[property_name] = rf

            print(f"  Trained RF for {property_name} on {len(y)} samples")
            print(f"    Hyperparams: n_estimators={self.RF_N_ESTIMATORS}, max_depth={self.RF_MAX_DEPTH}, "
                  f"min_samples_split={self.RF_MIN_SAMPLES_SPLIT}, min_samples_leaf={self.RF_MIN_SAMPLES_LEAF}")

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

        print(f"\nSaved {len(models)} models to {models_path}")
        print(f"Saved PCA transformer to {pca_path}")
        print(f"Saved subtype columns to {subtype_columns_path}")
        print(f"Saved ESM2 embeddings to {esm2_embeddings_path}")

    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions using trained Random Forest models.

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
        print(f"Generating ESM2 embeddings for {len(df)} samples...")
        esm2_embeddings = self._generate_esm2_embeddings(
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
        )

        # 2. Apply PCA transformation
        print(f"Applying PCA transformation: {esm2_embeddings.shape[1]}D → {self.PCA_COMPONENTS}D")
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

        print(f"Generated predictions for {len(df_output)} samples")
        print(f"  Properties: {', '.join(models.keys())}")

        return df_output
