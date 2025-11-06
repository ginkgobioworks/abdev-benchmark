"""ESM2 + TAP Ridge regression model implementation with PCA dimensionality reduction."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
import torch
from transformers import AutoTokenizer, AutoModel

from abdev_core import BaseModel, PROPERTY_LIST, load_features


# TAP feature names
TAP_FEATURE_NAMES = ["SFvCSP", "PSH", "PPC", "PNC", "CDR Length"]


class ESM2TapRidgeModel(BaseModel):
    """Ridge regression model on PCA-reduced ESM2 embeddings + TAP + subtype features.


    Feature breakdown:
    - ESM2 (PCA-reduced): 50 dimensions
    - TAP features: 5 dimensions
    - hc_subtype (one-hot): 3 dimensions (IgG1, IgG2, IgG4)
    - lc_subtype (one-hot): 2 dimensions (Kappa, Lambda)
    """

    MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
    PCA_COMPONENTS = 50  # Reduce 640D → 50D
    ALPHA = 1.0  # Ridge regression regularization parameter

    def __init__(self) -> None:
        """Initialize model (lazy load transformers on first use)."""
        self.tokenizer = None
        self.model = None
        self.device = None

    def _initialize_model(self) -> None:
        """Lazy initialize the transformer model and tokenizer."""
        if self.model is not None:
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()  # Set to evaluation mode

    def _embed_sequence(self, sequence: str) -> np.ndarray:
        """Generate mean-pooled embedding for a single sequence.

        Processes one sequence at a time to avoid padding token contamination.
        Uses attention mask to exclude padding tokens from mean pooling.

        Args:
            sequence: Protein sequence (amino acid string)

        Returns:
            1D array of shape (embedding_dim,) with mean-pooled representation
        """
        # Tokenize single sequence
        inputs = self.tokenizer(
            sequence,
            return_tensors="pt",
            padding=False,  # No padding for single sequence
            truncation=True,
            max_length=1024,
        )

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            # Extract last hidden state: (1, seq_len, embedding_dim)
            hidden_states = outputs.hidden_states[-1]

        # Mean pool over sequence length, excluding padding tokens
        # attention_mask is (1, seq_len), expand to (1, seq_len, 1) for broadcasting
        mask_expanded = attention_mask.unsqueeze(-1).float()

        # Sum embeddings where mask is 1, then divide by number of non-padding tokens
        sum_embeddings = torch.sum(
            hidden_states * mask_expanded, dim=1
        )  # (1, embedding_dim)
        sum_mask = torch.sum(mask_expanded, dim=1)  # (1, 1)
        mean_pooled = sum_embeddings / sum_mask  # (1, embedding_dim)

        # Convert to numpy and squeeze batch dimension
        embedding = mean_pooled.detach().cpu().numpy().squeeze(0)  # (embedding_dim,)

        return embedding

    def _generate_embeddings(
        self, vh_sequences: list[str], vl_sequences: list[str]
    ) -> np.ndarray:
        """Generate concatenated VH+VL embeddings for all sequence pairs.

        Each sequence is processed individually to avoid padding token contamination.
        Heavy and light chain embeddings are concatenated to form the final representation.

        Returns:
            Array of shape (n_sequences, 2*embedding_dim) where each row is
            the concatenation of VH and VL mean-pooled embeddings
        """
        self._initialize_model()

        embeddings_list = []

        for _, (vh_seq, vl_seq) in enumerate(zip(vh_sequences, vl_sequences)):
            vh_embedding = self._embed_sequence(vh_seq)
            vl_embedding = self._embed_sequence(vl_seq)

            combined_embedding = np.concatenate([vh_embedding, vl_embedding])
            embeddings_list.append(combined_embedding)

        embeddings = np.stack(embeddings_list)  # [n_sequences, 2*embedding_dim]
        return embeddings

    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train Ridge regression models on PCA-reduced ESM2 + TAP features.

        Args:
            df: Training dataframe with VH/VL sequences and property labels
            run_dir: Directory to save trained models
            seed: Random seed for reproducibility
        """
        # Set random seeds for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        run_dir.mkdir(parents=True, exist_ok=True)

        # 1. Generate ESM2 embeddings (640D)
        # print("Generating ESM2 embeddings...")
        esm2_embeddings = self._generate_embeddings(
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

        # 6. Train Ridge regression models for each property
        models = {}

        for property_name in PROPERTY_LIST:
            # Skip if property not in training data
            if property_name not in df.columns:
                continue

            # Filter to samples with non-null values
            not_na_mask = df[property_name].notna()

            if not_na_mask.sum() == 0:
                # print(f"  Skipping {property_name}: no training data")
                continue

            X = X_combined[not_na_mask]
            y = df[property_name][not_na_mask].values

            # Train Ridge regression with alpha=1.0 and seeded random state
            model = Ridge(alpha=self.ALPHA, random_state=seed)
            model.fit(X, y)
            models[property_name] = model

            # print(f"  Trained Ridge for {property_name} on {len(y)} samples")
            # print(f"    Hyperparams: alpha={self.ALPHA}")

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

        embeddings_path = run_dir / "embeddings.npy"
        np.save(embeddings_path, esm2_embeddings)

        # print(f"\nSaved {len(models)} models to {models_path}")
        # print(f"Saved PCA transformer to {pca_path}")
        # print(f"Saved subtype columns to {subtype_columns_path}")
        # print(f"Saved ESM2 embeddings to {embeddings_path}")

    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions using trained Ridge regression models.

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
        esm2_embeddings = self._generate_embeddings(
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
        df_output = df[
            ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        ].copy()

        for property_name, model in models.items():
            predictions = model.predict(X_combined)
            df_output[property_name] = predictions

        # print(f"Generated predictions for {len(df_output)} samples")
        # print(f"  Properties: {', '.join(models.keys())}")

        return df_output
