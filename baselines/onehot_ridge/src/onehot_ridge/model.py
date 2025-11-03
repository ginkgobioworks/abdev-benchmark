from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from abdev_core import BaseModel, PROPERTY_LIST


class OneHotRidgeModel(BaseModel):
    """Ridge regression model on one-hot encoded aligned VH and VL sequences.

    This model trains separate Ridge regression models for each property
    using one-hot encodings of the heavy and light chain sequences aligned
    in the AHo numbering scheme.

    The aligned VH and VL sequences are concatenated directly (without a
    separator), and each amino acid position is represented using a
    21-character vocabulary (20 canonical amino acids plus a '-' gap token).
    """

    ALPHA = 1.0  # Ridge regression regularization parameter
    VOCAB = list("ACDEFGHIKLMNPQRSTVWY-")

    def __init__(self) -> None:
        """Initialize model state."""
        self.encoder = None
        self.seq_len = None
        self.models = {}

    # ------------------------------------------------------------------
    def _prepare_onehot(
        self, heavy_sequences: list[str], light_sequences: list[str]
    ) -> np.ndarray:
        """Generate concatenated VH+VL one-hot encodings for all aligned sequence pairs."""
        print("[INFO] Preparing one-hot encodings...")

        # Concatenate heavy + light aligned sequences (no separator)
        combined = [f"{vh}{vl}" for vh, vl in zip(heavy_sequences, light_sequences)]
        print(f"[DEBUG] Example concatenated sequence (first): {combined[0][:60]}...")

        # Split each sequence into a list of amino acids
        split = [list(seq) for seq in combined]
        self.seq_len = len(split[0])
        print(f"[INFO] Total concatenated sequence length: {self.seq_len}")

        # Sanity check: ensure all sequences have identical length
        if not all(len(s) == self.seq_len for s in split):
            raise ValueError("All concatenated VH+VL sequences must have the same length.")

        # Create a DataFrame with one column per residue position
        df_split = pd.DataFrame(split, columns=[f"pos_{i}" for i in range(self.seq_len)])
        print(f"[INFO] Feature DataFrame shape before encoding: {df_split.shape}")

        # Define the amino acid alphabet (21 characters including '-')
        fixed_categories = [self.VOCAB] * self.seq_len

        # Initialize the one-hot encoder if not already created
        if self.encoder is None:
            self.encoder = ColumnTransformer([
                ("onehot", OneHotEncoder(
                    categories=fixed_categories,
                    handle_unknown="ignore",
                    sparse_output=False
                ), df_split.columns.tolist())
            ])
            print("[INFO] Initialized OneHotEncoder with fixed amino acid categories.")

        # Fit and transform sequences into one-hot encoded array
        X = self.encoder.fit_transform(df_split)
        print(f"[INFO] One-hot feature matrix shape: {X.shape}")
        print("[INFO] One-hot encoding complete.\n")
        return X

    # ------------------------------------------------------------------
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train Ridge regression models on one-hot encodings for each property."""
        print("[INFO] Starting training process...")
        run_dir.mkdir(parents=True, exist_ok=True)

        # --- Prepare features ---
        print("[STEP] Generating one-hot features from aligned sequences...")
        X_all = self._prepare_onehot(
            df["heavy_aligned_aho"].tolist(),
            df["light_aligned_aho"].tolist(),
        )

        models = {}

        # --- Train one model per property ---
        for property_name in PROPERTY_LIST:
            if property_name not in df.columns:
                print(f"[WARN] Property '{property_name}' not found in dataframe. Skipping.")
                continue

            mask = df[property_name].notna()
            if not mask.any():
                print(f"[WARN] No non-null values found for '{property_name}'. Skipping.")
                continue

            X = X_all[mask]
            y = df.loc[mask, property_name].values

            print(f"[TRAIN] Fitting Ridge regression for '{property_name}'...")
            print(f"         Using {mask.sum()} samples, feature dim = {X.shape[1]}")

            model = Ridge(alpha=self.ALPHA, random_state=seed)
            model.fit(X, y)
            models[property_name] = model

            print(f"[DONE] Trained model for '{property_name}'. Coeff shape: {model.coef_.shape}\n")

        # --- Save models and encoder ---
        models_path = run_dir / "models.pkl"
        npy_path = run_dir / "onehot_features.npy"

        with open(models_path, "wb") as f:
            pickle.dump(
                {"models": models, "encoder": self.encoder, "seq_len": self.seq_len},
                f,
            )

        np.save(npy_path, X_all)

        print(f"✓ Training complete.")
        print(f"  → Models saved to: {models_path}")
        print(f"  → One-hot feature matrix saved to: {npy_path}")
        print(f"  → Total trained models: {len(models)}\n")

    # ------------------------------------------------------------------
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions for all samples using trained Ridge models."""
        print("[INFO] Starting prediction...")

        models_path = run_dir / "models.pkl"
        if not models_path.exists():
            raise FileNotFoundError(f"[ERROR] Models not found: {models_path}")

        # --- Load trained models ---
        with open(models_path, "rb") as f:
            data = pickle.load(f)

        models = data["models"]
        encoder: ColumnTransformer = data["encoder"]
        seq_len = data["seq_len"]

        # --- Prepare features for inference ---
        combined = [f"{vh}{vl}" for vh, vl in zip(
            df["heavy_aligned_aho"], df["light_aligned_aho"]
        )]
        split = [list(seq) for seq in combined]

        if any(len(s) != seq_len for s in split):
            raise ValueError(f"Sequence length mismatch: expected {seq_len}")

        df_split = pd.DataFrame(split, columns=[f"pos_{i}" for i in range(seq_len)])
        print(f"[INFO] Transforming {len(df_split)} sequences into one-hot features...")
        X = encoder.transform(df_split)
        print(f"[INFO] Feature matrix shape for prediction: {X.shape}\n")

        # --- Predict properties ---
        df_output = df[["antibody_name", "heavy_aligned_aho", "light_aligned_aho"]].copy()

        for property_name, model in models.items():
            preds = model.predict(X)
            df_output[property_name] = preds
            print(f"[PREDICT] {property_name}: generated {len(preds)} predictions "
                  f"(mean={np.mean(preds):.3f}, std={np.std(preds):.3f})")

        print("\n✓ Prediction complete.")
        return df_output
