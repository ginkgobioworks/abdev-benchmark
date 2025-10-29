"""p-IgGen baseline model implementation."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from abdev_core import BaseModel, PROPERTY_LIST


class PiGGenModel(BaseModel):
    """Ridge regression model on p-IgGen embeddings.

    This model trains separate Ridge regression models for each property
    using embeddings from the p-IgGen foundation model.

    p-IgGen is a protein language model pre-trained on paired VH/VL antibody
    sequences. For each antibody, we concatenate heavy and light chains,
    tokenize them with special boundary markers, and generate mean-pooled
    embeddings across all tokens. These embeddings serve as features for
    downstream property prediction.
    """

    MODEL_NAME = "ollieturnbull/p-IgGen"
    BATCH_SIZE = 16

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
        print(f"Loading p-IgGen model on device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(self.MODEL_NAME).to(
            self.device
        )
        self.model.eval()  # Set to evaluation mode

    def _create_paired_sequences(
        self, vh_sequences: list[str], vl_sequences: list[str]
    ) -> list[str]:
        """Create paired VH/VL sequences for p-IgGen.

        Concatenates heavy and light chains with amino acid-level spacing
        and adds boundary tokens ("1" for start, "2" for end).

        Args:
            vh_sequences: List of VH protein sequences
            vl_sequences: List of VL protein sequences

        Returns:
            List of paired sequences with boundary markers
        """
        sequences = [
            "1" + " ".join(vh) + " " + " ".join(vl) + "2"
            for vh, vl in zip(vh_sequences, vl_sequences)
        ]
        return sequences

    def _generate_embeddings(self, sequences: list[str]) -> np.ndarray:
        """Generate mean-pooled embeddings from sequence pairs.

        Args:
            sequences: List of paired VH/VL sequences

        Returns:
            Array of shape (n_sequences, embedding_dim)
        """
        self._initialize_model()

        embeddings_list = []

        for i in range(0, len(sequences), self.BATCH_SIZE):
            batch_sequences = sequences[i : i + self.BATCH_SIZE]

            # Tokenize batch
            batch = self.tokenizer(
                batch_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )

            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(
                    batch["input_ids"].to(self.device),
                    return_rep_layers=[-1],
                    output_hidden_states=True,
                )
                hidden_states = outputs["hidden_states"][-1].detach().cpu().numpy()

            # Mean pool across token dimension
            mean_pooled = hidden_states.mean(axis=1)
            embeddings_list.append(mean_pooled)

        embeddings = np.concatenate(embeddings_list)
        return embeddings

    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train Ridge regression models on p-IgGen embeddings for each property.

        Args:
            df: Training dataframe with VH/VL sequences and property labels
            run_dir: Directory to save trained models
            seed: Random seed (for reproducibility)
        """
        run_dir.mkdir(parents=True, exist_ok=True)

        # Generate embeddings for all training samples
        print(f"Generating embeddings for {len(df)} training samples...")
        sequences = self._create_paired_sequences(
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
        )
        embeddings = self._generate_embeddings(sequences)

        print(f"Generated embeddings with shape: {embeddings.shape}")

        # Train Ridge regression models for each property
        models = {}

        for property_name in PROPERTY_LIST:
            # Skip if property not in training data
            if property_name not in df.columns:
                continue

            # Filter to samples with non-null values
            not_na_mask = df[property_name].notna()
            df_property = df[not_na_mask]

            if len(df_property) == 0:
                print(f"  Skipping {property_name}: no training data")
                continue

            # Get corresponding embeddings for non-null samples
            X = embeddings[not_na_mask]
            y = df_property[property_name].values

            # Train Ridge regression
            model = Ridge()
            model.fit(X, y)
            models[property_name] = model

            print(f"  Trained model for {property_name} on {len(df_property)} samples")

        # Save models and embeddings
        models_path = run_dir / "models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(models, f)

        embeddings_path = run_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)

        print(f"Saved {len(models)} models to {models_path}")
        print(f"Saved embeddings to {embeddings_path}")

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

        # Generate embeddings for input data
        print(f"Generating embeddings for {len(df)} samples...")
        sequences = self._create_paired_sequences(
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
        )
        embeddings = self._generate_embeddings(sequences)

        # Generate predictions for each property
        df_output = df[
            ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        ].copy()

        for property_name, model in models.items():
            predictions = model.predict(embeddings)
            df_output[property_name] = predictions

        print(f"Generated predictions for {len(df_output)} samples")
        print(f"  Properties: {', '.join(models.keys())}")

        return df_output
