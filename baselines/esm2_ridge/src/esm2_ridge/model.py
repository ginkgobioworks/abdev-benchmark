"""ESM2 Ridge baseline model implementation."""

from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
import torch
from transformers import AutoTokenizer, AutoModel

from abdev_core import BaseModel, PROPERTY_LIST


class ESM2RidgeModel(BaseModel):
    """Ridge regression model on ESM2 two-chain embeddings.
    
    This model trains separate Ridge regression models for each property
    using embeddings from the ESM2 protein language model.
    
    ESM2 is a general protein language model trained on evolutionary data.
    Unlike p-IgGen which processes paired sequences together, this baseline
    embeds heavy (VH) and light (VL) chains separately, then concatenates
    the embeddings to create a joint representation. This approach ensures
    no padding token contamination and allows the model to learn chain-specific
    features independently before combining them.
    
    Key differences from p-IgGen baseline:
    - Separate encoding: VH and VL are processed in separate forward passes
    - No batching: Each sequence is embedded individually to avoid padding
    - Concatenation: VH and VL embeddings are concatenated along feature dimension
    - General PLM: ESM2 is not antibody-specific, providing a general baseline
    """
    
    MODEL_NAME = "facebook/esm2_t6_8M_UR50D"
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
        print(f"Loading ESM2 model ({self.MODEL_NAME}) on device: {self.device}")
        
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
        
        # Move to device
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        # Generate embeddings
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
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)  # (1, embedding_dim)
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
        
        Args:
            vh_sequences: List of VH protein sequences
            vl_sequences: List of VL protein sequences
            
        Returns:
            Array of shape (n_sequences, 2*embedding_dim) where each row is
            the concatenation of VH and VL mean-pooled embeddings
        """
        self._initialize_model()
        
        n_sequences = len(vh_sequences)
        embeddings_list = []
        
        print(f"Generating embeddings for {n_sequences} sequences...")
        
        for i, (vh_seq, vl_seq) in enumerate(zip(vh_sequences, vl_sequences)):
            if (i + 1) % 50 == 0:
                print(f"  Processed {i + 1}/{n_sequences} sequences")
            
            # Embed VH and VL separately
            vh_embedding = self._embed_sequence(vh_seq)
            vl_embedding = self._embed_sequence(vl_seq)
            
            # Concatenate along feature dimension
            combined_embedding = np.concatenate([vh_embedding, vl_embedding])
            embeddings_list.append(combined_embedding)
        
        embeddings = np.stack(embeddings_list)
        return embeddings
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train Ridge regression models on ESM2 embeddings for each property.
        
        Args:
            df: Training dataframe with VH/VL sequences and property labels
            run_dir: Directory to save trained models
            seed: Random seed (for reproducibility)
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate embeddings for all training samples
        print(f"Generating embeddings for {len(df)} training samples...")
        embeddings = self._generate_embeddings(
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
        )
        
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
            
            # Train Ridge regression with alpha=1.0
            model = Ridge(alpha=self.ALPHA, random_state=seed)
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
        embeddings = self._generate_embeddings(
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
        )
        
        # Generate predictions for each property
        df_output = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
        
        for property_name, model in models.items():
            predictions = model.predict(embeddings)
            df_output[property_name] = predictions
        
        print(f"Generated predictions for {len(df_output)} samples")
        print(f"  Properties: {', '.join(models.keys())}")
        
        return df_output

