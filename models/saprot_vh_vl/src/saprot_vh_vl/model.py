"""Saprot_VH_VL model implementation."""

from pathlib import Path
import pandas as pd

from abdev_core import BaseModel, PROPERTY_LIST

import numpy as np

import torch
from transformers import AutoTokenizer, AutoModel
import pickle
from sklearn.linear_model import Ridge

import subprocess

import os
import time



class Saprot_VH_VL_Model(BaseModel):
    """Saprot_VH_VL: baseline using protein language model features.
        This model trains separate Ridge regression models for each property
        using embeddings from the SaProt protein language model.

        This baseline embeds heavy (VH) and light (VL) chains separately using their sequences and structures for input.
    """
    MODEL_NAME = "westlake-repl/SaProt_35M_AF2"
    ALPHA = 1.0  # Ridge regression regularization parameter

    structure_dir_vh = Path("../../data/structures/AntiBodyBuilder3/GDPa1")
    structure_dir_vl = Path("../../data/structures/MOE_structures/GDPa1")

    def __init__(self) -> None:
        """Initialize model (lazy load transformers on first use)."""
        self.tokenizer = None
        self.model = None
        self.device = None

    def _initialize_model(self) -> None:
        """Lazy initialize the transformer model and tokenizer."""
        if self.model is not None:
            return

        #Adding PyTorch Optimization for Apple Silicon
        device = ""
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()  # Set to evaluation mode

    def get_structure_aware_seq(self, sequence: str,pdb_path) -> str:
        """
        Encode PDB structure using Foldseek and create structure-aware sequences by interleaving 3di descriptors.
        Falls back to sequence only mode in case of missing/erroneous structures.
        
        Args:
            sequence: Amino acid sequence
            pdb_path: Path to PDB file

    
        Returns:
            structure_aware_seq: Sequence with structure tokens (for SaProt)

            or 

            sequence: if problems arise in fetching structure tokens
        """
        try:
            # Create temporary output file
            tmp_save_path = f"foldseek_tmp_{os.getpid()}_{time.time()}.tsv"
        
            # Run foldseek with official SaProt flags
            cmd = [
            "foldseek", "structureto3didescriptor",
            "-v", "0",  # Suppress verbose output
            "--threads", "1",
            "--chain-name-mode", "1",  # Include chain names
            str(pdb_path),
            tmp_save_path
            ]
        
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        
            if result.returncode != 0:
                raise RuntimeError(f"Foldseek failed: {result.stderr}")
        
            # Read output file
            if not os.path.exists(tmp_save_path):
                raise FileNotFoundError(f"Foldseek output not found: {tmp_save_path}")
        
            with open(tmp_save_path, 'r') as f:
                line = f.readline().strip()
            
                if not line:
                    raise ValueError("Foldseek output is empty")
            
                parts = line.split('\t')
            
                if len(parts) < 3:
                    raise ValueError(f"Unexpected output format: {len(parts)} columns")
            
                # Extract sequences
                aa_seq = parts[1]  # Amino acid sequence
                struc_seq = parts[2]  # 3Di structure sequence
        
            # Clean up temporary file
            try:
                os.remove(tmp_save_path)
                if os.path.exists(tmp_save_path + ".dbtype"):
                    os.remove(tmp_save_path + ".dbtype")
            except:
                pass
        
            # Interleave amino acids with structure tokens (lowercase)
            # SaProt format: Aa Bb Cc where A,B,C = amino acids, a,b,c = structure tokens
            structure_aware_seq = ''.join(
                f"{aa}{st.lower()}" for aa, st in zip(aa_seq, struc_seq)
            )
        
            # Validate length
            if len(aa_seq) != len(sequence):
                print(f"Warning: Length mismatch - PDB has {len(aa_seq)} residues, "
                      f"sequence has {len(sequence)}. Using sequence-only mode.")
                return sequence
        
            return structure_aware_seq
        
        except Exception as e:
            print(f"Warning: Structure encoding failed for {pdb_path}: {e}")
            print("Falling back to sequence-only mode")
            return sequence  # Fallback: use sequence without structure

       
    def extract_saprot_embedding(self, sequence: str, pdb_path=None) -> np.ndarray:
        """
        Extract SaProt embedding from sequence (with optional structure).
    
        Args:
            sequence: Amino acid sequence
            pdb_path: Optional path to PDB file for structure-aware encoding
    
        Returns:
            embedding: Numpy array of shape (480,)
        """
        # Get structure-aware sequence if PDB provided
        if pdb_path.exists():
            input_seq = self.get_structure_aware_seq(sequence, pdb_path)
        else:
            input_seq = sequence  # Sequence-only mode
    
        # Tokenize
        inputs = self.tokenizer(
            #input_seq,
            str(input_seq),
            return_tensors="pt",
            padding=False,  # Single sequence, no padding needed
            truncation=True,
            max_length=1024
        )
    
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
    
        # Extract embeddings with output_hidden_states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]  # Last layer
    
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

        
        return embedding  # Return shape (480,)
    
    def _generate_embeddings(
        self,
        antibody_names: list[str],
        vh_sequences: list[str],
        vl_sequences: list[str],
        *,
        context: str = "unknown",
    ) -> np.ndarray:
        """
        Generate concatenated VH+VL embeddings for all antibodies.
        Each antibody is processed individually(sequence + structure) to avoid padding token contamination.
        Heavy and light chain embeddings are concatenated to form the final representation.
        
        Args:
            antibody_names: list of antibody names
            vh_sequences: list of variable heavy sequences
            vl_sequences: list of variable light sequences
            context: string describing usage context, used to select structure roots
    
        Returns:
            embeddings: Numpy array of shape (n_sequences, 960)
        """

        self._initialize_model()

        embeddings_list = []

        # Select structure roots based on context:
        # - GDPa1 train/CV data lives under .../GDPa1
        # - heldout test structures live under .../heldout_test
        if context == "predict_heldout":
            vh_root = Path("../../data/structures/AntiBodyBuilder3/heldout_test")
            vl_root = Path("../../data/structures/MOE_structures/heldout_test")
        else:
            vh_root = self.structure_dir_vh
            vl_root = self.structure_dir_vl

        for _, (antibody_name, vh_seq, vl_seq) in enumerate(zip(antibody_names, vh_sequences, vl_sequences)):
            vh_pdb = vh_root / f"{antibody_name}.pdb"
            vl_pdb = vl_root / f"{antibody_name}.pdb"

            # Extract VH embedding (with structure if available)
            vh_embed = self.extract_saprot_embedding(vh_seq, vh_pdb) #if vh_pdb.exists() else None)
            # Extract VL embedding (with structure if available)
            vl_embed = self.extract_saprot_embedding(vl_seq, vl_pdb) #if vl_pdb.exists() else None)
            
            combined_embedding = np.concatenate([vh_embed, vl_embed])
            embeddings_list.append(combined_embedding)

            print("generating saprot embeddings", antibody_name)

        embeddings = np.stack(embeddings_list)  

        return embeddings


    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train Ridge regression models on SaProt embeddings for each property.
        
        Args:
            df: Training dataframe with VH/VL sequences and property labels
            run_dir: Directory to save trained models
            seed: Random seed (for reproducibility)
        """

        np.random.seed(seed)
        torch.manual_seed(seed)

        run_dir.mkdir(parents=True, exist_ok=True)
        

        # Generate embeddings for all training samples
        embeddings = self._generate_embeddings(
            df["antibody_name"].tolist(),
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
            context="train",
        )

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
            
            # Train Ridge regression with alpha=1.0 and seeded random state
            model = Ridge(alpha=self.ALPHA, random_state=seed)
            model.fit(X, y)
            models[property_name] = model

        # Save models and embeddings
        models_path = run_dir / "models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(models, f)

        embeddings_path = run_dir / "embeddings.npy"
        np.save(embeddings_path, embeddings)



    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions for all samples using trained models.
        
        Args:
            df: Input dataframe with VH/VL sequences
            run_dir: Directory containing trained models
            
        Returns:
            DataFrame with predictions for each property
        """
        # Load trained models

        self._initialize_model()
        
        models_path = run_dir / "models.pkl"
        if not models_path.exists():
            raise FileNotFoundError(f"Models not found: {models_path}")

        with open(models_path, "rb") as f:
            models = pickle.load(f)

        # Heuristic: heldout test data has no assay columns like 'Titer'
        if "Titer" in df.columns:
            pred_context = "predict_train_or_cv"
        else:
            pred_context = "predict_heldout"

        # Generate embeddings for input data
        embeddings = self._generate_embeddings(
            df["antibody_name"].tolist(),
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
            context=pred_context,
        )

        # Generate predictions for each property
        df_output = df[
            ["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]
        ].copy()

        for property_name, model in models.items():
            predictions = model.predict(embeddings)
            df_output[property_name] = predictions
        return df_output
