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
from Bio.PDB import PDBParser, PDBIO, Select
import tempfile

class Saprot_VH_VL_Model(BaseModel):
    """Saprot_VH_VL: baseline using protein language model features.
    
    This model trains separate Ridge regression models for each property
    using embeddings from the SaProt protein language model.
    
    This baseline embeds heavy (VH) and light (VL) chains separately using their sequences and structures for input.
    """
    
    MODEL_NAME = "westlake-repl/SaProt_35M_AF2"
    ALPHA = 1.0
    
    # Use MOE structures for both (they contain both chains)
    structure_dir_train = Path("../../data/structures/MOE_structures/GDPa1")  
    structure_dir_heldout = Path("../../data/structures/MOE_structures/heldout_test") 
    
    def __init__(self) -> None:
        """Initialize model (lazy load transformers on first use)."""
        self.tokenizer = None
        self.model = None
        self.device = None

    def _initialize_model(self) -> None:
        """Lazy initialize the transformer model and tokenizer."""
        if self.model is not None:
            return
        
        # Detect device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(self.MODEL_NAME)
        self.model = AutoModel.from_pretrained(self.MODEL_NAME).to(self.device)
        self.model.eval()

    # NEW: Chain splitting helper
    class _ChainSelect(Select):
        """Select specific chain from PDB structure."""
        def __init__(self, chain_id):
            self.chain_id = chain_id
        
        def accept_chain(self, chain):
            return chain.id == self.chain_id

    # NEW: Extract chain to temporary file
    def _extract_chain_to_temp(self, pdb_path, chain_id):
        """
        Extract specific chain from PDB to temporary file.
        
        Args:
            pdb_path: Path to complexed PDB file
            chain_id: Chain ID to extract ('B' for VH in MOE, 'A' for VL)
        
        Returns:
            temp_pdb_path: Path to temporary PDB file with single chain
        """
        try:
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure('antibody', pdb_path)
            
            # Create temporary file
            temp_fd, temp_path = tempfile.mkstemp(suffix='.pdb')
            os.close(temp_fd)
            
            # Write chain to temporary file
            io = PDBIO()
            io.set_structure(structure)
            io.save(temp_path, self._ChainSelect(chain_id))
            
            return temp_path
        except Exception as e:
            print(f"Warning: Chain extraction failed for {pdb_path}, chain {chain_id}: {e}")
            return None

    def get_structure_aware_seq(self, sequence: str, pdb_path) -> str:
        """
        Encode PDB structure using Foldseek and create structure-aware sequences.
        Falls back to sequence only mode in case of missing/erroneous structures.
        """
        try:
            tmp_save_path = f"foldseek_tmp_{os.getpid()}_{time.time()}.tsv"
            
            cmd = [
                "foldseek", "structureto3didescriptor",
                "-v", "0",
                "--threads", "1",
                "--chain-name-mode", "1",
                str(pdb_path),
                tmp_save_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            if result.returncode != 0:
                raise RuntimeError(f"Foldseek failed: {result.stderr}")
            
            if not os.path.exists(tmp_save_path):
                raise FileNotFoundError(f"Foldseek output not found: {tmp_save_path}")
            
            with open(tmp_save_path, 'r') as f:
                line = f.readline().strip()
                if not line:
                    raise ValueError("Foldseek output is empty")
            
            parts = line.split('\t')
            if len(parts) < 3:
                raise ValueError(f"Unexpected output format: {len(parts)} columns")
            
            aa_seq = parts[1]
            struc_seq = parts[2]
            
            # Clean up
            try:
                os.remove(tmp_save_path)
                if os.path.exists(tmp_save_path + ".dbtype"):
                    os.remove(tmp_save_path + ".dbtype")
            except:
                pass
            
            # Interleave
            structure_aware_seq = ''.join(
                f"{aa}{st.lower()}" for aa, st in zip(aa_seq, struc_seq)
            )
            
            if len(aa_seq) != len(sequence):
                print(f"Warning: Length mismatch - using sequence-only mode")
                return sequence
            
            return structure_aware_seq
            
        except Exception as e:
            print(f"Warning: Structure encoding failed for {pdb_path}: {e}")
            return sequence

    def extract_saprot_embedding(self, sequence: str, pdb_path=None) -> np.ndarray:
        """Extract SaProt embedding from sequence (with optional structure)."""
        if pdb_path.exists():
            input_seq = self.get_structure_aware_seq(sequence, pdb_path)
        else:
            input_seq = sequence
        
        inputs = self.tokenizer(
            str(input_seq),
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=1024
        )
        
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        hidden_states = outputs.hidden_states[-1]
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
        sum_mask = torch.sum(mask_expanded, dim=1)
        mean_pooled = sum_embeddings / sum_mask
        
        embedding = mean_pooled.detach().cpu().numpy().squeeze(0)
        
        return embedding

    def _generate_embeddings(self, antibody_names: list[str], 
                           vh_sequences: list[str], 
                           vl_sequences: list[str],
                       structure_dir: Path = None) -> np.ndarray:
        """Generate concatenated VH+VL embeddings for all antibodies."""
        self._initialize_model()
        
        embeddings_list = []
        
        for antibody_name, vh_seq, vl_seq in zip(antibody_names, vh_sequences, vl_sequences):
            # Use same PDB file for both (contains both chains)
            complexed_pdb = structure_dir / f"{antibody_name}.pdb"
            
            # Extract chains to temporary files if PDB exists
            if complexed_pdb.exists():
                # MOE uses chain B for VH, chain A for VL
                temp_vh_pdb = self._extract_chain_to_temp(complexed_pdb, 'B')
                temp_vl_pdb = self._extract_chain_to_temp(complexed_pdb, 'A')
                
                try:
                    vh_pdb_path = Path(temp_vh_pdb) if temp_vh_pdb else complexed_pdb
                    vh_embed = self.extract_saprot_embedding(vh_seq, vh_pdb_path)
                    
                    vl_pdb_path = Path(temp_vl_pdb) if temp_vl_pdb else complexed_pdb
                    vl_embed = self.extract_saprot_embedding(vl_seq, vl_pdb_path)
                    
                finally:
                    # Clean up temporary files
                    if temp_vh_pdb and os.path.exists(temp_vh_pdb):
                        os.unlink(temp_vh_pdb)
                    if temp_vl_pdb and os.path.exists(temp_vl_pdb):
                        os.unlink(temp_vl_pdb)
            else:
                # Fallback: sequence-only mode
                print(f"Warning: PDB not found for {antibody_name}, using sequence-only mode")
                vh_embed = self.extract_saprot_embedding(vh_seq, Path("nonexistent.pdb"))
                vl_embed = self.extract_saprot_embedding(vl_seq, Path("nonexistent.pdb"))
            
            combined_embedding = np.concatenate([vh_embed, vl_embed])
            embeddings_list.append(combined_embedding)
        
        embeddings = np.stack(embeddings_list)
        
        return embeddings

    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train Ridge regression models on SaProt embeddings for each property."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        run_dir.mkdir(parents=True, exist_ok=True)
        
        embeddings = self._generate_embeddings(
            df["antibody_name"].tolist(),
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
            structure_dir=self.structure_dir_train 
        )
        
        models = {}
        for property_name in PROPERTY_LIST:
            if property_name not in df.columns:
                continue
            
            not_na_mask = df[property_name].notna()
            df_property = df[not_na_mask]
            
            if len(df_property) == 0:
                print(f" Skipping {property_name}: no training data")
                continue
            
            X = embeddings[not_na_mask]
            y = df_property[property_name].values
            
            model = Ridge(alpha=self.ALPHA, random_state=seed)
            model.fit(X, y)
            models[property_name] = model
        
        models_path = run_dir / "models.pkl"
        with open(models_path, "wb") as f:
            pickle.dump(models, f)

        #please un-comment, if you would like to save embeddings -note: each CV generates its own embeddings which can be confusing
        #embeddings_path = run_dir / "embeddings.npy"
        #np.save(embeddings_path, embeddings)

    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions for all samples using trained models."""
        self._initialize_model()
        
        models_path = run_dir / "models.pkl"
        if not models_path.exists():
            raise FileNotFoundError(f"Models not found: {models_path}")
        
        with open(models_path, "rb") as f:
            models = pickle.load(f)
        
        embeddings = self._generate_embeddings(
            df["antibody_name"].tolist(),
            df["vh_protein_sequence"].tolist(),
            df["vl_protein_sequence"].tolist(),
            structure_dir=self.structure_dir_heldout 
        )
        
        df_output = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
        
        #embeddings_path = run_dir / "heldout_embeddings.npy"
        #np.save(embeddings_path, embeddings)
        
        for property_name, model in models.items():
            predictions = model.predict(embeddings)
            df_output[property_name] = predictions
        
        return df_output
