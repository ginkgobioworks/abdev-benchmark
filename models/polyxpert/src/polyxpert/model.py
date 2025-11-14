"""PolyXpert model implementation."""

import os
os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from pathlib import Path
import json
import urllib.request
import zipfile
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from abdev_core import BaseModel


class PolyXpertModel(BaseModel):
    """PolyXpert is a fine-tuned ESM-2 model for predicting antibody polyreactivity.
    
    This model uses a pre-trained transformer model to predict polyreactivity (PR_CHO).
    The model processes heavy (VH) and light (VL) chain sequences and outputs
    a probability score indicating polyreactivity risk.
    
    The model outputs a polyreactivity score where higher scores indicate higher
    polyreactivity (worse developability).
    
    Model weights are automatically downloaded to cache on first use.
    """
    
    BATCH_SIZE = 16
    MAX_LENGTH = 512
    MODEL_URL = "https://i.uestc.edu.cn/PolyXpert.zip"
    
    def __init__(self) -> None:
        """Initialize model (lazy load transformers on first use)."""
        self.tokenizer = None
        self.model = None
        self.device = None
        self.model_path = None
    
    def _get_cache_dir(self) -> Path:
        """Get the cache directory for model weights.
        
        Uses ~/.cache/polyxpert by default.
        """
        cache_dir = Path.home() / ".cache" / "polyxpert"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _download_model_weights(self) -> Path:
        """Download model weights to cache directory.
        
        Returns:
            Path to the downloaded and extracted model directory
        """
        cache_dir = self._get_cache_dir()
        # The zip file extracts to 'esm2_finetuning' directory
        model_dir = cache_dir / "esm2_finetuning"
        
        # Check if already downloaded
        if model_dir.exists() and (model_dir / "config.json").exists():
            print(f"Using cached PolyXpert model weights from {model_dir}")
            return model_dir
        
        # Download model weights
        print(f"Downloading PolyXpert model weights from {self.MODEL_URL}...")
        zip_path = cache_dir / "PolyXpert.zip"
        
        try:
            urllib.request.urlretrieve(self.MODEL_URL, zip_path)
            print(f"Downloaded to {zip_path}")
            
            # Extract
            print("Extracting model weights...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(cache_dir)
            
            # Clean up zip file
            zip_path.unlink()
            print(f"Model weights extracted to {model_dir}")
            
        except Exception as e:
            raise RuntimeError(
                f"Failed to download PolyXpert model weights from {self.MODEL_URL}. "
                f"Error: {e}"
            ) from e
        
        if not model_dir.exists():
            raise RuntimeError(
                f"Model directory {model_dir} not found after extraction. "
                f"The zip file structure may have changed."
            )
        
        return model_dir
    
    def _initialize_model(self) -> None:
        """Lazy initialize the transformer model and tokenizer.
        
        Automatically downloads model weights if not present in cache.
        """
        if self.model is not None:
            return
        
        # Download or get cached model weights
        self.model_path = self._download_model_weights()
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading PolyXpert model on device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(
            str(self.model_path), 
            num_labels=2
        ).to(self.device)
        self.model.eval()
    
    def _create_dataset(self, df: pd.DataFrame) -> Dataset:
        """Create dataset from dataframe with VH/VL sequences.
        
        Args:
            df: DataFrame with vh_protein_sequence and vl_protein_sequence columns
            
        Returns:
            HuggingFace Dataset with tokenized sequences
        """
        seqs_df = df.copy()
        
        # Replace non-standard amino acids with X
        seqs_df["vh_protein_sequence"] = seqs_df["vh_protein_sequence"].str.replace(
            '|'.join(["O", "B", "U", "Z"]), "X", regex=True
        )
        seqs_df["vl_protein_sequence"] = seqs_df["vl_protein_sequence"].str.replace(
            '|'.join(["O", "B", "U", "Z"]), "X", regex=True
        )
        
        # Add spaces between amino acids
        seqs_df['VH'] = seqs_df.apply(lambda row: " ".join(row["vh_protein_sequence"]), axis=1)
        seqs_df['VL'] = seqs_df.apply(lambda row: " ".join(row["vl_protein_sequence"]), axis=1)
        
        # Tokenize sequences
        tokenized = self.tokenizer(
            list(seqs_df['VH']),
            list(seqs_df['VL']), 
            max_length=self.MAX_LENGTH,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )
        
        dataset = Dataset.from_dict(tokenized)
        dataset = dataset.with_format("torch")
        return dataset
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """No-op training - this model uses pre-trained weights.
        
        Saves configuration to run_dir for consistency.
        
        Args:
            df: Training dataframe (not used)
            run_dir: Directory to save configuration
            seed: Random seed (not used)
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Get cache directory for reference
        cache_dir = self._get_cache_dir()
        
        # Save configuration for reference
        config = {
            "model_type": "polyxpert",
            "note": "Non-training baseline using pre-trained fine-tuned ESM-2 model",
            "cache_dir": str(cache_dir),
            "model_url": self.MODEL_URL
        }
        
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        
        print("=" * 60)
        print("PolyXpert Model Information")
        print("=" * 60)
        print("PolyXpert is a fine-tuned ESM-2 model for antibody polyreactivity prediction.")
        print("The model has been pre-trained and does not require additional training.")
        print("")
        print("Model Components:")
        print("- Fine-tuned ESM-2 transformer model")
        print("- Sequence-based prediction (VH + VL)")
        print("- Outputs PR_CHO (polyreactivity) score")
        print(f"- Model weights cached at: {cache_dir}")
        print("")
        print("Model weights will be automatically downloaded on first use.")
        print("To use PolyXpert, simply call the predict() method with your antibody sequences.")
        print("=" * 60)
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate polyreactivity predictions for all samples.
        
        Args:
            df: Input dataframe with vh_protein_sequence and vl_protein_sequence columns
            run_dir: Directory containing model configuration
            
        Returns:
            DataFrame with predictions including PR_CHO column
        """
        # Initialize model
        self._initialize_model()
        
        # Create dataset and dataloader
        dataset = self._create_dataset(df)
        dataloader = DataLoader(
            dataset, 
            batch_size=self.BATCH_SIZE, 
            shuffle=False,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # Generate predictions
        predictions = []
        self.model.eval()
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                # Get model predictions
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                
                # Convert to probabilities
                proba = torch.softmax(logits, dim=1)
                
                # Extract probability of polyreactive class (class 1)
                # Higher probability = higher polyreactivity (worse)
                for i in range(proba.size(0)):
                    predictions.append(proba[i, 1].item())
        
        # Create output dataframe
        result = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
        result["PR_CHO"] = predictions
        
        return result

