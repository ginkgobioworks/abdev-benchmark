"""Base interface for baseline models."""

from abc import ABC, abstractmethod
from pathlib import Path
import pandas as pd


class BaseModel(ABC):
    """Abstract base class defining the interface for baseline models.
    
    Contract:
    - train() trains on ALL provided data and writes model artifacts to run_dir
    - predict() reads model artifacts from run_dir and returns a predictions DataFrame
    
    Important: Models should NOT implement cross-validation internally. The orchestrator
    is responsible for splitting data and calling train/predict multiple times if needed.
    
    All baselines must implement both methods. Non-training baselines should implement
    a no-op train() method or save minimal state.
    """
    
    @abstractmethod
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        """Train the model on ALL provided data and save artifacts to run_dir.
        
        Args:
            df: Training dataframe with features and labels
            run_dir: Directory to save model artifacts (models, weights, etc.)
            seed: Random seed for reproducibility
            
        Notes:
            - Train on ALL samples in df (do NOT implement internal CV splitting)
            - For models that don't require training, this can be a no-op
            - Must create run_dir if it doesn't exist
            - All model artifacts must be saved to run_dir for later prediction
            - The orchestrator handles data splitting for cross-validation
        """
        ...
    
    @abstractmethod
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Generate predictions for ALL provided samples using saved model artifacts.
        
        Args:
            df: Input dataframe with sequences (and any required features)
            run_dir: Directory containing saved model artifacts from train()
            
        Returns:
            DataFrame with prediction results containing columns:
            ['antibody_name', 'vh_protein_sequence', 'vl_protein_sequence', ...]
            where ... are predicted property columns
            
        Notes:
            - Must load model artifacts from run_dir
            - Predict on ALL samples in df
            - Do NOT save predictions; return the DataFrame for orchestrator to handle
            - The orchestrator is responsible for saving predictions to the appropriate location
        """
        ...

