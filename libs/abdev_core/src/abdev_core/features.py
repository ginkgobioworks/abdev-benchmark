"""Feature loading and management for baseline models."""

from pathlib import Path
from typing import Optional, Dict
import pandas as pd


class FeatureLoader:
    """Centralized feature loading for baseline models.
    
    Provides access to pre-computed features as dictionary lookups by antibody_name.
    Baselines can optionally use features without needing to know file paths.
    """
    
    def __init__(self, features_dir: Optional[Path] = None):
        """Initialize the feature loader.
        
        Args:
            features_dir: Path to features directory. If None, uses standard location
                         relative to this package (../../features/processed_features)
        """
        if features_dir is None:
            # Default to standard repository location
            features_dir = Path(__file__).parent.parent.parent.parent.parent / "features" / "processed_features"
        self.features_dir = Path(features_dir)
        
        if not self.features_dir.exists():
            raise FileNotFoundError(
                f"Features directory not found: {self.features_dir}\n"
                "Make sure pre-computed features are available."
            )
    
    def load_features(
        self, 
        feature_name: str, 
        dataset: str = "GDPa1",
        index_by: str = "antibody_name"
    ) -> pd.DataFrame:
        """Load a feature set for a specific dataset.
        
        Args:
            feature_name: Name of the feature file (e.g., 'TAP', 'Aggrescan3D')
            dataset: Dataset name ('GDPa1' or 'heldout_test')
            index_by: Column to use as index for lookups (default: 'antibody_name')
            
        Returns:
            DataFrame with features, optionally indexed by antibody_name
            
        Raises:
            FileNotFoundError: If feature file doesn't exist
            
        Example:
            >>> loader = FeatureLoader()
            >>> tap_features = loader.load_features('TAP', dataset='GDPa1')
            >>> # Now can use as dictionary: tap_features.loc[antibody_name]
        """
        feature_path = self.features_dir / dataset / f"{feature_name}.csv"
        
        if not feature_path.exists():
            raise FileNotFoundError(
                f"Feature file not found: {feature_path}\n"
                f"Available features in {self.features_dir / dataset}: "
                f"{list((self.features_dir / dataset).glob('*.csv')) if (self.features_dir / dataset).exists() else 'directory not found'}"
            )
        
        df = pd.read_csv(feature_path)
        
        if index_by and index_by in df.columns:
            df = df.set_index(index_by)
        
        return df
    
    def get_feature_dict(
        self,
        feature_name: str,
        dataset: str = "GDPa1",
        key_col: str = "antibody_name"
    ) -> Dict[str, pd.Series]:
        """Load features as a dictionary mapping antibody_name -> features.
        
        Args:
            feature_name: Name of the feature file
            dataset: Dataset name
            key_col: Column to use as dictionary keys
            
        Returns:
            Dictionary mapping antibody_name to feature Series
            
        Example:
            >>> loader = FeatureLoader()
            >>> tap_dict = loader.get_feature_dict('TAP')
            >>> features = tap_dict['abagovomab']  # Get features for one antibody
        """
        df = self.load_features(feature_name, dataset, index_by=None)
        return {row[key_col]: row for _, row in df.iterrows()}
    
    def list_available_features(self, dataset: str = "GDPa1") -> list:
        """List all available feature files for a dataset.
        
        Args:
            dataset: Dataset name
            
        Returns:
            List of available feature names (without .csv extension)
        """
        dataset_dir = self.features_dir / dataset
        if not dataset_dir.exists():
            return []
        
        return [f.stem for f in dataset_dir.glob("*.csv")]


# Convenience function for quick access
def load_features(
    feature_name: str,
    dataset: str = "GDPa1",
    features_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Convenience function to load features with default settings.
    
    Args:
        feature_name: Name of the feature file (e.g., 'TAP', 'Aggrescan3D')
        dataset: Dataset name ('GDPa1' or 'heldout_test')
        features_dir: Optional custom features directory
        
    Returns:
        DataFrame indexed by antibody_name for easy lookup
        
    Example:
        >>> from abdev_core import load_features
        >>> tap = load_features('TAP', dataset='GDPa1')
        >>> features = tap.loc['abagovomab']  # Get features for one antibody
    """
    loader = FeatureLoader(features_dir)
    return loader.load_features(feature_name, dataset, index_by="antibody_name")

