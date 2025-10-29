"""Feature loading and management for baseline models."""

from pathlib import Path
from typing import Optional, Dict
import pandas as pd


class FeatureLoader:
    """Centralized feature loading for baseline models.

    Provides access to pre-computed features across all datasets as dictionary lookups by antibody_name.
    By default, loads and concatenates features from all available datasets (e.g., GDPa1, heldout_test).
    Baselines can optionally use features without needing to know file paths or datasets.
    """

    def __init__(self, features_dir: Optional[Path] = None):
        """Initialize the feature loader.

        Args:
            features_dir: Path to features directory. If None, uses standard location
                         relative to this package (../../features/processed_features)
        """
        if features_dir is None:
            # Default to standard repository location
            features_dir = (
                Path(__file__).parent.parent.parent.parent.parent
                / "data"
                / "features"
                / "processed_features"
            )
        self.features_dir = Path(features_dir)

        if not self.features_dir.exists():
            raise FileNotFoundError(
                f"Features directory not found: {self.features_dir}\n"
                "Make sure pre-computed features are available."
            )

    def load_features(
        self, feature_name: str, index_by: str = "antibody_name"
    ) -> pd.DataFrame:
        """Load a feature set, concatenating across all available datasets.

        Searches for feature_name.csv in all subdirectories and concatenates them.
        This allows a single call to get all available data for a feature across
        all datasets (e.g., GDPa1, heldout_test, etc.).

        Args:
            feature_name: Name of the feature file (e.g., 'TAP', 'Aggrescan3D')
            index_by: Column to use as index for lookups (default: 'antibody_name')

        Returns:
            DataFrame with features from all datasets, optionally indexed by antibody_name

        Raises:
            FileNotFoundError: If no feature files matching the name exist

        Example:
            >>> loader = FeatureLoader()
            >>> tap_features = loader.load_features('TAP')
            >>> # Loads TAP.csv from both GDPa1/ and heldout_test/, concatenates them
            >>> features = tap_features.loc['abagovomab']
        """
        feature_files = list(self.features_dir.glob(f"*/{feature_name}.csv"))

        if not feature_files:
            available = self._get_available_features_dict()
            raise FileNotFoundError(
                f"Feature '{feature_name}' not found in any dataset under {self.features_dir}\n"
                f"Available features by dataset: {available}"
            )

        # Load and concatenate all matching feature files
        dfs = []
        for feature_path in sorted(feature_files):
            df = pd.read_csv(feature_path)
            dfs.append(df)

        df_combined = pd.concat(dfs, ignore_index=False)

        # Remove duplicate rows (same antibody_name in multiple datasets)
        # Keep first occurrence
        if index_by and index_by in df_combined.columns:
            df_combined = df_combined.drop_duplicates(subset=[index_by], keep="first")
            df_combined = df_combined.set_index(index_by)

        return df_combined

    def get_feature_dict(
        self, feature_name: str, key_col: str = "antibody_name"
    ) -> Dict[str, pd.Series]:
        """Load features as a dictionary mapping key_col values -> feature rows.

        Concatenates features from all available datasets.

        Args:
            feature_name: Name of the feature file
            key_col: Column to use as dictionary keys (default: 'antibody_name')

        Returns:
            Dictionary mapping key_col values to feature Series

        Example:
            >>> loader = FeatureLoader()
            >>> tap_dict = loader.get_feature_dict('TAP')
            >>> features = tap_dict['abagovomab']  # Get features for one antibody
        """
        df = self.load_features(feature_name, index_by=None)
        return {row[key_col]: row for _, row in df.iterrows()}

    def list_available_features(self) -> list:
        """List all available feature files across all datasets.

        Returns:
            List of available feature names (without .csv extension)
        """
        if not self.features_dir.exists():
            return []

        feature_names = set()
        for csv_file in self.features_dir.glob("*/*.csv"):
            feature_names.add(csv_file.stem)

        return sorted(list(feature_names))

    def _get_available_features_dict(self) -> Dict[str, list]:
        """Internal method: Get available features organized by dataset.

        Returns:
            Dictionary mapping dataset name to list of available feature names
        """
        features_by_dataset = {}
        if not self.features_dir.exists():
            return features_by_dataset

        for dataset_dir in self.features_dir.iterdir():
            if dataset_dir.is_dir():
                features = [f.stem for f in dataset_dir.glob("*.csv")]
                if features:
                    features_by_dataset[dataset_dir.name] = sorted(features)

        return features_by_dataset


# Convenience function for quick access
def load_features(
    feature_name: str, features_dir: Optional[Path] = None
) -> pd.DataFrame:
    """Convenience function to load features from all available datasets.

    Args:
        feature_name: Name of the feature file (e.g., 'TAP', 'Aggrescan3D')
        features_dir: Optional custom features directory

    Returns:
        DataFrame indexed by antibody_name for easy lookup, containing data
        from all available datasets (e.g., GDPa1, heldout_test)

    Example:
        >>> from abdev_core import load_features
        >>> tap = load_features('TAP')
        >>> features = tap.loc['abagovomab']  # Get features for one antibody
    """
    loader = FeatureLoader(features_dir)
    return loader.load_features(feature_name, index_by="antibody_name")
