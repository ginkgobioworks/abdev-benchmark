"""Utility functions for the antibody developability benchmark."""

from pathlib import Path
from typing import List
import pandas as pd

from .constants import FOLD_COL


def get_indices(seq_with_gaps: str) -> List[int]:
    """
    Get the aligned indices into the gapless (unaligned) sequence.
    
    Args:
        seq_with_gaps: Sequence with gap characters ('-')
        
    Returns:
        List of indices for non-gap positions
    """
    return [i for i, c in enumerate(seq_with_gaps) if c != "-"]


def extract_region(residue_scores: list, aho_indices: List[int], region_name: str) -> list:
    """
    Given a bunch of residue-level features/scores, extract a region of interest.
    
    Args:
        residue_scores: List of per-residue scores
        aho_indices: Aho numbering indices
        region_name: Name of the region to extract
        
    Returns:
        Scores for the specified region
    """
    region_options = {
        # Inclusive start, ends
        "CDRH3": (112, 138),
    }
    if region_name in region_options:
        start, end = region_options[region_name]
        return residue_scores[[i for i in aho_indices if i >= start and i <= end]]
    else:
        raise ValueError(
            f"Region {region_name} not found. Only {region_options.keys()} are supported."
        )


def load_from_tamarind(
    filepath: str,
    strip_feature_suffix: bool = True,
    only_return_features_and_names: List[str] = None,
    df_sequences: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Convert outputs from Tamarind to the format expected by the benchmark.
    
    Args:
        filepath: Path to Tamarind output CSV
        strip_feature_suffix: Whether to strip feature suffixes from column names
        only_return_features_and_names: If provided, only return these columns
        df_sequences: DataFrame with sequence information for merging
        
    Returns:
        DataFrame in benchmark format
    """
    df = pd.read_csv(filepath)
    # If sequence df exists, merge on it and use its antibody_name. Written here for IgGs (i.e. heavy and light chains)
    if df_sequences is not None:
        sequence_heavy_col, sequence_light_col = "vh_protein_sequence", "vl_protein_sequence"
        if "heavySequence" in df.columns:
            heavy_col = "heavySequence"
            light_col = "lightSequence"
            df_merged = df.merge(
                df_sequences[[sequence_heavy_col, sequence_light_col, "antibody_name"]],
                left_on=[heavy_col, light_col],
                right_on=[sequence_heavy_col, sequence_light_col],
                how="left",
            )
            df = df_merged.drop(columns=[sequence_heavy_col, sequence_light_col]).dropna(
                subset=["antibody_name"]
            )
        elif "sequence" in df.columns:
            # Only merge on heavy chain
            df_merged = df.merge(
                df_sequences[[sequence_heavy_col, "antibody_name"]],
                left_on="sequence",
                right_on=sequence_heavy_col,
                how="left",
            )
            df = df_merged.drop(columns=[sequence_heavy_col]).dropna(subset=["antibody_name"])
        else:
            print(
                f"heavySequence not found in df columns. Found columns: {df.columns}. Using Job Name to get antibody name."
            )
            df["antibody_name"] = df["Job Name"].apply(lambda x: "-".join(x.split("-")[:-1]))
    else:
        # e.g. bleselumab-g5mst: Just strip the last part after hyphen
        df["antibody_name"] = df["Job Name"].apply(lambda x: "-".join(x.split("-")[:-1]))
    if only_return_features_and_names is not None:
        columns_with_dash = [col for col in df.columns if "-" in col]
        df = df[["antibody_name"] + columns_with_dash]
    if strip_feature_suffix:
        df.columns = [col.split(" - ")[0] for col in df.columns]
    return df


def split_data_by_fold(
    data_path: Path,
    fold: int,
    output_path: Path = None,
) -> pd.DataFrame:
    """Split training data by fold for cross-validation.
    
    This function creates a training split by excluding the specified fold.
    The excluded fold can be used as validation data.
    
    Args:
        data_path: Path to input CSV with fold assignments
        fold: Fold number to hold out (0-4 for 5-fold CV)
        output_path: Optional path to save the split data
        
    Returns:
        DataFrame with training data (excluding the specified fold)
        
    Raises:
        ValueError: If data doesn't contain fold column
    """
    # Read data
    df = pd.read_csv(data_path)
    
    if FOLD_COL not in df.columns:
        raise ValueError(f"Data must contain {FOLD_COL} column for cross-validation")
    
    # Filter out the specified fold (this creates the training split)
    df_train = df[df[FOLD_COL] != fold].copy()
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_train.to_csv(output_path, index=False)
    
    return df_train

