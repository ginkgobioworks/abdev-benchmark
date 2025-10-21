"""Utility functions for the antibody developability benchmark."""

from typing import List
import pandas as pd


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

