"""DeepSP feature generator for computing spatial properties from antibody sequences."""

import os
import tempfile
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.models import model_from_json


class DeepSPFeatureGenerator:
    """Generate DeepSP spatial features from antibody sequences.
    
    This class wraps the DeepSP model to compute 30 spatial properties
    (SAP_pos, SCM_neg, SCM_pos for 10 regions each) from VH/VL sequences.
    """
    
    # IMGT position mappings for sequence alignment
    H_INCLUSION_LIST = [str(i) for i in range(1, 111)] + \
                       ['111', '111A', '111B', '111C', '111D', '111E', '111F', '111G', '111H'] + \
                       ['112I', '112H', '112G', '112F', '112E', '112D', '112C', '112B', '112A', '112'] + \
                       [str(i) for i in range(113, 129)]
    
    L_INCLUSION_LIST = [str(i) for i in range(1, 128)]
    
    # Create position dictionaries for IMGT alignment
    H_DICT = {pos: idx for idx, pos in enumerate(H_INCLUSION_LIST)}
    L_DICT = {pos: idx for idx, pos in enumerate(L_INCLUSION_LIST)}
    
    FEATURE_NAMES = [
        'SAP_pos_CDRH1', 'SAP_pos_CDRH2', 'SAP_pos_CDRH3', 'SAP_pos_CDRL1', 'SAP_pos_CDRL2', 'SAP_pos_CDRL3',
        'SAP_pos_CDR', 'SAP_pos_Hv', 'SAP_pos_Lv', 'SAP_pos_Fv',
        'SCM_neg_CDRH1', 'SCM_neg_CDRH2', 'SCM_neg_CDRH3', 'SCM_neg_CDRL1', 'SCM_neg_CDRL2', 'SCM_neg_CDRL3',
        'SCM_neg_CDR', 'SCM_neg_Hv', 'SCM_neg_Lv', 'SCM_neg_Fv',
        'SCM_pos_CDRH1', 'SCM_pos_CDRH2', 'SCM_pos_CDRH3', 'SCM_pos_CDRL1', 'SCM_pos_CDRL2', 'SCM_pos_CDRL3',
        'SCM_pos_CDR', 'SCM_pos_Hv', 'SCM_pos_Lv', 'SCM_pos_Fv'
    ]
    
    def __init__(self, model_weights_dir: Path):
        """Initialize the feature generator.
        
        Args:
            model_weights_dir: Directory containing DeepSP model weights
        """
        self.model_weights_dir = Path(model_weights_dir)
        self.models = {}
        
    def _load_models(self) -> None:
        """Lazy load the three DeepSP Conv1D models."""
        if self.models:
            return
            
        model_names = ['SAPpos', 'SCMneg', 'SCMpos']
        
        for name in model_names:
            json_path = self.model_weights_dir / f"Conv1D_regression{name}.json"
            weights_path = self.model_weights_dir / f"Conv1D_regression_{name}.h5"
            
            with open(json_path, 'r') as f:
                model_json = f.read()
            
            model = model_from_json(model_json)
            model.load_weights(str(weights_path))
            model.compile(optimizer='adam', loss='mae', metrics=['mae'])
            
            self.models[name] = model
    
    def _align_sequences(
        self, 
        names: List[str], 
        heavy_seqs: List[str], 
        light_seqs: List[str]
    ) -> pd.DataFrame:
        """Align sequences using ANARCI and return aligned sequences.
        
        Args:
            names: List of antibody names
            heavy_seqs: List of VH sequences
            light_seqs: List of VL sequences
            
        Returns:
            DataFrame with aligned sequences in IMGT format
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Write FASTA files for heavy and light chains
            heavy_fasta = tmpdir / "seq_H.fasta"
            light_fasta = tmpdir / "seq_L.fasta"
            
            with open(heavy_fasta, "w") as f:
                for name, seq in zip(names, heavy_seqs):
                    record = SeqRecord(Seq(seq), id=name, name="", description="")
                    SeqIO.write(record, f, "fasta")
            
            with open(light_fasta, "w") as f:
                for name, seq in zip(names, light_seqs):
                    record = SeqRecord(Seq(seq), id=name, name="", description="")
                    SeqIO.write(record, f, "fasta")
            
            # Run ANARCI alignment
            heavy_out = tmpdir / "seq_aligned"
            light_out = tmpdir / "seq_aligned"
            
            os.system(f'ANARCI -i {heavy_fasta} -o {heavy_out} -s imgt -r heavy --csv > /dev/null 2>&1')
            os.system(f'ANARCI -i {light_fasta} -o {light_out} -s imgt -r light --csv > /dev/null 2>&1')
            
            # Read aligned sequences
            h_aligned = pd.read_csv(tmpdir / "seq_aligned_H.csv")
            l_aligned = pd.read_csv(tmpdir / "seq_aligned_KL.csv")
            
            return self._preprocess_aligned_sequences(h_aligned, l_aligned)
    
    def _preprocess_aligned_sequences(
        self, 
        h_aligned: pd.DataFrame, 
        l_aligned: pd.DataFrame
    ) -> List[str]:
        """Preprocess aligned sequences into fixed-length format.
        
        Args:
            h_aligned: Aligned heavy chain sequences from ANARCI
            l_aligned: Aligned light chain sequences from ANARCI
            
        Returns:
            List of concatenated aligned sequences (H+L)
        """
        n_mabs = len(h_aligned)
        aligned_sequences = []
        
        for i in range(n_mabs):
            # Initialize fixed-length arrays with gaps
            h_tmp = ['-'] * 145
            l_tmp = ['-'] * 127
            
            # Fill in heavy chain positions
            for col in h_aligned.columns:
                if col in self.H_INCLUSION_LIST:
                    h_tmp[self.H_DICT[col]] = h_aligned.iloc[i][col]
            
            # Fill in light chain positions
            for col in l_aligned.columns:
                if col in self.L_INCLUSION_LIST:
                    l_tmp[self.L_DICT[col]] = l_aligned.iloc[i][col]
            
            # Concatenate heavy + light
            aligned_sequences.append(''.join(h_tmp + l_tmp))
        
        return aligned_sequences
    
    def _one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode an aligned sequence.
        
        Args:
            sequence: Aligned amino acid sequence
            
        Returns:
            One-hot encoded matrix of shape (seq_len, 21)
        """
        aa_dict = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20
        }
        
        x = np.zeros((len(aa_dict), len(sequence)))
        x[[aa_dict[c] for c in sequence], range(len(sequence))] = 1
        
        return x
    
    def generate_features(
        self,
        names: List[str],
        vh_sequences: List[str],
        vl_sequences: List[str]
    ) -> pd.DataFrame:
        """Generate DeepSP features for antibody sequences.
        
        Args:
            names: List of antibody names
            vh_sequences: List of VH protein sequences
            vl_sequences: List of VL protein sequences
            
        Returns:
            DataFrame with antibody_name and 30 DeepSP feature columns
        """
        # Load models if not already loaded
        self._load_models()
        
        # Align sequences using ANARCI
        aligned_seqs = self._align_sequences(names, vh_sequences, vl_sequences)
        
        # One-hot encode all sequences
        X = [self._one_hot_encode(seq) for seq in aligned_seqs]
        X = np.transpose(np.asarray(X), (0, 2, 1))
        
        # Predict features using the three models
        sap_pos = self.models['SAPpos'].predict(X, verbose=0)
        scm_neg = self.models['SCMneg'].predict(X, verbose=0)
        scm_pos = self.models['SCMpos'].predict(X, verbose=0)
        
        # Combine predictions into DataFrame
        features_df = pd.DataFrame({
            'antibody_name': names,
            **{self.FEATURE_NAMES[i]: sap_pos[:, i] for i in range(10)},
            **{self.FEATURE_NAMES[10 + i]: scm_neg[:, i] for i in range(10)},
            **{self.FEATURE_NAMES[20 + i]: scm_pos[:, i] for i in range(10)}
        })
        
        return features_df

