# Saprot_VH_VL Baseline

Ridge regression on embeddings from the **SaProt** protein language model on VH (variable heavy) and VL (variable light) sequences with two-chain encoding.

## Description

SaProt (Structure-aware Protein Language Model) generates predictions for protein properties using sequence and structure information. 

This baseline uses locally computed SaProt embeddings(SaProt_35M_AF2) to generate fixed-length embeddings for VH and VL chains using their sequences and structures, then concatenates these embeddings(VH + VL) and trains simple Ridge regression models on top to predict antibody developability properties.

The rationale behind this joint representation is same as that of the ESM2 case(no token contamination and learning from features independently).

## Method

### 1. Separate Chain Embedding

For each antibody, we embed the heavy and light chains independently:

**VH Embedding:**
```
VH_pdb  → FoldSeek 3di Descriptors → Interleaved with VH_seq → SaProt Tokenizer → Last Hidden State → Mean Pool → vh_embedding
```

**VL Embedding:**
```
VL_pdb  → FoldSeek 3di Descriptors → Interleaved with VL_seq → SaProt Tokenizer → Last Hidden State → Mean Pool → vl_embedding
```
### 2. Feature Concatenation

After generating embeddings for both chains, we concatenate them:

```
combined_embedding = np.concatenate([vh_embed, vl_embed])
```

For SaProt_35M_AF2, the embedding dimension is 480, so:
- VH embedding: 480D
- VL embedding: 480D  
- Combined: 960D


## Requirements

- VH PDB structures in `../../data/structures/AntiBodyBuilder3`
  - `GDPa1/` for train/CV and `heldout_test/` for heldout predictions, files named `{antibody_name}.pdb`

- VL PDB structures in `../../data/structures/MOE_structures`
  - `GDPa1/` for train/CV and `heldout_test/` for heldout predictions, files named `{antibody_name}.pdb`

- `foldseek` installed

Note: While SaProt embeddings can be calculated from the sequence and structure information, in the absence of structure information, it defaults to calculating embeddings with sequence information only.

### Train

From the repository root:

```bash
cd model/saprot_vh_vl
pixi install

# Train on GDPa1 dataset
pixi run python -m saprot_vh_vl train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Predict

```bash
# Predict on training data
pixi run python -m saprot_vh_vl predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Full Workflow via Orchestrator

From repository root:

```bash
pixi run all
```

This automatically discovers and runs all models, including SaProt_VH_VL, with 5-fold cross-validation.

## Results

Mean Fold Spearman Values:
AC-SINS_pH7.4 - 0.368 
HIC - 0.393 
PR_CHO - 0.428 
Titer - 0.264 
Tm2 - -0.071 

Top 10% Recall:

AC-SINS_pH7.4 - 0.100 
HIC - 0.230 
PR_CHO - 0.217 
Titer - 0.290 
Tm2 - 0.000 

## Citation

Saprot: Su J, et al. (2023). "SaProt: Protein Language Modeling with Structure-aware Vocabulary." bioRxiv.

# Code References

SaProt - https://github.com/westlake-repl/SaProt
Foldseek - https://github.com/steineggerlab/foldseek

