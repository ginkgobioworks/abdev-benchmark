# PolyXpert

PolyXpert is a fine-tuned ESM-2 model for predicting polyreactivity of therapeutic antibody candidates using scFv sequence data.

## Description

PolyXpert uses a transformer-based architecture (fine-tuned ESM-2) to predict antibody polyreactivity from heavy (VH) and light (VL) chain sequences. The model requires only sequence data (no structural information) and achieves 0.9672 AUC on held-out test set.

The model processes paired VH/VL sequences and outputs a probability score indicating polyreactivity risk.

## Requirements

- python = ">=3.11.*"
- numpy = ">=1.24"
- pandas = "1.5.3"
- torch = "1.13.1"
- tqdm = "4.64.1"
- transformers = "4.26.1"
- datasets = "2.9.0"

## Installation

```bash
cd models/polyxpert
pixi install
```

**Note**: Model weights (~400MB) are automatically downloaded to cache on first use. The model uses `~/.cache/polyxpert/` for caching. An internet connection is required for the initial download.

## Usage

### Train

From the repository root:

```bash
cd models/polyxpert
pixi install

# Train (no-op for pre-trained model)
# Model weights will be automatically downloaded on first use
pixi run python -m polyxpert train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Predict

```bash
# Predict on training data
pixi run python -m polyxpert predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run  --out-dir ./output
```

### Full Workflow via Orchestrator

From repository root:

```bash
pixi run all
```

This automatically discovers and runs all models, including PolyXpert, with 5-fold cross-validation.

## Outputs

The model outputs a PR_CHO (polyreactivity) score ranging from 0 to 1. Higher scores indicate higher polyreactivity (worse developability).

## Model Architecture

### Fine-tuned ESM-2

- **Base model**: ESM-2 protein language model
- **Fine-tuning**: Trained on antibody polyreactivity data
- **Input**: Space-separated amino acid sequences for VH and VL
- **Output**: Binary classification (polyreactive vs. non-polyreactive)
- **Batch processing**: Processes sequences in batches of 16
- **Max sequence length**: 512 tokens

## Implementation Notes

- **Pre-trained model**: No training required; uses pre-trained fine-tuned weights
- **Automatic download**: Model weights are automatically downloaded to cache on first use
- **Cache location**: Uses `~/.cache/polyxpert/` for model weights
- **GPU support**: Automatically detects and uses CUDA if available; falls back to CPU
- **Batch processing**: Efficient batch processing with DataLoader
- **Non-standard amino acids**: Replaces O, B, U, Z with X
- **No internal CV**: Training is a no-op; the orchestrator handles cross-validation

## Property Predicted

- **PR_CHO**: Polyreactivity CHO (lower is better)

## Performance

- **AUC**: 0.9672 on held-out test set
- **Input requirements**: VH and VL Fv sequences only

## Important Note on the PolyXpert Model

The **PolyXpert** model provides valuable predictions for antibody polyreactivity, but its inclusion in this benchmark requires specific contextual interpretation regarding comparability.

- **Training Data Discrepancy**: PolyXpert was trained on polyreactivity labels derived from **four different reagents**. In contrast, this benchmark evaluates predictions against a single, specific **PR_CHO** value. This difference in target variables is a critical factor when comparing its performance directly against models trained specifically on the PR_CHO metric.

- **Inherent Challenge**: As discussed in our accompanying paper (https://doi.org/10.1016/j.jpha.2025.101448), achieving consistent polyreactivity predictions across diverse antibody types and assay reagents remains a significant challenge in the field.

- **Interpretation Guidance**:
    - **Technical Suitability**: The model is technically suitable for inclusion and provides a robust sequence-based prediction of polyreactivity risk.
    - **Contextual Comparison**: However, its performance scores (e.g., Spearman ρ) in the benchmark leaderboard should be interpreted with the understanding that it was not optimized specifically for the PR_CHO endpoint. Direct, quantitative comparisons with models trained end-to-end on the benchmark's PR_CHO data may not fully reflect its utility for predicting general polyreactivity.

In summary, while PolyXpert is a powerful tool for predicting antibody polyreactivity, its ranking relative to other models on the PR_CHO metric should be contextualized within these differences in training objectives and the inherent challenges of the prediction task.  

## Reference

Yuwei Zhou, Haoxiang Tang, Changchun Wu, Zixuan Zhang, Jinyi Wei, Rong Gong, Samarappuli Mudiyanselage Savini Gunarathne, Changcheng Xiang, Jian Huang. *Enhancing polyreactivity prediction of preclinical antibodies through fine-tuned protein language models.* Journal of Pharmaceutical Analysis, 2025, 101448. DOI: 10.1016/j.jpha.2025.101448

