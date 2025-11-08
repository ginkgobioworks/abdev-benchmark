# p-IgGen Baseline

Ridge regression on embeddings from the **p-IgGen** foundation model.

## Overview

This baseline uses the p-IgGen protein language model to generate fixed-length embeddings for paired VH/VL antibody sequences, then trains simple Ridge regression models on top of these embeddings to predict antibody developability properties.

### Why p-IgGen?

- **Pre-trained on antibody sequences**: p-IgGen is specifically trained on paired VH/VL antibody data, capturing domain-specific structure-function relationships
- **Transfer learning**: Leverages learned representations from large-scale pre-training to generalize across diverse antibody sequences
- **Single embedding per antibody**: Mean-pooled representation captures global sequence properties relevant to developability
- **Minimal fine-tuning**: Ridge regression is simple, fast, and provides interpretable baseline performance

## Method

### 1. Sequence Representation

For each antibody, we create a paired sequence representation:

```
"1" + SPACE_SEPARATED_VH + " " + SPACE_SEPARATED_VL + "2"
```

Where:
- `"1"` marks sequence start
- `" "` separates amino acids
- `"2"` marks sequence end

Example:
```
"1Q V K L Q E S G A E L A R ... D I Q M T ... 2"
```

### 2. Embedding Generation

We use the p-IgGen model (`ollieturnbull/p-IgGen`) to generate hidden states for the paired sequences:

- Tokenize paired sequences
- Forward pass through p-IgGen encoder
- Extract final hidden state for all tokens
- **Mean pool across token dimension** to get a fixed-length vector (typically ~768D)

Embeddings are generated in batches (batch size 16) for efficiency.

### 3. Ridge Regression Training

For each antibody property (HIC, Tm2, Titer, etc.):

- Extract embeddings and labels for all samples with valid measurements
- Train a Ridge regression model on embeddings → property values
- Save trained models for prediction time

### 4. Prediction

At prediction time:
- Generate embeddings for new samples (same process)
- Use trained Ridge models to predict each property
- Return predictions as CSV

## Usage

### Train

From the repository root:

```bash
cd model/piggen
pixi install

# Train on GDPa1 dataset
pixi run python -m piggen train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Predict

```bash
# Predict on training data
pixi run python -m piggen predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Full Workflow via Orchestrator

From repository root:

```bash
pixi run all
```

This automatically discovers and runs all models, including p-IgGen, with 5-fold cross-validation.

## Implementation Notes

- **Lazy loading**: The transformer model is loaded only on first use (train or predict) to save memory
- **GPU support**: Automatically detects and uses CUDA if available; falls back to CPU
- **Batch processing**: Embeddings are generated in batches to balance memory and speed
- **NaN handling**: Samples with missing property values are automatically excluded during training for that property
- **No internal CV**: Training data is not split internally; the orchestrator handles cross-validation

## Features Used

- **p-IgGen embeddings**: 768-dimensional vectors from the p-IgGen hidden layer
- **No auxiliary features**: Unlike TAP models, this approach relies solely on sequence information encoded in the embeddings

## Performance Considerations

### Training Time

- ~10-15 seconds per 100 samples (CPU) / ~1-2 seconds (GPU) for embedding generation
- Additional ~1 second for Ridge regression training

### Memory Usage

- Model: ~1.5GB (fp32)
- Embeddings: ~240MB for full GDPa1 dataset (768D × ~400K samples)

## References

- **p-IgGen**: Ollie Turnbull et al., preprint on antibody sequence modeling
- **Model checkpoint**: [ollieturnbull/p-IgGen](https://huggingface.co/ollieturnbull/p-IgGen)
- **GDPa1 dataset**: [ginkgo-datapoints/GDPa1](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1)
