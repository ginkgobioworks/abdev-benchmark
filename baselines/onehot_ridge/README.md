# Onehot Ridge Baseline

Ridge regression on one hot encoding of aligned heavy and light chains of antibody sequences.

## Overview

This baseline converts antibody sequences (aligned in the **AHo numbering scheme**) into **per-residue one-hot encodings**, concatenates heavy and light chains, and trains simple Ridge regression models to predict antibody developability properties.


### Why One-Hot Ridge?

- **Interpretable** — Each amino acid position contributes linearly to the property prediction.  
- **Simple & fast** — No embeddings or pretrained models are required.  
- **Alignment-aware** — The AHo numbering ensures positional correspondence across antibodies.  
- **Chain-aware** — VH and VL are concatenated directly to form a joint sequence representation.  
- **Reproducible** — Deterministic encoding, no random initialization or stochastic training.  

### Why *Not* One-Hot Ridge?

While the One-Hot Ridge baseline provides interpretability and simplicity, it has inherent limitations compared to embedding-based or nonlinear models:
 
- **Linear assumption** — Ridge regression assumes a linear relationship between features and target properties, which may not hold for complex biophysical phenomena.  
- **Limited expressiveness** — One-hot encoding treats each amino acid independently and cannot capture biochemical similarity (e.g., Leucine ≈ Isoleucine). 
- **Alignment dependency** — Requires strictly aligned sequences; performance degrades if alignment quality is poor or gaps are inconsistent.  

Despite these drawbacks, One-Hot Ridge serves as a **strong, interpretable baseline** — useful for benchmarking model behavior and verifying dataset consistency.

---

## Method

### 1. Sequence Representation

For each antibody, we use the aligned heavy and light chain sequences:

heavy_aligned_aho + light_aligned_aho → concatenated_sequence

Each concatenated sequence is tokenized at the residue level:

A A A C D E ... → ['A', 'A', 'A', 'C', 'D', 'E', ...]

Each residue position is represented as a **one-hot vector** over a 21-character vocabulary:

VOCAB = [A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y, -]

All concatenated VH+VL sequences must have identical lengths due to the AHo alignment. The resulting flattened one-hot encoding forms a fixed-length feature vector for each antibody:

Feature length = sequence_length × 21 (sequence_length = 149 (vl) + 159 (vh) = 298)

---

### 2. Feature Construction

The preprocessing pipeline:

1. Load `heavy_aligned_aho` and `light_aligned_aho` columns from the dataset.  
2. Concatenate aligned sequences (no separator).  
3. Split concatenated sequences into residue-level lists.  
4. Apply one-hot encoding with a fixed 21-character amino acid vocabulary.  
5. Flatten encoded features into 2D NumPy array `X` of shape `(n_samples, n_features)`.  

The one-hot encoder is fitted once during training and reused at inference time for consistency.

---

### 3. Ridge Regression Training

For each developability property (e.g., HIC, Tm2, Titer):

- Filter out samples with missing property values.  
- Train a separate **Ridge regression** model (`alpha = 1.0`) on the one-hot encoded features.  
- Save one model per property.  
- Store encoder and sequence length for reproducibility.

The Ridge regression minimizes the mean squared error (MSE) with L2 regularization:

\[
\text{minimize}_{w} \|y - Xw\|^2 + \alpha \|w\|^2
\]

where \( w \) are the model weights (interpretable coefficients for each amino acid–position pair).

---

### 4. Prediction

During prediction:

1. Load trained Ridge models and encoder from disk.  
2. Transform input sequences into one-hot encoded features using the stored encoder.  
3. Predict values for each property.  
4. Return a DataFrame containing:
   - `antibody_name`  
   - `heavy_aligned_aho`, `light_aligned_aho`  
   - predicted developability properties (HIC, Tm2, Titer, etc.)

---

## Usage

### Train

From the repository root:

```bash
cd baselines/onehot_ridge
pixi install

# Train on GDPa1 dataset
pixi run python -m onehot_ridge train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Predict

```bash
# Predict on training data
pixi run python -m onehot_ridge predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run
```

### Full Workflow via Orchestrator

From repository root:

```bash
pixi run all
```

This automatically discovers and runs all baselines, including ESM2 Ridge, with 5-fold cross-validation.


## Implementation Notes

- **No pretrained models**: This baseline relies solely on one-hot encoded amino acid identities — no embeddings or transformer models are used.  
- **Deterministic**: Training and prediction are fully reproducible since Ridge regression has a closed-form analytical solution.  
- **Fixed-length inputs**: All sequences must be aligned to the same length (AHo numbering ensures consistent positional correspondence).  
- **NaN handling**: Samples with missing property values are automatically excluded during training for that specific property.  
- **No internal CV**: The model trains on all available data for each run; cross-validation is managed externally by the orchestrator.  
- **Interpretable coefficients**: Each amino acid at each aligned position has a corresponding learned weight, enabling residue-level interpretation.  
- **Lightweight**: No GPU or deep model dependencies — training completes quickly even on CPUs.

---

## Features Used

- **Input features**: One-hot encoded amino acid identities (21-dimensional per residue).  
- **Sequence alignment**: Concatenated `heavy_aligned_aho` and `light_aligned_aho` sequences.  
- **Vocabulary**: 20 canonical amino acids + gap `'-'` token.  
- **Output**: Scalar developability property predictions (HIC, Tm2, Titer, etc.).  
- **No auxiliary features**: Unlike TAP or embedding-based models, this baseline uses only aligned sequence data.

---

## Performance Considerations

### Training Time

- **Fast training**: ~1–3 seconds per 1000 samples on CPU.  
- **Computation**: Simple linear regression (closed-form Ridge solution).  
- **Feature construction**: Scales linearly with sequence length and sample size.  

### Memory Usage

- **Feature matrix**: Each sequence contributes `sequence_length × 21` features.  
  - Example: 270 aligned positions → 5670 features per sequence.  
- **Model size**: Each Ridge model stores one coefficient per feature (~a few MB).  
- **No GPU memory usage**: Model is fully CPU-compatible and efficient.

---

## Model Architecture

### One-Hot Encoding

- **Input dimension**: `sequence_length × 21`  
- **Sequence composition**: Concatenated heavy and light chains (aligned AHo numbering).  
- **Encoding method**: Dense one-hot encoding using scikit-learn’s `OneHotEncoder`.  

### Ridge Regression

- **Regularization**: Alpha = 1.0 (L2 penalty).  
- **Loss function**: Mean squared error (MSE).  
- **Training**: Closed-form solution using scikit-learn’s `Ridge`.  
- **Output**: Predicted scalar property value per model.  
- **Interpretability**: Coefficients directly map amino acid identity → property contribution.

---

## References

- **AHo numbering** — Honegger & Plückthun, *Journal of Molecular Biology* (2001).  
  [https://www.sciencedirect.com/science/article/abs/pii/S0022283601946625](https://www.sciencedirect.com/science/article/abs/pii/S0022283601946625)

- **Ridge regression models for sequence–function predictions** — Hsu et al., *Nature Biotechnology* (2022).  
  [https://www.nature.com/articles/s41587-021-01146-5](https://www.nature.com/articles/s41587-021-01146-5)

- **GDPa1 dataset** — Ginkgo Bioworks, *Antibody Developability Benchmark Dataset* (2024).  
  [https://huggingface.co/datasets/ginkgo-datapoints/GDPa1](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1)

---

*This baseline emphasizes interpretability and reproducibility, providing a transparent linear foundation for comparing sequence-based developability predictors.*


























