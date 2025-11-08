# AbLang2 + ElasticNet - Baseline (`ablang2_elastic_net`)

Sequence-based antibody developability predictions using AbLang2 paired embeddings and ElasticNet regression.

What the model does:
- Embeds paired antibody chains (VH|VL) 
- Mean-pools token embeddings
- Standardizes and reduces dimensionality via PCA
- Fits one ElasticNetCV regressor per property in `PROPERTY_LIST`

---

## Requirements

- Managed via `pixi` (same as other models in this repo)
- Loads AbLang2 lazily (uses CUDA if available)

---

## Data Schema

**Required columns:**
- `vh_protein_sequence`
- `vl_protein_sequence`
- `antibody_name` 

**Targets:**  
Any subset of the properties in `PROPERTY_LIST` that exist in your input CSV.

---

## Quickstart

```bash
# 1) Install environment
cd abdev-benchmark/models/ablang2_elastic_net
pixi install

# 2) Train model
pixi run python -m ablang2_elastic_net train \
  --data  ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run --seed 42

# 3) Predict on training data
pixi run python -m ablang2_elastic_net predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/train

# 4) Predict on heldout set
pixi run python -m ablang2_elastic_net predict \
  --data ../../data/heldout-set-sequences.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/heldout
