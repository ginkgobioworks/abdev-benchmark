# SSH2
The SSH2.0 is a predictor that can be used to predict Hydrophobic interaction of monoclonal antibodies using sequences.

# Your Model Name

SSH2 - CKSAAGP + SVM ensemble model for antibodies hydrophobic interaction prediction

## Description

This model first concatenates the heavy (VH) and light (VL) chains, then extracts CKSAAGP 
features from the combined sequence using iFeature. The approach utilizes three distinct 
feature groups, each processed by a separate pre-trained model. The individual predictions 
are then integrated to produce the final hydrophobicity score.

LIBSVM (Chang and Lin., 2011) was employed to construct the SVM sub-models. 


## Requirements

python = ">=3.7.*"
pandas = ">=1.1.4"
numpy = ">=1.18"

## Usage

```

pixi run python -m ssh2 train --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./runs/my_run
pixi run python -m ssh2 predict --data ../../data/GDPa1_v1.2_20250814.csv --run-dir ./runs/my_run --out-dir ./outputs

```

### outputs

The model outputs a hydrophobicity score ranging from 0 to 1.
Scores above 0.5(default) indicate high hydrophobicity, while scores below 0.5 indicate low hydrophobicity.


## Important Note on the SSH2 Model

The **SSH2** model included in this benchmark was originally developed for a *classification task* (distinguishing between "good" and "poor" developability profiles), not for direct *regression* (predicting continuous property values).

- **Model Output**: The model outputs a probability score (ranging from 0 to 1) representing the likelihood of a sequence belonging to the "good" developability class.
- **Benchmark Interpretation**: Within the context of this benchmark, this probability score is treated as a **relative metric** for comparison across antibodies in the test set. A higher score indicates a predicted better developability profile.
- **Caution in Comparison**: It is important to be aware that this probability score is **not directly equivalent** to the predicted values from regression-based models (e.g., Ridge, XGBoost) for properties like HIC or Titer. The scores should be interpreted as ranks for relative comparison rather than absolute property values.

**In summary**: While the SSH2 model's predictions are useful for ranking antibodies by their relative developability potential, direct numerical comparison of its output scores against the raw value predictions from other models in the benchmark may not be appropriate. 


## Reference

Zhou Y, Xie S, Yang Y, et al. SSH2.0: A Better Tool for Predicting the Hydrophobic Interaction Risk of Monoclonal Antibody. Front Genet. 2022;13:842127. doi:10.3389/fgene.2022.842127
