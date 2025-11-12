# MOE-STABL Baseline

Regression models trained on Stabl-selected features from MOE (Molecular Operating Environment) molecular descriptors.

## Description

This baseline implements the Stabl feature selection framework (Discovery of sparse, reliable omic biomarkers with Stabl, J. Hédou et al., Nature Biotechnology, 2024) for sparse and interpretable feature selection. A set of regressors are then fitted to predict the antibody properties. 

Stabl is first called over all target properties to produce target-level lists of MOE features that should be later used for prediction. We loop through cross-validation folds and target properties to generate a ```stabl_feature_selection_results.pkl``` data structure that contains: 
- For every GDPa1 fold and target property, a list of Stabl-selected features
- Derivation of the optimal threshold from the FDP plot (see ```build_stabl_features.ipynb``` and **Methods** section)
- Feature-level Stability paths (see ```build_stabl_features.ipynb``` and **Methods** section)

At training time, we access the precomputed lists of features and fit a set of regressors with a randomized CV search to search for the best hyperparameters. Models, preprocessors and features selected are then stored in some ```artifact.pkl``` file. 

At inference time, we recover the trained models from the artifact file and make predictions. 

## Regression heads 

The following model have displayed highest performance: 
- HIC: Ridge 
- AC-SINS_pH7.4: LGBMRegressor
- PR_CHO: XGBRegressor 
- Titer: LGBMRegressor
- Tm2: MLP

## Results
The results reported for every target were obtained with the specified method which achieved optimal performance (in terms of average test spearman on test fold):

| Target | AC-SINS_pH7.4 |  HIC | PR_CHO | Titer | Tm2 |
| ------ | --------------| -----| ------ | ----- | --- |
| Models | LGBM | Ridge | XGB | LGBM | MLP |
| Spearman | 0.395  | 0.645 |   0.453 | 0.189 |  0.132
| $N_{features}$ | 11 | 17 | 16 | 13 | 3 |

## Requirements

- Pre-computed MOE features in `../../data/features/processed_features/`
  - `GDPa1/MOE_properties.csv` (training features)
  - `heldout_test/MOE_properties.csv` (test features)

- Stabl-selected features for every fold in ``` stabl_feature_selection_results.pkl```. 

MOE molecular descriptors computed from predicted antibody structures by Nels Thorsteinsen.

## Installation

### CLI Interface

The baseline implements a standardized CLI interface. MOE features are loaded automatically from the centralized feature store.

#### Train

```bash
pixi run python -m moe_stabl_baseline train \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run \
  [--seed 42]
```

Trains 5 optimized models (one per property) and saves to `run-dir/model_artifacts.pkl`.

#### Predict

```bash
# Training data
pixi run python -m moe_stabl_baseline predict \
  --data ../../data/GDPa1_v1.2_20250814.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/train

# Heldout test set
pixi run python -m moe_stabl_baseline predict \
  --data ../../data/heldout-set-sequences.csv \
  --run-dir ./runs/my_run \
  --out-dir ./outputs/heldout
```

Generates predictions for all samples and writes to `out-dir/predictions.csv`.

### Full Workflow via Orchestrator

From repository root:

```bash
pixi run all
```

Automatically runs all models with 5-fold cross-validation and evaluation.

## Methods 
### MOE features

MOE molecular descriptors capture structural, electrostatic, hydrophobic, geometric, and secondary structure properties computed from predicted antibody structures. The descriptor set includes ~246 features covering:
- **Structural**: radius of gyration, packing scores, surface areas
- **Electrostatic**: charge distribution, dipole moments, multipole moments
- **Hydrophobic**: patch hydrophobicity, hydrophobic moments
- **Secondary structure**: helicity, strand content
### Stabl feature selection

Stabl was first developped in the context of single cell mass cytometry to allow for reliable and interpretable feature selection in high-dimensional datasets (where $n << p$). In this competition, we thought that Stabl could bring a nice additionnal layer of interpretability to our MOE-based models. 

**Steps**: Consider a trainset (or a fold) with $n$ observations and $p$ parameters. For every hyperparameter $\lambda \in [\lambda_{min}, \lambda_{max}]$, we train a set of LASSO estimators. 

- Generate $n_{bootstraps}$ bootstraps of the train fold
- Then generate $p$ artificial (uninformative) features from actual permutated features
- On every bootstrap with $2p$ features, run LASSO with hyperparameter $\lambda$.
- Report every selected features frequency across boostraps for this choice of hyperparmeter $\lambda$: $(f_i^{\lambda})_i$
- Repeat previous steps for every $\lambda \in [\lambda_{min}, \lambda_{max}]$.

For every $\lambda \in [\lambda_{min}, \lambda_{max}]$, the previous algorithm produces a list of per-feature selection frequency $(f_j^{\lambda})_j$, known as **stability paths**. Taking the max of these selection frequencies over all $\lambda$ values, we get a set of maximum selection frequencies $(\hat{f}_j)_j$. 

For any given threshold $t \in [0,1]$, we could apply a simple selection rule: select any "real" feature $j$ whose maximum selection frequency is larger than $t$. However, choosing one such threshold a priori would be totally arbitrary. We can in fact use the artificial features to tune an optimal threshold $t_{opt}$. 

More specifically, applying the previous selection rule, gives us a set of selected features $O_t$, which contains a possibly empty set $A_t$ of artificial features. We compute:
$$FDP_{+}(t) = \frac{1+\#A_t}{ \max(\#O_t,1)}$$

The Stabl paper gives nice theoretical guarantees that this quantity can serve as an estimator of the false discovery rate in this set of selected features. Moreover, we choose $t_{opt} = \arg \min_{t \in [0,1]} FDP_{+}(t)$ and derive our final selection method:

$$\text{Choose feature across all real and artificial features only if their selection frequency is larger than }t_{opt}$$

### Regression Head Selection

For each property, Ridge, XGBoost, LightGBM, and MLP models were compared across multiple feature sets. Best configurations were selected based on 5-fold cross-validation performance.

### Prediction

Features are standardized using training set statistics. Ridge models apply linear regression; MLP models use a single hidden layer with early stopping for Tm2.

## Implementation

This baseline implements the `BaseModel` interface from `abdev_core`:

```python
from abdev_core import BaseModel, load_features

class MoeStablBaselineModel(BaseModel):
    
    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42) -> None:
        # Load MOE features from centralized store
        moe_features = load_features("MOE_properties")
        # Train 5 separate models with optimized configs
        # ...
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame: 
        # Load models and MOE features
        # Generate predictions for all 5 properties
        # ...
```
Features are managed centrally by `abdev_core`. See the [abdev_core documentation](../../libs/abdev_core/README.md) for details.

## Output

Predictions are written to `<out-dir>/predictions.csv` with columns:
- `antibody_name`
- `vh_protein_sequence`, `vl_protein_sequence`
- Predicted values for: `HIC`, `Tm2`, `Titer`, `PR_CHO`, `AC-SINS_pH7.4`

## References

- **MOE descriptors**: Nels Thorsteinsen
- **Stabl selection**: Hédou et al. (2024), "Discovery of sparse, reliable omic biomarkers with Stabl", Nature Biotechnology
- **GDPa1 dataset**: [ginkgo-datapoints/GDPa1](https://huggingface.co/datasets/ginkgo-datapoints/GDPa1)
