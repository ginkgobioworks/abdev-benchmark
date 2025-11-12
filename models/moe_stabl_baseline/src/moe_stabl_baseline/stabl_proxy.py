"""
STABL Proxy Feature Selection
-----------------------------

This module implements the STABL (Stable Feature Selection via Bootstrapping and Artificial Features)
algorithm for robust feature selection using Lasso regressions on bootstrapped datasets.

Algorithmic logic identical to the reference notebook version.
Optimized for speed (vectorized bootstrapping, parallel Lasso fits) but without changing semantics.

Author: <your name>
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------
def stabl_preprocessor(variance_threshold: float = 0.0, scale: bool = True) -> Pipeline:
    """Build preprocessing pipeline without imputation."""
    steps = []
    if variance_threshold and variance_threshold > 0.0:
        steps.append(("var", VarianceThreshold(threshold=variance_threshold)))
    if scale:
        steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    return Pipeline(steps)

# ---------------------------------------------------------------------
# Core bootstrapping and artificial feature generation
# ---------------------------------------------------------------------
def generate_bootstrap_from_dataset(df_main_dataset: pd.DataFrame,
                                    df_target: pd.DataFrame,
                                    n_bootstraps: int):
    """Generate concatenated bootstrapped dataset (rows only)."""
    df_merged = pd.concat(
        [df_main_dataset.reset_index(drop=True),
         df_target.reset_index(drop=True)], axis=1
    )

    n = len(df_merged)
    # Each row in idx is a bootstrap of length n
    idx = np.random.randint(0, n, size=(n_bootstraps, n))
    bootstraps = [df_merged.iloc[idx[i]] for i in range(n_bootstraps)]
    df_bootstrapped = pd.concat(bootstraps, ignore_index=True)

    df_target_bootstrapped = df_bootstrapped[df_target.columns]
    df_bootstrapped = df_bootstrapped.drop(columns=df_target.columns)
    return df_bootstrapped, df_target_bootstrapped


def generate_artificial_permuted_features(df_bootstrapped: pd.DataFrame,
                                          fraction: float = 1.0) -> pd.DataFrame:
    """
    Add artificial permuted features to the bootstrapped dataset.
    Each artificial column is a row-permuted copy of a real feature.
    """
    cols = df_bootstrapped.columns
    n_cols = len(cols)
    k = int(np.ceil(fraction * n_cols))
    sampled_cols = np.random.choice(cols, size=k, replace=False)
    d_old_sampled = df_bootstrapped[sampled_cols]

    # Shuffle rows (permute each selected column)
    perm_idx = np.random.permutation(len(df_bootstrapped))
    df_artificial_shuffled = d_old_sampled.iloc[perm_idx].reset_index(drop=True)
    df_artificial_shuffled.columns = [f"{c}_artificial" for c in df_artificial_shuffled.columns]

    return pd.concat([df_bootstrapped.reset_index(drop=True),
                      df_artificial_shuffled], axis=1)


# ---------------------------------------------------------------------
# Lasso fitting and per-bootstrap feature selection
# ---------------------------------------------------------------------
def fit_lasso_on_one_bootstrap(Xb: np.ndarray,
                               yb: np.ndarray,
                               columns: np.ndarray,
                               alpha_param: float,
                               max_iter: int = 1000):
    """Fit Lasso on one bootstrap and return selected feature names."""
    model = Lasso(alpha=alpha_param, max_iter=max_iter)
    model.fit(Xb, yb)
    return columns[model.coef_ != 0]


def train_stabl_proxy_one_alpha(df_main_dataset: pd.DataFrame,
                                df_target: pd.DataFrame,
                                n_bootstraps: int,
                                alpha_param: float,
                                fraction: float = 1.0,
                                max_iter: int = 1000,
                                n_jobs: int = -1):
    """
    Train STABL proxy for one alpha parameter.
    Performs Lasso selection across n_bootstraps on the augmented dataset.
    """
    df_bootstrapped, df_target_bootstrapped = generate_bootstrap_from_dataset(
        df_main_dataset, df_target, n_bootstraps
    )
    df_artificial = generate_artificial_permuted_features(df_bootstrapped, fraction)
    n_main = df_main_dataset.shape[0]

    X_all = df_artificial.values
    y_all = df_target_bootstrapped.values.ravel()
    cols = np.array(df_artificial.columns)

    def one_bootstrap(i: int):
        start, end = n_main * i, n_main * (i + 1)
        Xb = X_all[start:end]
        yb = y_all[start:end]
        return fit_lasso_on_one_bootstrap(Xb, yb, cols, alpha_param, max_iter)

    results = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(one_bootstrap)(i) for i in range(n_bootstraps)
    )

    return {f"bootstrap_{i}": res for i, res in enumerate(results)}


# ---------------------------------------------------------------------
# Frequency aggregation and FDP computation
# ---------------------------------------------------------------------
def compute_fdr_and_feature_freq(all_selected_features: dict,
                                 n_bootstraps: int) -> dict:
    """Compute selection frequency per feature."""
    feature_occ = {}
    for features in all_selected_features.values():
        for f in features:
            feature_occ[f] = feature_occ.get(f, 0) + 1
    return {f: c / n_bootstraps for f, c in feature_occ.items()}


def merge_all_feature_freq(every_feature_freq: dict) -> dict:
    """Aggregate per-alpha frequency dictionaries."""
    merged = {}
    for alpha, freq_dict in every_feature_freq.items():
        inva = 1 / alpha
        for f, v in freq_dict.items():
            d = merged.setdefault(f, {"1/alpha": [], "freq": []})
            d["1/alpha"].append(inva)
            d["freq"].append(v)
    return merged


def get_fdp_curve(merged_feature_freq: dict, n: int = 1000):
    """
    Compute the empirical FDP(t) curve.
    FDP(t) = (1 + # artificial features >= t) / max(# total features >= t, 1)
    """
    t = np.linspace(0, 1, n)
    max_freqs = {f: np.max(v["freq"]) for f, v in merged_feature_freq.items()}

    arr_freq_all = np.array(list(max_freqs.values()))
    arr_freq_artif = np.array(
        [v for f, v in max_freqs.items() if "artificial" in f],
        dtype=float
    )

    selected_all = arr_freq_all[:, None] >= t
    count_all = selected_all.sum(axis=0)
    if len(arr_freq_artif) > 0:
        selected_art = arr_freq_artif[:, None] >= t
        count_art = selected_art.sum(axis=0)
    else:
        count_art = np.zeros_like(count_all)

    fdp = (1 + count_art) / np.maximum(count_all, 1)
    return fdp, t


# ---------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------
def select_stabl_features(df_x_train: pd.DataFrame,
                          df_y_train: pd.DataFrame,
                          n_bootstraps: int = 500,
                          fraction: float = 1.0,
                          alpha_grid: np.ndarray = np.linspace(0.01, 1.0, 100),
                          max_iter: int = 1000,
                          fdp_grid_size: int = 1000,
                          n_jobs: int = -1 # no seed
                          ) -> dict:
    """
    Run full STABL proxy feature selection.
    Returns selected features, merged frequency table, and FDP info.
    """ 
    every_feature_freq = {}
    for alpha in tqdm(alpha_grid, desc="STABL α-grid"):
        all_selected_features = train_stabl_proxy_one_alpha(
            df_x_train, df_y_train, n_bootstraps, alpha,
            fraction=fraction, max_iter=max_iter, n_jobs=n_jobs
        )
        feature_freq = compute_fdr_and_feature_freq(all_selected_features, n_bootstraps)
        every_feature_freq[alpha] = feature_freq

    merged_feature_freq = merge_all_feature_freq(every_feature_freq)
    fdp, t = get_fdp_curve(merged_feature_freq, n=fdp_grid_size)
    thr_opt = t[np.argmin(fdp)]

    selected_features = [
        f for f, v in merged_feature_freq.items()
        if np.max(v["freq"]) >= thr_opt and "artificial" not in f
    ]

    return {
        "selected_features": selected_features,
        "merged_freq": merged_feature_freq,
        "fdp_info": {"fdp": fdp, "t": t, "thr_opt": thr_opt},
    }
