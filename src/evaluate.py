import os
from collections import defaultdict
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

from utils import ROOT_DIR, PROPERTY_LIST, ASSAY_HIGHER_IS_BETTER

def recall_at_k(y_true: np.ndarray, y_pred: np.ndarray, frac: float = 0.1) -> float:
    """Calculate recall (TP)/(TP+FN) for top fraction of true values.

    A recall of 1 would mean that the top fraction of true values are also the top fraction of predicted values.
    There is no penalty for ranking the top k differently.

    Args:
        y_true (np.ndarray): true values with shape (num_data,)
        y_pred (np.ndarray): predicted values with shape (num_data,)
        frac (float, optional): fraction of data points to consider as the top. Defaults to 0.1.

    Returns:
        float: recall at top k of data
    """
    top_k = int(len(y_true) * frac)
    y_true, y_pred = np.array(y_true).flatten(), np.array(y_pred).flatten()
    true_top_k = np.argsort(y_true)[-1 * top_k :]
    predicted_top_k = np.argsort(y_pred)[-1 * top_k :]

    return (
        len(
            set(list(true_top_k.flatten())).intersection(
                set(list(predicted_top_k.flatten()))
            )
        )
        / top_k
    )

def evaluate(predictions_series: pd.Series, target_series: pd.Series, assay_col: str) -> dict[str, float]:
    results_dict = {"spearman": spearmanr(predictions_series, target_series, nan_policy="omit").correlation}
    # Top 10% recall
    y_true = target_series.values
    y_pred = predictions_series.values
    if not ASSAY_HIGHER_IS_BETTER[assay_col]:
        y_true = -1 * y_true
        y_pred = -1 * y_pred
    results_dict["top_10_recall"] = recall_at_k(y_true=y_true, y_pred=y_pred, frac=0.1)
    return results_dict

def evaluate_cross_validation(predictions_series: pd.Series, target_series: pd.Series, folds_series: pd.Series, assay_col: str) -> dict[str, float]:
    # Run evaluate in a cross-validation loop
    results_dict = defaultdict(list)
    if folds_series.nunique() != 5:
        raise ValueError(f"Expected 5 folds, got {folds_series.nunique()}")
    for fold in folds_series.unique():
        predictions_series_fold = predictions_series[folds_series == fold]
        target_series_fold = target_series[folds_series == fold]
        results = evaluate(predictions_series_fold, target_series_fold, assay_col)
        # Update the results_dict with the results for this fold
        for key, value in results.items():
            results_dict[key].append(value)
    # Calculate the mean of the results for each key (could also add std dev later)
    for key, values in results_dict.items():
        results_dict[key] = np.mean(values)
    return results_dict

def evaluate_model(preds_path, target_path, results_path, model_name, dataset_name=None):
    """
    Evaluates a single model, where the predictions dataframe has columns named by property.
    eg. my_model.csv has columns antibody_name, HIC, Tm2
    """
    predictions_df = pd.read_csv(preds_path)
    target_df = pd.read_csv(target_path)
    properties_in_preds = [col for col in predictions_df.columns if col in PROPERTY_LIST]
    FOLD_COL = "hierarchical_cluster_IgG_isotype_stratified_fold"
    df_merged = pd.merge(target_df[["antibody_name", FOLD_COL] + PROPERTY_LIST], predictions_df[["antibody_name"] + properties_in_preds], on="antibody_name", how="left", suffixes=("_true", "_pred"))
    results_list = []
    for assay_col in properties_in_preds:
            if dataset_name == "GDPa1_cross_validation":
                results = evaluate_cross_validation(df_merged[assay_col+"_pred"], df_merged[assay_col+"_true"], df_merged[FOLD_COL], assay_col)
            else:
                results = evaluate(df_merged[assay_col+"_pred"], df_merged[assay_col+"_true"], assay_col)
            # Later pivot this dataset to get metrics per dataset
            results["dataset"] = dataset_name
            results["assay"] = assay_col
            results["model"] = model_name
            results_list.append(results)
    
    return results_list


# For now, run with `python -m src.evaluate` to get the import to work
if __name__ == "__main__":
    DATASET_NAMES = ["GDPa1", "GDPa1_cross_validation"]
    results_list = []
    # Evaluate these models on the full dataset
    for dataset_name in DATASET_NAMES:
        predictions_path = f"{ROOT_DIR}/predictions/{dataset_name}/"
        for model_dir in os.listdir(predictions_path):
            for model_name in os.listdir(f"{predictions_path}/{model_dir}"):
                model_name = model_name.replace(".csv", "")
                # One file per model
                df_preds = pd.read_csv(f"{predictions_path}/{model_dir}/{model_name}.csv")
                print(f"Evaluating {model_name}")
                model_results = evaluate_model(
                    preds_path=f"{predictions_path}/{model_dir}/{model_name}.csv",
                    target_path=f"{ROOT_DIR}/data/GDPa1_v1.2_20250814.csv",
                    results_path=f"{ROOT_DIR}/results/{dataset_name}/{model_name}.csv",
                    model_name=model_name,
                    dataset_name=dataset_name,
                )
                results_list.extend(model_results)
    
    # Take all these individual results and concat them into a single file
    df_metrics_all = pd.DataFrame(results_list)
    df_metrics_all["spearman_abs"] = df_metrics_all["spearman"].abs()
    # Optional: reset index if you want "assay" and "model" as columns
    df_metrics_all = df_metrics_all.sort_values(by="spearman_abs", ascending=False)
    df_metrics_all.to_csv(f"{ROOT_DIR}/results/metrics_all.csv", index=False)
