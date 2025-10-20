import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import PredefinedSplit
from utils import ROOT_DIR, PROPERTY_LIST, GDPA1_PATH, HELDOUT_PATH

FEATURE_NAMES = ["SFvCSP", "PSH", "PPC", "PNC", "CDR Length"]
FOLD_COL = "hierarchical_cluster_IgG_isotype_stratified_fold"

def get_cross_validation_predictions(df: pd.DataFrame, property_name: str, df_heldout: pd.DataFrame) -> None:
    """Run a quick cross-validation loop, should be quick with a small dataset."""
    model = Ridge()
    df_out = df.copy()
    splits = PredefinedSplit(test_fold=df[FOLD_COL])
    for i, (train_index, test_index) in enumerate(splits.split()):
        subset = df.iloc[train_index].dropna(subset=[property_name])
        model.fit(subset[FEATURE_NAMES], subset[property_name])
        df_out.loc[test_index, "pred_cv"] = model.predict(df.iloc[test_index][FEATURE_NAMES])
    
    # Re-fit whole model to get overall predictions
    model.fit(df[FEATURE_NAMES], df[property_name])
    df_heldout["pred_heldout"] = model.predict(df_heldout[FEATURE_NAMES])
    
    return df_out["pred_cv"].values, df_heldout["pred_heldout"].values

def predict(df: pd.DataFrame, df_heldout: pd.DataFrame) -> pd.DataFrame:
    # Load / train model (add a notebook example exploring TAP features for hobbyists)
    df_tap_train = pd.read_csv(f"{ROOT_DIR}/features/processed_features/GDPa1/TAP.csv")
    df_tap_heldout = pd.read_csv(f"{ROOT_DIR}/features/processed_features/heldout_test/TAP.csv")
    
    df_merged = df.merge(df_tap_train, on="antibody_name", how="left")
    df_merged_heldout = df_heldout.merge(df_tap_heldout, on="antibody_name", how="left")
    
    df_out_cv = df_merged.copy()
    
    
    for property_name in PROPERTY_LIST:
        not_na_mask = df_merged[property_name].notna()
        df_merged_subset = df_merged[not_na_mask].reset_index(drop=True)
        cv_preds, heldout_preds = get_cross_validation_predictions(df_merged_subset, property_name, df_merged_heldout)
        df_out_cv.loc[not_na_mask, property_name] = cv_preds
        df_merged_heldout[property_name] = heldout_preds
    
    return df_out_cv[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"] + PROPERTY_LIST], df_merged_heldout[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"] + PROPERTY_LIST]

def main():
    # Note: Later we'll load this table from huggingface
    df = pd.read_csv(GDPA1_PATH)
    df_heldout = pd.read_csv(HELDOUT_PATH)
    df_predictions_cv, df_predictions_heldout = predict(df, df_heldout)
    
    # Write out under this model name
    model_path = "TAP/TAP - linear regression"
    df_predictions_cv.to_csv(f"predictions/GDPa1_cross_validation/{model_path}.csv", index=False)
    df_predictions_heldout.to_csv(f"predictions/heldout_test/{model_path}.csv", index=False)

if __name__ == "__main__":
    main()