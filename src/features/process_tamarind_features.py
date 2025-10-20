from ast import literal_eval
import os
import pandas as pd

from utils import ROOT_DIR, load_from_tamarind, get_indices, extract_region

TAMARIND_DIR = f"{ROOT_DIR}/features/raw_features/tamarind_scores/heldout_test"
MODEL_NAMES = {
    "viscosity": "DeepViscosity",
    "saprot": "Saprot_VH",
    "tap": "TAP",
    "antifold": "AntiFold",
    "aggrescan": "Aggrescan3D",
    "deepsp": "DeepSP",
    "tempro": "TEMPRO",
    "balm_paired": "BALM_Paired",
}

# Optional
FEATURE_NAMES_PER_MODEL = {
    "DeepViscosity": ["Viscosity"],
    "Saprot_VH": ["stability_score", "solubility_probability"],
    "Aggrescan3D": ["aggrescan_average_score", "aggrescan_max_score", "aggrescan_90_score", "aggrescan_cdrh3_average_score"],
    "AntiFold": ["Score"],
}
# DeepSP, TAP, BALM_Paired, TEMPRO: Return all features

def process_aggrescan(df_sequences, aggrescan_path):
    # The sequences file is used to get the CDR3 positions for slicing
    scores = []
    antibodies_missing = []
    assert os.path.exists(aggrescan_path), f"Aggrescan3D directory not found at {aggrescan_path}"
    for i, antibody_name in enumerate(df_sequences["antibody_name"]):
        try:
            matching_dir = [d for d in os.listdir(aggrescan_path) if antibody_name in d]
            assert len(matching_dir) == 1, f"Multiple matching directories found for {antibody_name}: {matching_dir}"
            matching_dir = matching_dir[0]
            tmp = pd.read_csv(os.path.join(TAMARIND_DIR, aggrescan_path, matching_dir, "A3D.csv"))
            result = {"antibody_name": antibody_name, "residue_scores": tmp["score"].values, "residues": "".join(tmp["residue_name"].values), "aggrescan_average_score": tmp["score"].mean()}
            
            result["aggrescan_max_score"] = tmp["score"].max()
            result["aggrescan_90_score"] = tmp["score"].quantile(0.9)
            
            gdpa1_row = df_sequences.loc[df_sequences["antibody_name"] == antibody_name].iloc[0]
            
            cdrh3_scores = tmp["score"].values[[a for a in gdpa1_row["heavy_aho_indices"] if (a >= 112) and (a <= 138)]]
            if len(cdrh3_scores) == 0:
                print(f"No CDR3 scores found for {antibody_name}")
                result["aggrescan_cdrh3_average_score"] = None
            else:
                result["aggrescan_cdrh3_average_score"] = cdrh3_scores.mean()
            
            result["cdrh3_scores"] = extract_region(tmp["score"].values, gdpa1_row["heavy_aho_indices"], "CDRH3")
            scores.append(result)
        except FileNotFoundError:
            antibodies_missing.append(antibody_name)
    print(f"missing {len(antibodies_missing)} antibodies")
    df_aggrescan = pd.DataFrame(scores)
    return df_aggrescan

def main(sequences_path, output_dir):
    # For each file in the scores, find the corresponding model name
    df_sequences = pd.read_csv(sequences_path)
    
    for filename in os.listdir(TAMARIND_DIR):
        if filename.endswith(".csv"):
            matching_substring = [s for s in MODEL_NAMES.keys() if s in filename]
            if len(matching_substring) == 0:
                print(f"No matching model found for {filename}")
                continue
            elif len(matching_substring) > 1:
                print(f"Multiple matching models found for {filename}: {matching_substring}")
                continue
            model = MODEL_NAMES[matching_substring[0]]
            df_scores = load_from_tamarind(f"{TAMARIND_DIR}/{filename}", only_return_features_and_names=True, df_sequences=df_sequences)
            if model == "Saprot_VH":
                # Take quantitative solubility score
                df_scores["solubility_probability"] = df_scores["solubility_probabilities"].apply(lambda x: literal_eval(x)[0][1])
            elif model == "AntiFold":
                print(df_scores.head(50))
                df_scores = df_scores.query("`Sample` == 'input_imgt_HL'")
            if model in FEATURE_NAMES_PER_MODEL:
                # Only select a subset of features
                df_scores = df_scores[["antibody_name"] + FEATURE_NAMES_PER_MODEL[model]]
            df_scores.to_csv(f"{output_dir}/{model}.csv", index=False)

    # Handle directories differently
    df_sequences["light_aho_indices"] = df_sequences["light_aligned_aho"].apply(get_indices)
    df_sequences["heavy_aho_indices"] = df_sequences["heavy_aligned_aho"].apply(get_indices)
    for folder in os.listdir(TAMARIND_DIR):
        if os.path.isdir(f"{TAMARIND_DIR}/{folder}") and "aggrescan" in folder:
            model = "Aggrescan3D"
            df_aggrescan = process_aggrescan(df_sequences, f"{TAMARIND_DIR}/{folder}")
            df_aggrescan[["antibody_name"] + FEATURE_NAMES_PER_MODEL[model]].to_csv(f"{output_dir}/{model}.csv", index=False)
    
if __name__ == "__main__":
    # sequences_path = f"{ROOT_DIR}/data/GDPa1_v1.2_20250814.csv"
    # output_dir = f"{ROOT_DIR}/features/processed_features/GDPa1/"
    sequences_path = "data/heldout-set-sequences.csv"
    output_dir = f"{ROOT_DIR}/features/processed_features/heldout_test/"
    main(sequences_path, output_dir)
