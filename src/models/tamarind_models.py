import os
import pandas as pd
from utils import ROOT_DIR

# Manually choose a few features from Saprot that correlate >=0.2 (later add a model)
# model_name: {feature_name: (assay, directionality)}
MODEL_ASSAY_CHOICES = {
    "TAP": {
        'PNC': [('AC-SINS_pH7.4', -1), ('PR_CHO', -1)], 
        'SFvCSP': [('AC-SINS_pH7.4', 1), ('PR_CHO', 1), ('HIC', -1)], 
        'PPC': [('AC-SINS_pH7.4', 1), ('Titer', 1)], 
        'CDR Length': [('AC-SINS_pH7.4', -1), ('HIC', 1)],
    },
    "Saprot_VH": {
        'solubility_probability': [('PR_CHO', 1)], 
        'stability_score': [('Tm2', -1)],   # Note: This is kinda strange that it's negative
    },
    "DeepViscosity": {
        'Viscosity': [('HIC', 1)],
    },
    "Aggrescan3D": {
        'aggrescan_average_score': [('HIC', 1)],
        'aggrescan_max_score': [('HIC', 1), ("PR_CHO", -1)],
        'aggrescan_90_score': [('HIC', 1)],
        'aggrescan_cdrh3_average_score': [('HIC', 1)],
    },
    "AntiFold": {
        'Score': [('Tm2', -1), ("Titer", -1)],  # Note: This is kinda strange that it's negative
    },
    # TODO check directions with BALM_Paired, TEMPRO, DeepSP
}


def main(features_dir, df_sequences, output_dir):
    """
    Reads in Tamarind feature files and writes out in the expected format for the benchmark (one file per feature of interest)
    """
    model_files = os.listdir(features_dir)
    for model, assay_choices_dict in MODEL_ASSAY_CHOICES.items():
        if f"{model}.csv" not in model_files:
            print(f"Model {model} not found in {features_dir}")
            continue
        df_features = pd.read_csv(f"{features_dir}/{model}.csv")
        num_features = len(assay_choices_dict)
        for feature_name, assay_choices in assay_choices_dict.items():
            model_combined_name = model
            if num_features > 1:
                model_combined_name += f" - {feature_name}"
            print(f"Predicting {model_combined_name}")
            df_merged = df_sequences.merge(df_features[["antibody_name", feature_name]], on="antibody_name", how="left")
            
            assay_names = [assay_name for assay_name, _ in assay_choices]
            for assay_name, directionality in assay_choices:
                df_merged[assay_name] = df_merged[feature_name] * directionality
            
            df_predictions = df_merged[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"] + assay_names]
            os.makedirs(f"{output_dir}/{model}", exist_ok=True)
            df_predictions.to_csv(f"{output_dir}/{model}/{model_combined_name}.csv", index=False)

if __name__ == "__main__":
    # sequences_path = f"{ROOT_DIR}/data/GDPa1_v1.2_20250814.csv"
    # output_dir = f"{ROOT_DIR}/features/processed_features/GDPa1/"
    sequences_path = "data/heldout-set-sequences.csv"
    features_dir = f"{ROOT_DIR}/features/processed_features/heldout_test/"
    output_dir = f"{ROOT_DIR}/predictions/heldout_test/"
    
    df_sequences = pd.read_csv(sequences_path)
    main(features_dir, df_sequences, output_dir)
