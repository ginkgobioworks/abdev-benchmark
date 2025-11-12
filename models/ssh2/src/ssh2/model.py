"""SSH2 model implementation."""

from pathlib import Path
import os
import stat
import pandas as pd
from .codes import readFasta, saveCode, CKSAAGP
import json
import platform

from abdev_core import BaseModel



class SSH2Model(BaseModel):
    """SSH2 is a pre-trained ensemble SVM classification model that predicts antibody hydrophobicity using CKSAAGP features.

    This model first concatenates the heavy (VH) and light (VL) chains, then extracts CKSAAGP 
    features from the combined sequence using iFeature. The approach utilizes three distinct 
    feature groups, each processed by a separate pre-trained model. The individual predictions 
    are then integrated to produce the final hydrophobicity score.

    The model outputs a hydrophobicity score ranging from 0 to 1 and a predicted label with a default threshold of 0.5.
    Scores above 0.5 indicate high hydrophobicity (1), while scores below 0.5 indicate low hydrophobicity (0).

    LIBSVM (Chang and Lin., 2011) was employed to construct the SVM sub-models. 
    """
    
    def _create_fasta_file(self, df: pd.DataFrame, fasta_path: Path):
        """create FASTA file"""
        df['seq_all'] = df['vh_protein_sequence'] + df['vl_protein_sequence']
        with open(fasta_path, 'w') as f:
            for _, row in df.iterrows():
                f.write(f">{row['antibody_name']}\n")
                f.write(f"{row['seq_all']}\n")
        print(f"FASTA file created: {fasta_path}")

    def _extract_CKSAAGP_features(self, fasta_path: Path, output_path: Path):        
        fastas = readFasta.readFasta(fasta_path)
        kw = {'path': None, 'train': None, 'label': None, 'order': "ACDEFGHIKLMNPQRSTVWY"}
        encodings = CKSAAGP.CKSAAGP(fastas,gap = 5,**kw)
        saveCode.savetsv(encodings, output_path)
        print(f"CKSAAGP feature extraction file created: {output_path}")

    def _generate_svm_file(self, csv_file: Path, svm_file: Path):
        """format conversion"""
        with open(svm_file, 'w') as g:
            with open(csv_file) as f:
                lines = f.readlines()[1:]  
                
                for eachline in lines:
                    parts = eachline.strip().split(',')
                    label = parts[0]
                    values = parts[1:]
                    
                    svm_line = label
                    for i, val in enumerate(values, 1):
                        svm_line += f' {i}:{val}'
                            
                    g.write(f'{svm_line}\n')

    def train(self, df: pd.DataFrame, run_dir: Path, *, seed: int = 42):
        """No-op training - this baseline uses pre-trained model.
        
        Saves temp files to run_dir for consistency.
        
        Args:
            df: Training dataframe (not used)
            run_dir: Directory to temp save files
            seed: Random seed (not used)
        """
        run_dir.mkdir(parents=True, exist_ok=True)

        # Save configuration for reference
        config = {
            "model_type": "ssh2",
            "note": "Non-training baseline using pre-trained SVM-based ensemble models"
        }

        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        
        print("=" * 60)
        print("SSH2 Model Information")
        print("=" * 60)
        print("SSH2 is a pre-trained ensemble model for antibody hydrophobicity prediction.LIBSVM (Chang and Lin., 2011) was employed to construct the SVM sub-models.")
        print("The training process has already been completed using three SVM sub-models. The trained models and all the necessary program/files can be found in ./trained folder")
        print("")
        print("Model Components:")
        print("- Three pre-trained SVM classifiers with CKSAAGP features")
        print("- Ensemble voting mechanism for final prediction")
        print("- Default threshold: 0.5 (scores > 0.5 indicate high hydrophobicity)")
        print("")
        print("To use SSH2, simply call the predict() method with your antibody sequences.")
        print("=" * 60)
    
    
    def predict(self, df: pd.DataFrame, run_dir: Path) -> pd.DataFrame:
        """Create a FASTA file"""
        test_fasta = run_dir / "test_temp.fasta"
        self._create_fasta_file(df, test_fasta)

        """Generate CKSAAGP feature for all samples """
        test_features = run_dir / "test_features.tsv"
        self._extract_CKSAAGP_features(test_fasta, test_features)

        """load selected feature files"""
        
        trained_dir = Path(__file__).parent.parent.parent / "trained"
        
        # Ensure executable permissions on binary files
        for exe_name in ['svm-scale', 'svm-predict', 'svm-scale.exe', 'svm-predict.exe']:
            exe_path = trained_dir / exe_name
            if exe_path.exists():
                exe_path.chmod(exe_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        
        f1 = open(trained_dir / 'feature1.txt').readlines()
        f2 = open(trained_dir / 'feature2.txt').readlines()
        f3 = open(trained_dir / 'feature3.txt').readlines()

        feature_df = pd.read_csv(run_dir / "test_features.tsv",sep = '\t')
        f_list = [f1, f2, f3]
        flag = 1
        for fi in f_list:
            temp = []
            for i in fi:
                i = i.replace('\n','')
                temp.append(i)
            df1 = feature_df[temp]
            df1 = df1.copy()
            df1['class'] = 1
            cols = df1.columns.tolist()
            cols = cols[-1:] + cols[:-1]
            df1 = df1[cols]
            df1.to_csv(f'{run_dir}/df_{str(flag)}.csv',sep=',',index=False)
            self._generate_svm_file(f'{run_dir}/df_{str(flag)}.csv',f'{run_dir}/df_{str(flag)}.svm')
            flag += 1

        range_files = sorted([x for x in os.listdir(trained_dir) if x.endswith('.range')])
        model_files = sorted([x for x in os.listdir(trained_dir) if x.endswith('.model')])
        test_files = sorted([x for x in os.listdir(run_dir) if x.endswith('.svm')])     

        """predit using three submodels"""
        system = platform.system()
        if system == "Windows":
            scale_exe = "svm-scale.exe"
            predict_exe = "svm-predict.exe"
        else:
            scale_exe = "svm-scale"
            predict_exe = "svm-predict"

        for i in range(3):
            """feature data scale"""
            scale_cmd = f"{trained_dir}/{scale_exe} -r {trained_dir}/{range_files[i]} {run_dir}/{test_files[i]} > {run_dir}/temp.scale"
            os.system(scale_cmd)

            """predict"""
            predict_cmd = f"{trained_dir}/{predict_exe} -b 1 {run_dir}/temp.scale {trained_dir}/{model_files[i]} {run_dir}/{i+1}_result.txt"
            os.system(predict_cmd)


        """ensemble submodels"""    
        result_files = [x for x in os.listdir(run_dir) if x.endswith('_result.txt')]
        group1 = pd.read_csv(run_dir / result_files[0], sep=' ', header=0)
        group2 = pd.read_csv(run_dir / result_files[1], sep=' ', header=0)
        group3 = pd.read_csv(run_dir / result_files[2], sep=' ', header=0)

        result_df = pd.concat([group1, group2, group3], axis=1)

        vote_class = result_df.iloc[:,[0,3,6]].apply(lambda x:x.sum(),axis = 1)
        result_df['vote'] = vote_class
        result_df.loc[result_df['vote'] < 2,'Predict'] = 0
        result_df.loc[result_df['vote'] >= 2,'Predict'] = 1

        """When calculating the probability, take the average of the probabilities that contribute to the final classification."""
        probas = []
        for i in range(result_df.shape[0]):
            Sum = 0
            vote = result_df.loc[i,'vote'].astype(int)
            if result_df.loc[i,'vote'] > 1:
                for i in result_df.loc[i,'1']:
                    if i >= 0.5:
                        Sum += i
                proba = Sum/vote
                probas.append(proba)
            else:
                for i in result_df.loc[i,'1']:
                    if i < 0.5:
                        Sum += i
                proba = Sum/(3-vote)
                probas.append(proba)
        
        
        result = df[["antibody_name", "vh_protein_sequence", "vl_protein_sequence"]].copy()
        result["HIC"] = probas
        return result

