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
git clone https://github.com/zzyywww/ssh2.git
cd ssh2
chmod 777 -R ./trained
```
```
import sys
import pandas as pd
from pathlib import Path
import os
import model

SSH2Model = model.SSH2Model()


test_data = pd.DataFrame({
        'antibody_name': ['test_ab1', 'test_ab2'],
        'vh_protein_sequence': ['QVKLQESGAELARPGASVKLSCKASGYTFTNYWMQWVKQRPGQGLDWIGAIYPGDGNTRYTHKFKGKATLTADKSSSTAYMQLSSLASEDSGVYYCARGEGNYAWFAYWGQGTTVTVSS', 
                                'QVQLQQSGGELAKPGASVKVSCKASGYTFSSFWMHWVRQAPGQGLEWIGYINPRSGYTEYNEIFRDKATMTTDTSTSTAYMELSSLRSEDTAVYYCASFLGRGAMDYWGQGTTVTVSS'],
        'vl_protein_sequence': ['DIELTQSPASLSASVGETVTITCQASENIYSYLAWHQQKQGKSPQLLVYNAKTLAGGVSSRFSGSGSGTHFSLKIKSLQPEDFGIYYCQHHYGILPTFGGGTKLEIK', 
                                'DIQMTQSPSSLSASVGDRVTITCRASQDISNYLAWYQQKPGKAPKLLIYYTSKIHSGVPSRFSGSGSGTDYTFTISSLQPEDIATYYCQQGNTFPYTFGQGTKVEIK']
    })
	
	
run_dir = Path("./test_output")
SSH2Model.train(test_data, run_dir, seed=42)

result = SSH2Model.predict(test_data, run_dir)
```

or Get started with the SSH2 model by running the example provided in example.ipynb.


### outputs

The model outputs a hydrophobicity score ranging from 0 to 1 and a predicted label with a default threshold of 0.5.
Scores above 0.5 indicate high hydrophobicity (1), while scores below 0.5 indicate low hydrophobicity (0).

## Reference

Zhou Y, Xie S, Yang Y, et al. SSH2.0: A Better Tool for Predicting the Hydrophobic Interaction Risk of Monoclonal Antibody. Front Genet. 2022;13:842127. doi:10.3389/fgene.2022.842127
