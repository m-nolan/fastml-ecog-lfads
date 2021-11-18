# FastML - LFADS for ECoG Signals
Michael Nolan
2021.11.17

### Dependencies
numpy
scipy
torch
yaml
pandas
tqdm

### Description
This package contains model code and analysis/evaluation scripts for a small instance of the multiblock LFADS model (1 block, 256 units). The model checkpoint is ~15MB in pytorch's .pth format.

A trained model is included in the models directory. The ECoG dataset is found in the data directory.

### Running model
From the package directory, run either `example_eval_ecog_lfads.bat` or `example_eval_ecog_lfads.sh` to run the model evalution. This will assess model performance and create reconstruction task figures. The `analysis.py` script contains all functions required to load and evaluate the model.

##### NOTE:
The current code is quite raw and lacks documentation. Please reach out to me with any questions: manolan@uw.edu