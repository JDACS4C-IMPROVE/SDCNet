# IMPROVE - SDCNet: Drug Synergy Prediction

---

This is the IMPROVE implementation of the original model with original data. This is the implementation of the script with Loewe synergy values. The code has been restructured so that instead of the original 5-fold cross validation, it is one fold when all three scripts are run.

## Dependencies and Installation
### Conda Environment
```
conda create --name sdcnet_IMPROVE python=3.8 pandas=1.3.5 numpy=1.21.2 tensorflow-gpu=2.4.1 scikit-learn=1.2.2
conda activate sdcnet_IMPROVE
pip install git+https://github.com/ECP-CANDLE/candle_lib@develop
```

### Clone this repository
```
git clone https://github.com/JDACS4C-IMPROVE/SDCNet
cd matchmaker
git checkout IMPROVE-original
cd ..
```

### Clone IMPROVE repository
```
git clone https://github.com/JDACS4C-IMPROVE/IMPROVE
cd IMPROVE
git checkout develop
cd ..
```

### Download Original Data
The original data is in this repo in /data.


## Running the Model
Activate the conda environment:

```
conda activate sdcnet_IMPROVE
```

Set environment variables:
```
export IMPROVE_DATA_DIR="./"
export PYTHONPATH=$PYTHONPATH:/your/path/to/IMPROVE
```

Run preprocess, train, and infer scripts:
```
python sdcnet_preprocess_improve.py
python sdcnet_train_improve.py
python sdcnet_infer_improve.py
```



## References
Original GitHub: https://github.com/yushenshashen/SDCNet

Original paper: https://pubmed.ncbi.nlm.nih.gov/36136353/

