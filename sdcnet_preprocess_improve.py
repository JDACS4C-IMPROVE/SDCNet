""" Preprocess data to generate datasets for the prediction model.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import joblib

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve import drug_resp_pred as drp

# Model-specific imports

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_preproc_params
# 2. model_preproc_params
app_preproc_params = []
model_preproc_params = [
    {"name": "learning_rate",
     "type": float,
     "default": 0.001,
     "help": "Initial learning rate.",
    },
    {"name": "dropout",
     "type": float,
     "default": 0.2,
     "help": "Dropout rate (1 - keep probability).",
    },
    {"name": "embedding_dim",
     "type": int,
     "default": 320,
     "help": "Number of the dim of embedding.",
    },
    {"name": "weight_decay",
     "type": float,
     "default": 0.,
     "help": "Weight for L2 loss on embedding matrix.",
    },
    {"name": "val_test_size",
     "type": float,
     "default": 0.1,
     "help": "the rate of validation and test samples.",
    },
    {"name": "datasets",
     "type": str,
     "default": "ALMANAC",
     "help": "datasets to use",
    },
]

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
preprocess_params = app_preproc_params + model_preproc_params
# ---------------------


# [Req]
def run(params: Dict):
    # ------------------------------------------------------
    # [Req] Build paths and create output dir
    # ------------------------------------------------------
    params = frm.build_paths(params)  

    frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # Load X data (feature representations)
    # ------------------------------------------------------
    

    # ------------------------------------------------------
    # Load Y data 
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------

    # ------------------------------------------------------
    # [Req] Create data names for ML data
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_name(params, stage="val")  # [Req]
    test_data_fname = frm.build_ml_data_name(params, stage="test")  # [Req]

    train_data_path = params["ml_data_outdir"] + "/" + train_data_fname
    val_data_path = params["ml_data_outdir"] + "/" + val_data_fname
    test_data_path = params["ml_data_outdir"] + "/" + test_data_fname

    # ------------------------------------------------------
    # Save ML data
    # ------------------------------------------------------
    with open(train_data_path, 'wb+') as f:
        pickle.dump(train_data, f, protocol=4)

    with open(val_data_path, 'wb+') as f:
        pickle.dump(val_data, f, protocol=4)
    
    with open(test_data_path, 'wb+') as f:
        pickle.dump(test_data, f, protocol=4)
   

    return params["ml_data_outdir"]


# [Req]
def main(args):
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    ml_data_outdir = run(params)
    print("\nFinished data preprocessing.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])