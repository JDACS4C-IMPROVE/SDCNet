""" Inference with model for synergy prediction.
"""

import sys
from pathlib import Path
from typing import Dict

import pandas as pd

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm

# Model-specific imports
import time
import os
import scipy.sparse as sp
from itertools import islice, combinations
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, auc, f1_score, accuracy_score, precision_score, recall_score
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_v2_behavior() 
import sdcnet_utils
import pickle
from models.model_mult import sdcnet
#from models.optimizer_mult import Optimizer

# [Req] Imports from preprocess and train scripts
from sdcnet_preprocess_improve import preprocess_params
from sdcnet_train_improve import metrics_list, train_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
app_infer_params = []
model_infer_params = []
infer_params = app_infer_params + model_infer_params
# ---------------------


# [Req]
def run(params):
    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    resultspath = params["model_outdir"]

    def open_file(file_name):
        path_name = params["ml_data_outdir"] + "/" + file_name + ".pkl"
        with open(path_name, 'rb') as f:
            file = pickle.load(f)
        return file
    
    d_net1_norm = open_file("d_net1_norm")
    d_test_edges = open_file("d_test_edges")
    d_test_labels = open_file("d_test_labels")
    drug_feat = open_file("drug_feat")

    #d_pos_weights = open_file("d_pos_weights")
    #d_train_indexs = open_file("d_train_indexs")
    #d_train_labels = open_file("d_train_labels")

    counts_needed_path = params["ml_data_outdir"] + "/counts_needed.pkl"
    with open(counts_needed_path, 'rb') as f:
            cellscount, num_drug_feat, num_drug_nonzeros = pickle.load(f)

    # ------------------------------------------------------
    # Configure tf
    # ------------------------------------------------------
    tf.compat.v1.disable_v2_behavior() 
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    config.allow_soft_placement = True
    config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    # Create placeholders
    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({'net1_adj_norm_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})

    # Create model
    model = sdcnet(placeholders, num_drug_feat, params["embedding_dim"], num_drug_nonzeros, name='sdcnet', use_cellweights=True, use_layerweights=True,  fncellscount =cellscount )
    #with tf.name_scope('optimizer'):
    #    opt = Optimizer(preds= model.reconstructions, d_labels= d_train_labels, model=model, lr= params["learning_rate"], d_pos_weights = d_pos_weights, d_indexs = d_train_indexs )

    # ------------------------------------------------------
    # Load best model
    # ------------------------------------------------------
    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)
    best_model_file = resultspath + '/best_model.ckpt'
    #best_model_meta = resultspath + '/best_model.ckpt.meta'
    #saver = tf.train.import_meta_graph(best_model_meta)
    saver.restore(sess, best_model_file )
    
    feed_dict = dict()
    feed_dict.update({placeholders['features']: drug_feat})
    feed_dict.update({placeholders['dropout']: params["dropout"]})
    feed_dict.update({placeholders['net1_adj_norm_'+str(cellidx)] : d_net1_norm[cellidx] for cellidx in range(cellscount)})
    feed_dict.update({placeholders['dropout']: 0})

    # ------------------------------------------------------
    # Compute predictions
    # ------------------------------------------------------
    # Get results
    res = sess.run( model.reconstructions , feed_dict=feed_dict)

    y_pred = []
    y_labels = []
    for cellidx in range(cellscount):
        preds_all_improve = res[cellidx][ tuple( d_test_edges[cellidx].T )].tolist()
        this_pred = [ 1 if x>=0.5 else 0 for x in preds_all_improve ]
        this_labels = d_test_labels[cellidx]
        y_pred += this_pred
        y_labels += this_labels

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=y_labels, y_pred=y_pred, stage="test",
        outdir=params["infer_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=y_labels, y_pred=y_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores


# [Req]
def main(args):
    additional_definitions = preprocess_params + train_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        default_model="params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])