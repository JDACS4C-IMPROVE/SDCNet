""" Train model for synergy prediction.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd


# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from improve.metrics import compute_metrics

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

# [Req] Imports from preprocess script
from sdcnet_preprocess_improve import preprocess_params

filepath = Path(__file__).resolve().parent # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_train_params
# 2. model_train_params
app_train_params = []
model_train_params = []
train_params = app_train_params + model_train_params
# ---------------------

# [Req] List of metrics names to compute prediction performance scores
# metrics_list = ["mse", "rmse", "pcc", "scc", "r2"] 
# or
metrics_list = ["mse", "acc", "recall", "precision", "f1"]


# [Req]
def run(params):
    
    # ------------------------------------------------------
    # [Req] Create output dir and build model path
    # ------------------------------------------------------
    # Create output dir for trained model, val set predictions, val set
    # performance scores
    frm.create_outdir(outdir=params["model_outdir"])

    # Build model path
    modelpath = frm.build_model_path(params, model_dir=params["model_outdir"])

    # ------------------------------------------------------
    # [Req] Create data names for train and val sets
    # ------------------------------------------------------
    train_data_fname = frm.build_ml_data_name(params, stage="train")  # [Req]
    val_data_fname = frm.build_ml_data_name(params, stage="val")  # [Req]

    train_data_path = params["ml_data_outdir"] + "/" + train_data_fname
    val_data_path = params["ml_data_outdir"] + "/" + val_data_fname
    
    # ------------------------------------------------------
    # CUDA/CPU device
    # ------------------------------------------------------
    # ------------------------------------------------------
    # Prepare model
    # ------------------------------------------------------
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)
    config.allow_soft_placement = True
    config.log_device_placement = True
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    # ------------------------------------------------------
    # Load data
    # ------------------------------------------------------
    def open_file(file_name):
        path_name = params["ml_data_outdir"] + "/" + file_name + ".pkl"
        with open(path_name, 'rb') as f:
            file = pickle.load(f)
        return file
    
    d_pos_weights = open_file("d_pos_weights")
    d_net1_norm = open_file("d_net1_norm")
    d_net1_orig = open_file("d_net1_orig")
    d_test_edges = open_file("d_test_edges")
    d_test_labels = open_file("d_test_labels")
    d_train_edges = open_file("d_train_edges")
    d_train_indexs = open_file("d_train_indexs")
    d_train_labels = open_file("d_train_labels")
    d_valid_edges = open_file("d_valid_edges")
    d_valid_labels = open_file("d_valid_labels")

    counts_needed_path = params["ml_data_outdir"] + "/counts_needed.pkl"
    with open(counts_needed_path, 'rb') as f:
            cellscount, num_drug_feat, num_drug_nonzeros = pickle.load(f)

    placeholders = {
        'features': tf.sparse_placeholder(tf.float32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({'net1_adj_norm_'+str(cellidx) : tf.sparse_placeholder(tf.float32) for cellidx in range(cellscount)})

    # Create model
    from models.model_mult import sdcnet
    model = sdcnet(placeholders, num_drug_feat, params["embedding_dim"], num_drug_nonzeros, name='sdcnet', use_cellweights=True, use_layerweights=True,  fncellscount =cellscount )
    #optimizer
    from models.optimizer_mult import Optimizer
    with tf.name_scope('optimizer'):
        opt = Optimizer(preds= model.reconstructions, d_labels= d_train_labels, model=model, lr= params["learning_rate"], d_pos_weights = d_pos_weights, d_indexs = d_train_indexs )

    # Initialize session
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1)

    best_model_file = resultspath + '/best_model_' + '.ckpt'
    best_acc = 0
    # -----------------------------
    # Train. Iterate over epochs.
    # -----------------------------
    for epoch in range(params["epochs"]):
        # epoch =  0
        feed_dict = dict()
        feed_dict.update({placeholders['features']: drug_feat})
        feed_dict.update({placeholders['dropout']: params["dropout"]})
        feed_dict.update({placeholders['net1_adj_norm_'+str(cellidx)] : d_net1_norm[cellidx] for cellidx in range(cellscount)})
        _, avg_cost = sess.run([opt.opt_op, opt.cost], feed_dict=feed_dict)
        if epoch % 10 == 0:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(avg_cost))
        feed_dict.update({placeholders['dropout']: 0})
        res = sess.run( model.reconstructions, feed_dict=feed_dict)

        merged_preds = []
        merged_labels = []
        for cellidx in range(cellscount):
            preds_all = res[cellidx][ tuple( d_valid_edges[cellidx].T )].tolist()
            labels_all = d_valid_labels[cellidx]
            merged_preds += preds_all
            merged_labels += labels_all
        merged_preds_binary = [1 if x >= 0.5 else 0 for x in merged_preds ]
        merged_acc = accuracy_score(merged_preds_binary, merged_labels)
        if best_acc < merged_acc:
            best_acc = merged_acc
            saver.save(sess, best_model_file)


    ###### this should all go to infer
    saver.restore(sess, best_model_file )

    feed_dict = dict()
    feed_dict.update({placeholders['features']: drug_feat})
    feed_dict.update({placeholders['dropout']: params["dropout"]})
    feed_dict.update({placeholders['net1_adj_norm_'+str(cellidx)] : d_net1_norm[cellidx] for cellidx in range(cellscount)})

    ##test predict
    feed_dict.update({placeholders['dropout']: 0})
    res = sess.run( model.reconstructions , feed_dict=feed_dict)

    # improve metrics
    y_pred = []
    y_labels = []
    for cellidx in range(cellscount):
        preds_all_improve = res[cellidx][ tuple( d_test_edges[cellidx].T )].tolist()
        this_pred = [ 1 if x>=0.5 else 0 for x in preds_all_improve ]
        this_labels = d_test_labels[cellidx]
        y_pred += this_pred
        y_labels += this_labels

    scores = frm.compute_performace_scores(params, y_true=y_labels, y_pred=y_pred, stage="val", outdir=params["model_outdir"], metrics=metrics_list)




    # -----------------------------
    # Save model
    # -----------------------------

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------


    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    # frm.store_predictions_df(
    #     params,
    #     y_true=val_true, y_pred=val_pred, stage="val",
    #     outdir=params["model_outdir"]
    # )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    # val_scores = frm.compute_performace_scores(
    #     params,
    #     y_true=val_true, y_pred=val_pred, stage="val",
    #     outdir=params["model_outdir"], metrics=metrics_list
    # )
    val_scores = 0

    return val_scores




# [Req]
def main(args):
    additional_definitions = preprocess_params + train_params
    params = frm.initialize_parameters(
        filepath,
        default_model="params.txt",
        additional_definitions=additional_definitions,
        required=None,
    )
    val_scores = run(params)
    print("\nFinished training model.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])