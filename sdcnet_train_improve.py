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
    data = pd.read_csv('./data/oneil_dataset_loewe.txt', sep='\t', header=0)
    data.columns = ['drugname1','drugname2','cell_line','synergy']

    drugslist = sorted(list(set(list(data['drugname1']) + list(data['drugname2'])))) #38
    drugscount = len(drugslist)
    cellslist = sorted(list(set(data['cell_line']))) 
    cellscount = len(cellslist)

    features = pd.read_csv('./data/oneil_drug_informax_feat.txt',sep='\t', header=None)

    drug_feat = sp.csr_matrix( np.array(features) )
    drug_feat = sdcnet_utils.sparse_to_tuple(drug_feat.tocoo())
    num_drug_feat = drug_feat[2][1]
    num_drug_nonzeros = drug_feat[1].shape[0]

    resultspath = params["model_outdir"]

    all_indexs = []
    all_edges = []
    for idx1 in range(drugscount):
        for idx2 in range(drugscount):
            all_indexs.append([idx1,idx2])
            if idx1 < idx2:
                all_edges.append([idx1, idx2])

    diags_edges = []
    diags_indexs = []
    for idx in range(drugscount):
        diags_edges.append([idx, idx])
        diags_indexs.append( all_indexs.index([idx, idx]) )

    num_folds = 1
    all_stats = np.zeros((num_folds, 6))
    merged_stats = np.zeros((num_folds, 6))

    # -----------------------------
    # Begin CV
    # -----------------------------
    #for foldidx in range( num_folds):
    foldidx = 0
    print('processing fold ', foldidx)

    d_net1_norm = {}
    d_net2_norm = {}
    d_net1_orig = {}
    d_net2_orig = {}
    d_pos_weights = {}
    d_train_edges = {}
    d_train_indexs = {}
    d_train_labels = {}
    d_test_edges = {}
    d_test_labels = {}
    d_new_edges = {}
    d_net3_edges = {}
    d_valid_edges = {}
    d_valid_labels = {}
    for cellidx in range(cellscount):
        cellname = cellslist[cellidx]
        print('processing ', cellname)
        each_data = data[data['cell_line']==cellname]
        net1_data = each_data[each_data['synergy'] >= 10]
        net2_data = each_data[each_data['synergy'] < 0] 
        net3_data = each_data[(each_data['synergy'] >= 0) & (each_data['synergy'] < 10)]
        print(net1_data.shape, net2_data.shape, net3_data.shape)
        d_net1 = {}
        for each in net1_data.values:
            drugname1, drugname2, cell_line, synergy = each
            key = drugname1+ '&' + drugname2
            d_net1[key] = each
            key = drugname2+ '&' + drugname1
            d_net1[key] = each
        d_net2 = {}
        for each in net2_data.values:
            drugname1, drugname2, cell_line, synergy = each
            key = drugname1+ '&' + drugname2
            d_net2[key] = each
            key = drugname2 + '&' + drugname1
            d_net2[key] = each

        adj_net1_mat = np.zeros((drugscount, drugscount))
        adj_net2_mat = np.zeros((drugscount, drugscount))

        for i in range(drugscount):
            for j in range(drugscount):
                drugname1 = drugslist[i]
                drugname2 = drugslist[j]
                key1 = drugname1 + '&' + drugname2
                key2 = drugname2 + '&' + drugname1
                if key1 in d_net1.keys() or key2 in d_net1.keys():
                    adj_net1_mat[i, j] = 1
                elif key1 in d_net2.keys() or key2 in d_net2.keys():
                    adj_net2_mat[i, j] = 1

        adj_net1 = sp.csr_matrix(adj_net1_mat)
        adj_net2 = sp.csr_matrix(adj_net2_mat)

        net1_edges = sdcnet_utils.sparse_to_tuple(sp.triu(adj_net1))[0]
        net2_edges = sdcnet_utils.sparse_to_tuple(sp.triu(adj_net2))[0]

        #split the train and test edges
        num_test = int(np.floor(net1_edges.shape[0] * params["val_test_size"]))
        net1_edge_idx = list(range(net1_edges.shape[0]))
        np.random.seed(1)
        np.random.shuffle(net1_edge_idx)
        net1_test_edge_idx = net1_edge_idx[(foldidx - 1) * num_test: ]
        net1_valid_edge_idx = net1_edge_idx[foldidx * num_test: (foldidx+1)*num_test]
        net1_test_edges = net1_edges[ net1_test_edge_idx ]
        net1_valid_edges = net1_edges[ net1_valid_edge_idx ]
        net1_train_edge_idx = [ x for x in net1_edge_idx if x not in net1_test_edge_idx + net1_valid_edge_idx ]
        net1_train_edges = net1_edges[net1_train_edge_idx]
        net1_train_data = np.ones(net1_train_edges.shape[0])
        net1_adj_train = sp.csr_matrix( (net1_train_data, (net1_train_edges[:, 0], net1_train_edges[:, 1])), shape= adj_net1.shape )
        net1_adj_train = net1_adj_train + net1_adj_train.T
        net1_adj_norm = sdcnet_utils.preprocess_graph(net1_adj_train)
        net1_adj_orig = net1_adj_train.copy() #this the label
        net1_adj_orig = sdcnet_utils.sparse_to_tuple(sp.csr_matrix(net1_adj_orig))

        ##net2
        ##net2
        net2_edge_idx = list(range(net2_edges.shape[0]))
        #1.the number of negative samples are split into equal subsets
        # num_test2 = int(np.floor(net2_edges.shape[0] * FLAGS.val_test_size))

        #2. the number of negative sample is equal to positive samples
        num_test2 = num_test

        np.random.seed(2)
        np.random.shuffle(net2_edge_idx)

        net2_test_edge_idx = net2_edge_idx[(foldidx - 1) * num_test2: ]
        net2_valid_edge_idx = net2_edge_idx[foldidx * num_test2: (foldidx+1)*num_test2]
        net2_test_edges = net2_edges[ net2_test_edge_idx ]
        net2_valid_edges = net2_edges[ net2_valid_edge_idx ]
        net2_train_edge_idx = [ x for x in net2_edge_idx if x not in net2_test_edge_idx + net2_valid_edge_idx ]
        net2_train_edges = net2_edges[net2_train_edge_idx]
        ##
        net1_train_edges_symmetry = np.array([  [x[1],x[0]] for x in net1_train_edges ])
        net2_train_edges_symmetry = np.array([  [x[1],x[0]] for x in net2_train_edges ])
        net1_train_edges = np.concatenate([net1_train_edges, net1_train_edges_symmetry])
        net2_train_edges = np.concatenate([net2_train_edges, net2_train_edges_symmetry])

        test_edges = np.concatenate([net1_test_edges, net2_test_edges])
        y_test = [1] * net1_test_edges.shape[0] + [0] * net2_test_edges.shape[0]
        valid_edges = np.concatenate([net1_valid_edges, net2_valid_edges])
        y_valid = [1] * net1_valid_edges.shape[0] + [0] * net2_valid_edges.shape[0]
        train_edges = np.concatenate([net1_train_edges, net2_train_edges])
        y_train = [1] * net1_train_edges.shape[0] + [0] * net2_train_edges.shape[0]
        train_indexs = [ all_indexs.index(x) for x in train_edges.tolist() ]
        each_pos_weight = len(net2_train_edges) / len(net1_train_edges)

        d_pos_weights[cellidx] = each_pos_weight
        d_net1_norm[cellidx] = net1_adj_norm
        d_net1_orig[cellidx] = net1_adj_orig
        d_test_edges[cellidx] = test_edges
        d_test_labels[cellidx] = y_test
        d_train_edges[cellidx] = train_edges
        d_train_indexs[cellidx] = train_indexs
        d_train_labels[cellidx] = y_train
        d_valid_edges[cellidx] = valid_edges
        d_valid_labels[cellidx] = y_valid

    # save and restore files here 

    frm.create_outdir(outdir=params["ml_data_outdir"])
    def save_file(file_name):
        path_name = params["ml_data_outdir"] + "/" + file_name + ".pkl"
        with open(path_name, 'wb+') as f:
            pickle.dump(eval(file_name), f, protocol=4)

    save_file("d_pos_weights")

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