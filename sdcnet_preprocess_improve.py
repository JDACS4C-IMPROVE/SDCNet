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
import pickle
import scipy.sparse as sp
import sdcnet_utils

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
    #params = frm.build_paths(params)  

    frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # Load X data (feature representations)
    # ------------------------------------------------------
    data = pd.read_csv('./data/oneil_dataset_loewe.txt', sep='\t', header=0)
    data.columns = ['drugname1','drugname2','cell_line','synergy']

    drugslist = sorted(list(set(list(data['drugname1']) + list(data['drugname2'])))) #38
    drugscount = len(drugslist)
    cellslist = sorted(list(set(data['cell_line']))) 
    cellscount = len(cellslist)
    print("cellscount: ", cellscount)

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


    # ------------------------------------------------------
    # Load Y data 
    # ------------------------------------------------------

    # ------------------------------------------------------
    # Construct ML data for every stage (train, val, test)
    # ------------------------------------------------------

    # ------------------------------------------------------
    # [Req] Create data names for ML data
    # ------------------------------------------------------


    # ------------------------------------------------------
    # Save ML data
    # ------------------------------------------------------
    def save_file(file, file_name):
        path_name = params["ml_data_outdir"] + "/" + file_name + ".pkl"
        with open(path_name, 'wb+') as f:
            pickle.dump(file, f, protocol=4)

    save_file(d_pos_weights, "d_pos_weights")
    save_file(d_net1_norm, "d_net1_norm")
    save_file(d_net1_orig, "d_net1_orig")
    save_file(d_test_edges, "d_test_edges")
    save_file(d_test_labels, "d_test_labels")
    save_file(d_train_edges, "d_train_edges")
    save_file(d_train_indexs, "d_train_indexs")
    save_file(d_train_labels, "d_train_labels")
    save_file(d_valid_edges, "d_valid_edges")
    save_file(d_valid_labels, "d_valid_labels")
    save_file(drug_feat, "drug_feat")
   
    counts_needed_path = params["ml_data_outdir"] + "/counts_needed.pkl"
    with open(counts_needed_path, 'wb+') as f:
            pickle.dump([cellscount, num_drug_feat, num_drug_nonzeros], f, protocol=1)

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