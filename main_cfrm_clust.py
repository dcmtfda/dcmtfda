#!/usr/bin/env python
# coding: utf-8


from dfmf import dfmf

import numpy as np
import pickle as pkl
import pandas as pd
import os
import sys
import scipy as sp

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
#from sklearn.metrics.cluster import rand_score
from numpy import linalg as la
from sklearn.cluster import KMeans

import collections


import random

np.random.seed(0)
random.seed(0)


import bz2
import pickle as cPickle

# Load any compressed pickle file
def decompress_pickle(file):
    assert file.endswith("pbz2")
    data = bz2.BZ2File(file, "rb")
    data = cPickle.load(data)
    return data


def get_size_fac_pp(X_cg):
    #print("get_size_fac_pp - X.shape: ",X_cg.shape)
    #
    epsilon1 = 1
    epsilon2 = 0
    X_cg = X_cg + epsilon1
    #print("get_size_fac_pp - X.shape: ",X_cg.shape)
    X_cg_sum = np.sum(X_cg,axis=1)
    X_cg_size_fac = X_cg_sum / (np.median(X_cg_sum)+epsilon2)
    #print("get_size_fac_pp - X_cg_size_fac.shape: ",X_cg_size_fac.shape)
    #Eqn (4)
    X_cg_size_fac_diag = np.diag(1.0 / X_cg_size_fac)
    X_cg_size_fac_diag_mm_x = np.dot(X_cg_size_fac_diag,X_cg)
    X_cg_size_fac_diag_mm_x_log = np.log(X_cg_size_fac_diag_mm_x + 1)
    #print("get_size_fac_pp - X_cg_size_fac_diag_mm_x_log.shape: ",X_cg_size_fac_diag_mm_x_log.shape)
    pp_scaler = StandardScaler()
    X_cg_size_fac_diag_mm_x_log_std = pp_scaler.fit_transform(X_cg_size_fac_diag_mm_x_log)
    #
    #X_cg_size_fac = X_cg_size_fac - 1
    #
    return X_cg_size_fac, X_cg_size_fac_diag_mm_x_log_std

#assert len(sys.argv) == 2 or len(sys.argv) == 3,"Usage: python main_cfrm_clust.py dataset_id data_version_name"
assert len(sys.argv) == 2 ,"Usage: python main_cfrm_clust.py dataset_id"

dataset_id = sys.argv[1]

data_version_name = "2"
# adr_data_version_name = None
# if len(sys.argv) == 3:
#     adr_data_version_name = sys.argv[2]
#     print("data_version_name: ",adr_data_version_name)

assert dataset_id in ["wiki1","wiki2","wiki3","wiki4","ade1","ade2","ade3","ade4", "pubmed_heuristic"],"Unknown dataset_id: "+str(dataset_id)
print("dataset_id: ",dataset_id)

# if dataset_id in ["freebase"]:
#     assert not adr_data_version_name == None

data_dir = "./data/"
fname = data_dir + dataset_id + "/" 


if dataset_id == "wiki1":
   fname += "dict_data_item_subj_v4_hidden_run1.pkl.p2.pbz2"
elif dataset_id == "wiki2":
   fname += "dict_data_item_subj_v4_hidden_run2.pkl.p2.pbz2"
elif dataset_id == "wiki3":
   fname += "dict_data_item_subj_v4_hidden_run3.pkl.p2.pbz2"
elif dataset_id == "wiki4":
   fname += "dict_data_item_subj_v4_hidden_run4.pkl.p2.pbz2"     
elif dataset_id == "pubmed":
   data_version_name = "2"
   fname += "pubmed_data_dict_v"+data_version_name+"_p2.pkl.pbz2" 
elif dataset_id == "pubmed_heuristic":
   data_version_name = "2"
   fname += "pubmed_heuristic_data_dict_v"+data_version_name+"_p2.pkl.pbz2"    
elif dataset_id == "freebase":
   data_version_name = "5"
   fname += "freebase_data_dict_v"+data_version_name+"_p2.pkl.pbz2" 
elif dataset_id == "genephene":
   data_version_name = "2"
   fname += "genephene_data_dict_v"+data_version_name+"_p2.pkl.pbz2"
elif dataset_id == "ade1":
   num_batches = 1
   data_version_name = "5"
   fname += "dict_nsides_mimic_data_v"+data_version_name+"_case1_part.pkl.pbz2"
elif dataset_id == "ade2":
   num_batches = 1
   data_version_name = "1"
   fname += "dict_nsides_mimic_data_v"+data_version_name+"_case1_part.pkl.pbz2"    
elif dataset_id == "ade3":
   num_batches = 1
   data_version_name = "2"
   fname += "dict_nsides_mimic_data_v"+data_version_name+"_case1_part.pkl.pbz2"    
elif dataset_id == "ade4":
   num_batches = 1
   data_version_name = "4"
   fname += "dict_nsides_mimic_data_v"+data_version_name+"_case1_part.pkl.pbz2"    
else:
   assert False,"Unknown dataset_id: "+str(dataset_id) 

print("data_version_name: ",data_version_name)

#
out_dir = "./out_clust_cfrm/"+dataset_id+"/"+"version_"+data_version_name+"/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)
#

#####################################################################


#data_dict = pkl.load(open(fname,"rb"))
data_dict = decompress_pickle(fname)

if dataset_id in ["wiki1","wiki2","wiki3","wiki4"]:
    is_fac_pp = True
    print("is_fac_pp: unused ")
    print("#")

    
    X_ts = data_dict["data_hidden"]['X_item_subj']
    X_sb = data_dict["data_hidden"]['X_subj_bow']
    X_tb = data_dict["data_hidden"]['X_item_bow']

    #
    print("#")
    print("Before pp:")
    print("#")
    print("X_ts:")
    print("min: ",X_ts.min()," max: ",X_ts.max()," mean: ",X_ts.mean()," sd: ",X_ts.std())
    print("X_sb:")
    print("min: ",X_sb.min()," max: ",X_sb.max()," mean: ",X_sb.mean()," sd: ",X_sb.std())
    print("X_tb:")
    print("min: ",X_tb.min()," max: ",X_tb.max()," mean: ",X_tb.mean()," sd: ",X_tb.std())

    X_data_bef_pp = {
        "X_ts":X_ts, \
        "X_sb":X_sb,
        "X_tb":X_tb}

    scaler1 = StandardScaler()
    X_ts = scaler1.fit_transform(X_ts)
    scaler2 = StandardScaler()
    X_sb = scaler2.fit_transform(X_sb)
    scaler3 = StandardScaler()
    X_tb = scaler3.fit_transform(X_tb)

    #X_sb, _ = get_size_fac_pp(X_sb)
    #X_t2b, _ = get_size_fac_pp(X_t2b)

    # In[16]:

    print("#")
    print("After pp:")
    print("#")
    print("X_ts:")
    print("min: ",X_ts.min()," max: ",X_ts.max()," mean: ",X_ts.mean()," sd: ",X_ts.std())
    print("X_sb:")
    print("min: ",X_sb.min()," max: ",X_sb.max()," mean: ",X_sb.mean()," sd: ",X_sb.std())
    print("X_tb:")
    print("min: ",X_tb.min()," max: ",X_tb.max()," mean: ",X_tb.mean()," sd: ",X_tb.std())

    y_t = data_dict["data_hidden"]["cluster_labels"]

    print("#")
    print("X_ts.shape: ",X_ts.shape)
    print("X_sb.shape: ",X_sb.shape)
    print("X_tb.shape: ",X_tb.shape)
    print("#")
    print("y_t.shape: ",len(y_t))
    print("#")


    dict_e_size = {}
    dict_e_size["t"] = X_ts.shape[0]
    dict_e_size["s"] = X_ts.shape[1]
    dict_e_size["b"] = X_sb.shape[1]

    dict_e_code = {}
    dict_e_code["t"] = 0
    dict_e_code["s"] = 1
    dict_e_code["b"] = 2

    dict_e_code_reverse = {}
    dict_e_code_reverse[0] = "t"
    dict_e_code_reverse[1] = "s"
    dict_e_code_reverse[2] = "b"

    # X_ts = torch.from_numpy(X_ts).float()
    # X_sb = torch.from_numpy(X_sb).float()
    # X_tb = torch.from_numpy(X_tb).float()

    # if is_gpu:
    #     X_ts = X_ts.cuda()
    #     X_sb = X_sb.cuda()
    #     X_tb = X_tb.cuda()

    dict_num_clusters =  {"t":3,"s":3,"b":3}

    # dict_e_loss_weight = {
    #                         "t":1.0,\
    #                         "s":1.0,\
    #                         "b":1.0}
    # dict_loss_weight = {
    #                         "aec":1.0,
    #                         "mat":1.0,
    #                         "clust":1.0
    #                     }
    

    G = {
        "t":["X_ts","X_tb"],\
        "s":["X_ts","X_sb"],\
        "b":["X_tb","X_sb"]}

    X_data = {
        "X_ts":X_ts, \
        "X_tb":X_tb, \
        "X_sb":X_sb
    }

    X_meta = {
        "X_ts":["t","s"],\
        "X_tb":["t","b"],\
        "X_sb":["s","b"]
    }

    # X_dtype = {
    #     "X_ts":"real", \
    #     "X_tb":"real",\
    #     "X_sb":"real"
    # }

    y_val_dict = {}
    y_val_dict["t"] = np.array(y_t)
    #y_val_dict["b"] = np.array([])
    #y_val_dict["s"] = np.array([])

    #cfrm
    entity_size_dict = dict_e_size
    num_clusters_dict = dict_num_clusters
    max_count = 10000
    num_entities = len(list(G.keys()))
    num_rels = len(list(X_data.keys()))
    entity_list = list(G.keys())
    entity_rel_list = list(X_data.keys())
    e_pair_dict = X_meta
    #
    R = {}
    for x_id in X_data.keys():
        R[x_id] = X_data[x_id]

elif dataset_id in ["ade1","ade2","ade3","ade4"]:

    is_fac_pp = False

    #
    print("#")
    print("is_fac_pp: ",is_fac_pp)
    print("#")
    
    dict_mat = data_dict["matrices_data"]
    X_pd = dict_mat["mat_pat_dis_treat"]
    X_pr = dict_mat["mat_pat_drugs"]
    X_rd = dict_mat["mat_drugs_dis_side"]

    print("#")
    print("Before pp:")
    print("#")
    print("X_pd:")
    print("min: ",X_pd.min()," max: ",X_pd.max()," mean: ",X_pd.mean()," sd: ",X_pd.std())
    print("X_pr:")
    print("min: ",X_pr.min()," max: ",X_pr.max()," mean: ",X_pr.mean()," sd: ",X_pr.std())
    print("X_rd:")
    print("min: ",X_rd.min()," max: ",X_rd.max()," mean: ",X_rd.mean()," sd: ",X_rd.std())

    X_data_bef_pp = {
        "X_pd":X_pd, \
        "X_pr":X_pr,
        "X_rd":X_rd}

    scaler1 = StandardScaler()
    X_pd = scaler1.fit_transform(X_pd)
    scaler2 = StandardScaler()
    X_pr = scaler2.fit_transform(X_pr)
    scaler3 = StandardScaler()
    X_rd = scaler3.fit_transform(X_rd)

    # In[16]:

    print("#")
    print("After pp:")
    print("#")
    print("X_pd:")
    print("min: ",X_pd.min()," max: ",X_pd.max()," mean: ",X_pd.mean()," sd: ",X_pd.std())
    print("X_pr:")
    print("min: ",X_pr.min()," max: ",X_pr.max()," mean: ",X_pr.mean()," sd: ",X_pr.std())
    print("X_rd:")
    print("min: ",X_rd.min()," max: ",X_rd.max()," mean: ",X_rd.mean()," sd: ",X_rd.std())

    print("#")
    print("X_pd.shape: ",X_pd.shape)
    print("X_pr.shape: ",X_pr.shape)
    print("X_rd.shape: ",X_rd.shape)
    print("#")

    dict_e_size = {}
    dict_e_size["p"] = X_pd.shape[0]
    dict_e_size["r"] = X_pr.shape[1]
    dict_e_size["d"] = X_rd.shape[1]

    dict_e_code = {}
    dict_e_code["p"] = 0
    dict_e_code["r"] = 1
    dict_e_code["d"] = 2

    dict_e_code_reverse = {}
    dict_e_code_reverse[0] = "p"
    dict_e_code_reverse[1] = "r"
    dict_e_code_reverse[2] = "d"

    if data_version_name in ["1"]:
        dict_num_clusters =  {'p': 4, 'r': 10, 'd': 10}
    elif data_version_name in ["2"]:
        dict_num_clusters =  {'p': 5, 'r': 10, 'd': 10}
    elif data_version_name in ["3"]:
        dict_num_clusters =  {'p': 4, 'r': 8, 'd': 10}
    elif data_version_name in ["4"]:
        dict_num_clusters =  {'p': 5, 'r': 10, 'd': 9} 
    elif data_version_name in ["5"]:
        dict_num_clusters =  {'p': 5, 'r': 10, 'd': 9} 
    else:
        assert False,"Unknown adr_data_version_name: "+data_version_name


    G = {
        "p":["X_pr","X_pd"],\
        "r":["X_pr","X_rd"],\
        "d":["X_rd","X_pd"]}

    X_data = {
        "X_pr":X_pr, \
        "X_pd":X_pd, \
        "X_rd":X_rd
        }

    X_meta = {
        "X_pr":["p","r"],
        "X_pd":["p","d"],
        "X_rd":["r","d"]}

    X_dtype = {
        "X_pr":"real", \
        "X_pd":"real", \
        "X_rd":"real"
    }
    #
    y_val_dict = {}
    y_val_dict["p"] = np.array([])
    y_val_dict["r"] = np.array([])
    y_val_dict["d"] = np.array([])

    #cfrm
    entity_size_dict = dict_e_size
    num_clusters_dict = dict_num_clusters
    max_count = 10000
    num_entities = len(list(G.keys()))
    num_rels = len(list(X_data.keys()))
    entity_list = list(G.keys())
    entity_rel_list = list(X_data.keys())
    e_pair_dict = X_meta
    #
    R = {}
    for x_id in X_data.keys():
        R[x_id] = X_data[x_id]            

elif dataset_id == "freebase":
    is_fac_pp = True
    print("is_fac_pp: ",is_fac_pp)
    print("#")

   #
    list_x_id = list(data_dict["matrices"].keys())
    list_e_id = list(data_dict["metadata"]["dict_e_size"].keys())
    #
    #list_x_id = data_dict["list_x_id"]
    #list_e_id = data_dict["list_e_id"]
    #
    print("Loading the matrices: ")
    dict_id_X = {}
    X_data_bef_pp = {}
    for x_id in list_x_id:
        dict_id_X[x_id] = data_dict["matrices"][x_id]
        X_data_bef_pp[x_id] = dict_id_X[x_id]
        print("x_id: ",x_id," X.shape: ",dict_id_X[x_id].shape)
    print("#")
    #
    print("#")
    print("Before pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",dict_id_X[temp_id].min()," max: ",dict_id_X[temp_id].max()," mean: ",np.round(dict_id_X[temp_id].mean(),4)," sd: ",np.round(dict_id_X[temp_id].std(),4))
    print("#")
    #
    print("Preprocessing the matrices: ")
    dict_id_X_pp = {}
    X_data_size_fac = {}
    for temp_id in list_x_id:
        if is_fac_pp:
            _, dict_id_X_pp[temp_id] = get_size_fac_pp(dict_id_X[temp_id])
        else:
            scaler1 = StandardScaler()
            dict_id_X_pp[temp_id] = scaler1.fit_transform(dict_id_X[temp_id])
        print("temp_id: ",temp_id," X.shape: ",dict_id_X_pp[temp_id].shape)        
    print("#")
    #
    print("#")
    print("After pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",np.round(dict_id_X_pp[temp_id].min(),4)," max: ",np.round(dict_id_X_pp[temp_id].max(),4)," mean: ",np.round(dict_id_X_pp[temp_id].mean(),4)," sd: ",np.round(dict_id_X_pp[temp_id].std(),4))
    print("#")
    #
    X_meta = data_dict["metadata"]["x_meta"] 
    #
    G = {}
    for e_id in list_e_id:
        temp_list = []
        for x_id in X_meta.keys():
            if e_id in X_meta[x_id]:
                temp_list.append(x_id)
        G[e_id] = temp_list
    #
    print("#")
    print("G: ")
    print("#")
    print(G)
    print("#")
    #
    y_val_dict = {}
    y_val_dict["e1"] = np.array(data_dict["gt"]["list_e1_labels"])
    # for e_id in G.keys():
    #     if e_id in ["e1"]:
    #         y_val_dict[e_id] = np.array(data_dict["gt"]["list_e1_labels"])
    #     else:
    #         y_val_dict[e_id] = np.array([])

    #
    X_data = {}
    for x_id in dict_id_X_pp:
        X_data[x_id] = dict_id_X_pp[x_id]
        print("X"+x_id,", shape: ",X_data[x_id].shape)
    #
    dict_e_size = {}
    for e_id in G.keys():
        x_id = G[e_id][0]
        if X_meta[x_id][0] == e_id:
            dict_e_size[e_id] = X_data[x_id].shape[0]
        else:
            dict_e_size[e_id] = X_data[x_id].shape[1]
    #
    print("#")
    print("dict_e_size: ")
    print("#")
    for e_id in dict_e_size.keys():
        print("e_id: ",e_id,", size: ",dict_e_size[e_id])
    print("#")
    #
    num_cluster = 8 #books has 8 classes #cluster all entities to have the same number of clusters
    dict_num_clusters = {}
    dict_e_loss_weight = {}
    for e_id in list_e_id:
        dict_num_clusters[e_id] = num_cluster
        dict_e_loss_weight[e_id] = 1.0
    #
    dict_loss_weight = {
                        "aec":1.0,
                        "mat":1.0,
                        "clust":1.0
                    }
    #
    X_dtype = {}
    for x_id in X_data:
        X_dtype[x_id] = "real" 

    #
    #cfrm
    entity_size_dict = dict_e_size
    num_clusters_dict = dict_num_clusters
    max_count = 10000
    num_entities = len(list(G.keys()))
    num_rels = len(list(X_data.keys()))
    entity_list = list(G.keys())
    entity_rel_list = list(X_data.keys())
    e_pair_dict = X_meta
    #
    R = {}
    for x_id in X_data.keys():
        R[x_id] = X_data[x_id]    

elif dataset_id in ["pubmed","pubmed_heuristic"]:
    is_fac_pp = False
    print("is_fac_pp: ",is_fac_pp)
    print("#")

    list_x_id = list(data_dict["matrices"].keys())
    list_e_id = list(data_dict["metadata"]["dict_e_size"].keys())
    #
    #list_x_id = data_dict["list_x_id"]
    #list_e_id = data_dict["list_e_id"]
    #
    print("Loading the matrices: ")
    dict_id_X = {}
    X_data_bef_pp = {}
    for x_id in list_x_id:
        dict_id_X[x_id] = data_dict["matrices"][x_id]
        X_data_bef_pp[x_id] = dict_id_X[x_id]
        print("x_id: ",x_id," X.shape: ",dict_id_X[x_id].shape)
    print("#")
    #
    print("#")
    print("Before pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",dict_id_X[temp_id].min()," max: ",dict_id_X[temp_id].max()," mean: ",np.round(dict_id_X[temp_id].mean(),4)," sd: ",np.round(dict_id_X[temp_id].std(),4))
    print("#")
    #
    print("Preprocessing the matrices: ")
    dict_id_X_pp = {}
    X_data_size_fac = {}
    for temp_id in list_x_id:
        if is_fac_pp:
            _, dict_id_X_pp[temp_id] = get_size_fac_pp(dict_id_X[temp_id])
        else:
            scaler1 = StandardScaler()
            dict_id_X_pp[temp_id] = scaler1.fit_transform(dict_id_X[temp_id])
        print("temp_id: ",temp_id," X.shape: ",dict_id_X_pp[temp_id].shape)        
    print("#")
    #
    print("#")
    print("After pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",np.round(dict_id_X_pp[temp_id].min(),4)," max: ",np.round(dict_id_X_pp[temp_id].max(),4)," mean: ",np.round(dict_id_X_pp[temp_id].mean(),4)," sd: ",np.round(dict_id_X_pp[temp_id].std(),4))
    print("#")
    #
    X_meta = data_dict["metadata"]["x_meta"] 
    #
    G = {}
    for e_id in list_e_id:
        temp_list = []
        for x_id in X_meta.keys():
            if e_id in X_meta[x_id]:
                temp_list.append(x_id)
        G[e_id] = temp_list
    #
    print("#")
    print("G: ")
    print("#")
    print(G)
    print("#")
    #
    y_val_dict = {}
    for e_id in G.keys():
        if e_id in ["e2"]:
            y_val_dict[e_id] = np.array(data_dict["gt"]["list_e2_labels"])
    #
    X_data = {}
    for x_id in dict_id_X_pp:
        X_data[x_id] = dict_id_X_pp[x_id]
        print("X"+x_id,", shape: ",X_data[x_id].shape)
    #
    dict_e_size = {}
    for e_id in G.keys():
        x_id = G[e_id][0]
        if X_meta[x_id][0] == e_id:
            dict_e_size[e_id] = X_data[x_id].shape[0]
        else:
            dict_e_size[e_id] = X_data[x_id].shape[1]
    #
    print("#")
    print("dict_e_size: ")
    print("#")
    for e_id in dict_e_size.keys():
        print("e_id: ",e_id,", size: ",dict_e_size[e_id])
    print("#")
    #
    num_cluster = 8 #books has 8 classes #cluster all entities to have the same number of clusters
    dict_num_clusters = {}
    dict_e_loss_weight = {}
    for e_id in list_e_id:
        dict_num_clusters[e_id] = num_cluster
        dict_e_loss_weight[e_id] = 1.0
    #
    dict_loss_weight = {
                        "aec":1.0,
                        "mat":1.0,
                        "clust":1.0
                    }
    #
    X_dtype = {}
    for x_id in X_data:
        X_dtype[x_id] = "real" 

    #
    #cfrm
    entity_size_dict = dict_e_size
    num_clusters_dict = dict_num_clusters
    max_count = 10000
    num_entities = len(list(G.keys()))
    num_rels = len(list(X_data.keys()))
    entity_list = list(G.keys())
    entity_rel_list = list(X_data.keys())
    e_pair_dict = X_meta
    #
    R = {}
    for x_id in X_data.keys():
        R[x_id] = X_data[x_id]  

elif dataset_id == "genephene":
    is_fac_pp = False
    print("is_fac_pp: unused")
    print("#")

    list_x_id = list(data_dict["matrices"].keys())
    list_e_id = list(data_dict["metadata"]["dict_e_size"].keys())
    #
    #list_x_id = data_dict["list_x_id"]
    #list_e_id = data_dict["list_e_id"]
    #
    print("Loading the matrices: ")
    dict_id_X = {}
    X_data_bef_pp = {}
    for x_id in list_x_id:
        dict_id_X[x_id] = data_dict["matrices"][x_id]
        X_data_bef_pp[x_id] = dict_id_X[x_id]
        print("x_id: ",x_id," X.shape: ",dict_id_X[x_id].shape)
    print("#")

    def get_nz(X):
        nz_num = float(np.sum(X > 0))
        tot_num = float(np.prod(X.shape))
        nz_per = np.round((nz_num/tot_num) * 100.0,4)
        return nz_per

    print("#")
    print("nz %")
    print("#")
    for x_id in list_x_id:
        print("x_id: ",x_id," nz %: ",get_nz(data_dict["matrices"][x_id]))
    print("#")

    #
    print("#")
    print("Before pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",dict_id_X[temp_id].min()," max: ",dict_id_X[temp_id].max()," mean: ",np.round(dict_id_X[temp_id].mean(),4)," sd: ",np.round(dict_id_X[temp_id].std(),4))
    print("#")
    #
    print("Preprocessing the matrices: ")
    dict_id_X_pp = {}
    X_data_size_fac = {}
    epsilon = 1e-9
    for temp_id in list_x_id:
        # if is_fac_pp:
        #     _, dict_id_X_pp[temp_id] = get_size_fac_pp(dict_id_X[temp_id])
        # else:
        scaler1 = StandardScaler()
        #
        X_temp = dict_id_X[temp_id]
        X_temp = pd.DataFrame(X_temp).fillna(0).as_matrix()
        X_temp = np.log(X_temp + 1.0) + epsilon
        X_temp = np.nan_to_num(X_temp)
        #
        dict_id_X_pp[temp_id] = scaler1.fit_transform(X_temp)
        #dict_id_X_pp[temp_id] = scaler1.fit_transform(dict_id_X[temp_id])
        print("temp_id: ",temp_id," X.shape: ",dict_id_X_pp[temp_id].shape)        
    print("#")
    #
    print("#")
    print("After pp:")
    print("#")
    for temp_id in list_x_id:
        print("X_"+str(temp_id),": min: ",np.round(dict_id_X_pp[temp_id].min(),4)," max: ",np.round(dict_id_X_pp[temp_id].max(),4)," mean: ",np.round(dict_id_X_pp[temp_id].mean(),4)," sd: ",np.round(dict_id_X_pp[temp_id].std(),4))
    print("#")
    #
    X_meta = data_dict["metadata"]["x_meta"] 
    #
    G = {}
    for e_id in list_e_id:
        temp_list = []
        for x_id in X_meta.keys():
            if e_id in X_meta[x_id]:
                temp_list.append(x_id)
        G[e_id] = temp_list
    #
    print("#")
    print("G: ")
    print("#")
    print(G)
    print("#")
    #
    y_val_dict = {}
    for e_id in G.keys():
        if e_id in ["p"]:
            if data_version_name in ["1"]:
                y_val_dict[e_id] = np.array(data_dict["gt"]["list_p_vitality_status_labels"])
            elif data_version_name in ["2"]:
                y_val_dict[e_id] = np.array(data_dict["gt"]["list_p_cancer_stage_labels"])
            else:
                assert False,"Unknown data_version_name: "+str(data_version_name)+" for dataset_id: "+str(dataset_id)
        # else:
        #     y_val_dict[e_id] = np.array([])
    #
    X_data = {}
    for x_id in dict_id_X_pp:
        X_data[x_id] = dict_id_X_pp[x_id]
        print("X"+x_id,", shape: ",X_data[x_id].shape)
    #
    dict_e_size = {}
    for e_id in G.keys():
        x_id = G[e_id][0]
        if X_meta[x_id][0] == e_id:
            dict_e_size[e_id] = X_data[x_id].shape[0]
        else:
            dict_e_size[e_id] = X_data[x_id].shape[1]
    #
    print("#")
    print("dict_e_size: ")
    print("#")
    for e_id in dict_e_size.keys():
        print("e_id: ",e_id,", size: ",dict_e_size[e_id])
    print("#")
    #

    if data_version_name in ["1"]:
        num_cluster = 2 
    elif data_version_name in ["2"]:
        num_cluster = 4
    else:
        assert False,"Unknown data_version_name: "+str(data_version_name)+" for dataset_id: "+str(dataset_id)

    dict_num_clusters = {}
    dict_e_loss_weight = {}
    for e_id in list_e_id:
        dict_num_clusters[e_id] = num_cluster
        dict_e_loss_weight[e_id] = 1.0
    #
    dict_loss_weight = {
                        "aec":1.0,
                        "mat":1.0,
                        "clust":1.0
                    }
    #
    X_dtype = {}
    for x_id in X_data:
        X_dtype[x_id] = "real"  

    #cfrm
    entity_size_dict = dict_e_size
    num_clusters_dict = dict_num_clusters
    max_count = 10000
    num_entities = len(list(G.keys()))
    num_rels = len(list(X_data.keys()))
    entity_list = list(G.keys())
    entity_rel_list = list(X_data.keys())
    e_pair_dict = X_meta
    #
    R = {}
    for x_id in X_data.keys():
        R[x_id] = X_data[x_id]  

else:
    assert False

print("#")
print("dataset_id: ",dataset_id)
print("dict_num_clusters: ")
print(dict_num_clusters)
print("#")

#utilities - start
#init C with orth matrices
def get_orth_mat(m,n):
    a = np.random.randn(m, n)
    #print("a.shape: ",a.shape)
    q,_ = np.linalg.qr(a,mode='reduced')
    #print("q.shape: ",q.shape)
    assert np.sum(np.round(np.diag(q.T.dot(q)))==1) == q.T.shape[0]
    return q

def filter_rel(e,entity_rel_list,e_pair_dict):
    temp_list = []
    for rel in entity_rel_list:
        if e in e_pair_dict[rel]:
            temp_list.append(rel)
    return temp_list
            
def get_M(dict_R,dict_C,e,entity_rel_list):
    print("#")
    print("Calculating M...")
    print("e: ",e, ", size: ",entity_size_dict[e])
    rel_pairs = []
    rel_pairs = filter_rel(e,entity_rel_list,e_pair_dict) 
    # if e is "c1":
    #     rel_pairs = filter_rel(e,entity_rel_list) 
    # elif e is "c2":
    #     rel_pairs = filter_rel(e,entity_rel_list)
    # elif e is "g":
    #     rel_pairs = filter_rel(e,entity_rel_list)
    # elif e is "r":
    #     rel_pairs = filter_rel(e,entity_rel_list)
    print("rel_pairs: ",rel_pairs)
    M_list = []
    for e_pair in rel_pairs:
        print("processing pair: ",e_pair)
        temp = None
        if e == e_pair_dict[e_pair][0]:
            temp = np.dot(np.dot(np.dot(dict_R[e_pair],dict_C[e_pair_dict[e_pair][1]]),dict_C[e_pair_dict[e_pair][1]].T),dict_R[e_pair].T)
        elif e ==  e_pair_dict[e_pair][1]:
            temp = np.dot(np.dot(np.dot(dict_R[e_pair].T,dict_C[e_pair_dict[e_pair][0]]),dict_C[e_pair_dict[e_pair][0]].T),dict_R[e_pair])
        else:
            assert False
        print("M_temp.shape: ",temp.shape)
        M_list.append(temp)
    M = np.sum(M_list,axis=0)
    print("M.shape: ",M.shape)
    print("#")
    assert M.shape[0]== entity_size_dict[e]
    assert M.shape[1]== entity_size_dict[e]
    return M

def get_eigen(M,k):
    print("start: get_eigen")
    #w,v = eigh(M,eigvals=(M.shape[0]-k,M.shape[0]-1))
    w,v = sp.sparse.linalg.eigsh(M, k=k)
    print("end: get_eigen")
    return v

def get_loss(dict_R,dict_C,entity_rel_list):
    print("#")
    print("Computing loss...")
    loss = 0
    for e_pair in entity_rel_list:
        print("e_pair: ",e_pair)
        X = dict_R[e_pair]
        A_pair = np.dot(dict_C[e_pair_dict[e_pair][0]].T,np.dot(dict_R[e_pair],dict_C[e_pair_dict[e_pair][1]]))
        X_pred = np.dot(np.dot(dict_C[e_pair_dict[e_pair][0]],A_pair),dict_C[e_pair_dict[e_pair][1]].T)
        loss_cur = np.square(la.norm(X-X_pred, 'fro'))
        print("loss_cur: ",loss_cur)
        loss+=loss_cur
    print("loss: ",1.0 * loss)
    print("#")
    return 1.0 * loss

def get_loss_M(dict_M,dict_C,entity_list):
    #print("#")
    print("Computing loss...")
    loss = 0
    for e_id in entity_list:
        #print("e_id: ",e_id)
        loss_cur = np.trace(np.dot(np.dot(dict_C[e_id].T,dict_M[e_id]),dict_C[e_id]))
        #print("loss_cur: ",loss_cur)
        loss+=loss_cur
    print("loss: ",loss)
    print("#")
    #return (1.0 * -loss)
    return (1.0 * loss)

def get_c_ind_mat(C,entity_list,num_clusters_dict):
    C_ind = {}
    dict_pred_labels = {}
    for e in entity_list:
        print("e: ",e)
        print("C[e].shape: ",C[e].shape)
        list_labels = []
        kmc = KMeans(n_clusters=num_clusters_dict[e])
        labels = kmc.fit_predict(C[e])
        #
        C_ind_temp = np.zeros(C[e].shape)
        for i in np.arange(C[e].shape[0]):
            k = labels[i]
            C_ind_temp[i,k] = 1
            list_labels.append(k)
        #
        C_ind[e] = C_ind_temp
        dict_pred_labels[e] = list_labels
    return C_ind, dict_pred_labels

#utilities - end


#CFRM - start
# C = {}
# C["t"] = get_orth_mat(entity_size_dict["t"],num_clusters_dict["t"])
# C["t2"] = get_orth_mat(entity_size_dict["t2"],num_clusters_dict["t2"])
# C["s"] = get_orth_mat(entity_size_dict["s"],num_clusters_dict["s"])
# C["b"] = get_orth_mat(entity_size_dict["b"],num_clusters_dict["b"])

C = {}
for e_id in G.keys():
    C[e_id] = get_orth_mat(entity_size_dict[e_id],num_clusters_dict[e_id])

for e in C.keys():
    print("e: ",e,", C[e].shape: ",C[e].shape)
print("#")

#init M 

M = {}
#C = {}
for e in entity_list:
    M[e]=None
    #C[e]=None

count = 1
max_count = 1000
is_converged = False
epsilon = 0.00001
is_tr_loss = True
#is_tr_loss = False
# if is_tr_loss: 
#     L = get_loss_M(M,C,entity_list)
# else:
#     L = get_loss(R,C,entity_rel_list)
L_prev = np.Inf
while not is_converged:
    print("#")
    print("count: ",count)
    print("#")
    if count > max_count:
        break
    #
    for e in entity_list:
        print("e: ",e)
        #Compute M_rel
        M[e] = get_M(R,C,e,entity_rel_list)
        #udate C_rel
        C[e] = get_eigen(M[e],num_clusters_dict[e])
    #
    if is_tr_loss: 
        L = get_loss_M(M,C,entity_list)
    else:
        L = get_loss(R,C,entity_rel_list)
    count+=1
    #update convg status
    if epsilon > 0:
        if L_prev - L < epsilon:
            print("#Converged#")
            is_converged = True
    L_prev = L

print("#")
print("count: ",count)
print("L: ",L)
print("L_prev: ",L_prev)
print("#")

#transform C to C indicator matrix

C_ind, dict_pred_labels = get_c_ind_mat(C,entity_list,num_clusters_dict)
A = {}
for e_pair in entity_rel_list:
    print("e_pair: ",e_pair)
    A_pair = np.dot(C[e_pair_dict[e_pair][0]].T,np.dot(R[e_pair],C[e_pair_dict[e_pair][1]]))
    A[e_pair] = A_pair
    print("A_pair.shape: ",A_pair.shape)

print("#")
print("C_ind.keys(): ")
print(C_ind.keys())
print("#")
print("A.keys(): ")
print(A.keys())
print("#")

#compute X reconstructions X_prime

X_prime = {}
for e_pair in entity_rel_list:
    print("e_pair: ",e_pair)
    cur_x_prime = np.dot(C_ind[e_pair_dict[e_pair][0]],np.dot(A[e_pair],C_ind[e_pair_dict[e_pair][1]].T))
    X_prime[e_pair] = cur_x_prime

#CFRM - end


#https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation
import numpy as np
from scipy.misc import comb
def rand_index_score(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


#ARI
print("#")
dict_ari = {}
for e_id in y_val_dict.keys():
    print("e_id: ",e_id)
    #
    if np.sum(y_val_dict[e_id] < 0) > 0: #labels contain -ve values => ignore them from performance computation
        y_true_e_all = y_val_dict[e_id]
        y_pred_e_all = np.array(dict_pred_labels[e_id])
        #filter ytrue -> only non-negative values
        print("type(y_true_e_all): ",type(y_true_e_all))
        print("type(y_pred_e_all): ",type(y_pred_e_all))
        y_true_e = y_true_e_all[y_true_e_all >= 0]
        y_pred_e = y_pred_e_all[y_true_e_all >= 0]
        assert y_true_e.shape[0] == y_pred_e.shape[0]
        print("#")
        print("Ground truth labels contain -ve values. Considering only +ve labels for performance computation.")
        print("Reduced len(y_true_e_all): ",y_true_e_all.shape[0]," to len(y_true_e): ",y_true_e.shape[0])
        print("Reduced len(y_pred_e_all): ",y_pred_e_all.shape[0]," to len(y_pred_e): ",y_pred_e.shape[0])
        print("#")
    else:
        y_true_e = y_val_dict[e_id]
        y_pred_e = dict_pred_labels[e_id] 
        #
        y_true_e_all = y_val_dict[e_id]
        y_pred_e_all = np.array(dict_pred_labels[e_id])
    #
    if len(y_true_e) > 0:
        cur_ari = adjusted_rand_score(y_true_e, y_pred_e)
        cur_ami = adjusted_mutual_info_score(y_pred_e, y_true_e)
        cur_nmi = normalized_mutual_info_score(y_pred_e, y_true_e)
        cur_ri = rand_index_score(y_true_e, y_pred_e) 
        # cur_ari = adjusted_rand_score(y_val_dict[e_id],dict_pred_labels[e_id])
        # cur_ami = adjusted_mutual_info_score(dict_pred_labels[e_id], y_val_dict[e_id])
        # cur_nmi = normalized_mutual_info_score(dict_pred_labels[e_id], y_val_dict[e_id])
        # cur_ri = rand_index_score(y_val_dict[e_id],dict_pred_labels[e_id]) 
        dict_ari[e_id] = cur_ari
        print("###")
        print("Clustering Results: ")
        print("###")
        print("ARI: ",np.round(cur_ari,6))
        print("AMI: ",np.round(cur_ami,6))
        print("NMI: ",np.round(cur_nmi,6))
        print("RI: ",np.round(cur_ri,6))
        print("###")
    # print("y_true_e: ")
    # print(y_true_e)
    # print("#")
    # print("y_pred_e: ")
    # print(y_pred_e)
    # print("Cluster Size: ")
    # print("y_true_e: ")
    # print(collections.Counter(y_true_e_all))
    # print(collections.Counter(y_true_e))
    # print("y_pred_e: ")
    # print(collections.Counter(y_pred_e_all))
    # print(collections.Counter(y_pred_e))
#persist
pkl.dump(C_ind,open(out_dir+"dict_I.pkl","wb"))
pkl.dump(C,open(out_dir+"dict_C.pkl","wb"))
pkl.dump(A,open(out_dir+"dict_A.pkl","wb"))
pkl.dump(X_prime,open(out_dir+"dict_X_prime.pkl","wb"))
pkl.dump(dict_ari,open(out_dir+"dict_ari.pkl","wb"))
pkl.dump(dict_pred_labels,open(out_dir+"dict_pred_labels.pkl","wb"))



