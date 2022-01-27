import numpy as np
import random
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import NMF

np.random.seed(0)
random.seed(0)

#NN
def get_k_neigh_ind(X,k):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X)
    neigh_dist, neigh_ind = neigh.kneighbors(X) 
    return neigh_dist, neigh_ind

#Downsampling
def get_downsample_size(dict_U):
    list_entity_size = []
    for e_id in dict_U.keys():
        #print("e_id: ",e_id,", dict_U[e_id].shape: ",dict_U[e_id].shape)
        list_entity_size.append(dict_U[e_id].shape[0])
    sample_size = np.min(list_entity_size)
    return sample_size

def get_dict_U_downsampled(dict_U):
    downsample_size = get_downsample_size(dict_U)
    #
    dict_U_downsampled = {}
    for e_id in dict_U.keys():
        #print("e_id: ",e_id,", dict_U[e_id].shape: ",dict_U[e_id].shape)
        cur_entity_size = dict_U[e_id].shape[0]
        if cur_entity_size > downsample_size:
            #print("Downsampling e_id: ",e_id)
            cur_downsample_list_idx = random.sample(list(np.arange(cur_entity_size)),downsample_size)
            dict_U_downsampled[e_id] = dict_U[e_id][cur_downsample_list_idx]
        else:
            dict_U_downsampled[e_id] = dict_U[e_id]
    return dict_U_downsampled

def get_downsample_size_nmf(dict_U):
    list_entity_size = []
    for e_id in dict_U.keys():
        #print("e_id: ",e_id,", dict_U[e_id].shape: ",dict_U[e_id].shape)
        list_entity_size.append(dict_U[e_id][0].shape[0])
    sample_size = np.min(list_entity_size)
    return sample_size
    

def get_dict_U_downsampled_nmf(dict_U):
    downsample_size = get_downsample_size_nmf(dict_U)
    #
    dict_U_downsampled_nmf = {}
    for e_id in dict_U.keys():
        list_U = dict_U[e_id]
        list_U_downsampled = []
        for cur_U in list_U:
            #print("e_id: ",e_id,", cur_U.shape: ",cur_U.shape)
            cur_entity_size = cur_U.shape[0]
            if cur_entity_size > downsample_size:
                #print("Downsampling e_id: ",e_id)
                cur_downsample_list_idx = random.sample(list(np.arange(cur_entity_size)),downsample_size)
                list_U_downsampled.append(cur_U[cur_downsample_list_idx])
                
            else:
                list_U_downsampled.append(cur_U)
        dict_U_downsampled_nmf[e_id] = list_U_downsampled
    return dict_U_downsampled_nmf


#Batchid generation

def get_dict_U_batchids(dict_U_downsampled): 
    batch_no = 1
    dict_U_batchids = {}
    for e_id in dict_U_downsampled.keys():
        #print("e_id: ",e_id,", dict_U_downsampled[e_id].shape: ",dict_U_downsampled[e_id].shape)
        cur_entity_size = dict_U_downsampled[e_id].shape[0]
        dict_U_batchids[e_id] = np.array([batch_no for i in np.arange(cur_entity_size)])
        batch_no+=1
    return dict_U_batchids

def get_align_score(dict_U, k):
    #Down sample all U's to the min of the entity sizes
    dict_U_downsampled = get_dict_U_downsampled(dict_U)
    #Give each matrix a batch id and get the list of batchids of all the instances
    dict_U_batchids = get_dict_U_batchids(dict_U_downsampled)
    #
    #Get concatenated U and batchid list
    list_U = []
    list_batchids = []
    for e_id in dict_U_downsampled.keys():
        list_U.append(dict_U_downsampled[e_id])
        list_batchids.append(dict_U_batchids[e_id])
    #
    U_all = np.concatenate(list_U)
    batchids_all = np.concatenate(list_batchids)
    #
    #print("U_all.shape: ",U_all.shape)
    #print("batchids_all.shape: ",batchids_all.shape)
    #
    # Get k NN of all the entity instances
    neigh_dist, neigh_ind = get_k_neigh_ind(U_all,k=k)
    #
    batchids_unique = np.unique(batchids_all)
    assert len(batchids_unique) == len(list(dict_U.keys()))
    #print("batchids_unique: ",batchids_unique)
    #
    N = neigh_ind.shape[0]
    #compute batch match counts for each instance
    dict_idx_count = {}
    for i in np.arange(N):
        cur_idx_batchid = batchids_all[i]
        cur_neigh_idxs = neigh_ind[i]
        cur_neigh_batchids = batchids_all[cur_neigh_idxs]
        dict_idx_count[i] = np.sum(cur_neigh_batchids == cur_idx_batchid)
    #
    f = k/N
    x_bar = np.mean(list(dict_idx_count.values()))
    align_score = 1 - ((x_bar - f) / (k-f))
    return align_score

def get_eid_agree_score(dict_U, k, G, X_data_bef_pp, X_meta, nmf_k):
    #Down sample all U's to the min of the entity sizes
    dict_U_downsampled = get_dict_U_downsampled(dict_U)
    #Give each matrix a batch id and get the list of batchids of all the instances
    dict_U_batchids = get_dict_U_batchids(dict_U_downsampled)
    #
    #Get concatenated U and batchid list
    list_U = []
    list_batchids = []
    for e_id in dict_U_downsampled.keys():
        list_U.append(dict_U_downsampled[e_id])
        list_batchids.append(dict_U_batchids[e_id])

    #
    U_all = np.concatenate(list_U)
    batchids_all = np.concatenate(list_batchids)

    #
    #print("U_all.shape: ",U_all.shape)
    #print("batchids_all.shape: ",batchids_all.shape)

    #
    # G_all
    # Get k NN of all the entity instances
    neigh_dist_all, neigh_ind_all = get_k_neigh_ind(U_all,k=k)
    #
    batchids_unique = np.unique(batchids_all)
    assert len(batchids_unique) == len(list(dict_U.keys()))
    #print("batchids_unique: ",batchids_unique)

    #for each entity U matrix that got concatenated to form U_all
    #keep track of the idx before and after concatenation
    #nmf_k = 10

    dict_U_nmf = {}
    for e_id in dict_U_downsampled.keys():
        temp_list_U = []
        for x_id in G[e_id]:
            cur_X = X_data_bef_pp[x_id]
            model = NMF(n_components=nmf_k, init='random', random_state=0) #, max_iter=10000)
            W = model.fit_transform(cur_X)
            H = model.components_
            if X_meta[x_id][0] == e_id:
                #Take row factor W
                temp_list_U.append(W)
            else:
                #Take col factor H
                temp_list_U.append(H.T)
        dict_U_nmf[e_id] = temp_list_U

    #print("#")
    for e_id in dict_U_nmf:
        #print("len(dict_U_nmf[e_id]): ",len(dict_U_nmf[e_id]))
        #print("e_id: ",e_id," dict_U_nmf[e_id][0].shape: ",dict_U_nmf[e_id][0].shape)  
        dict_U_nmf_downsampled = get_dict_U_downsampled_nmf(dict_U_nmf)

    #print("#")
    for e_id in dict_U_nmf_downsampled:
        #print("len(dict_U_nmf_downsampled[e_id]): ",len(dict_U_nmf_downsampled[e_id]))
        #print("e_id: ",e_id," dict_U_nmf_downsampled[e_id][0].shape: ",dict_U_nmf_downsampled[e_id][0].shape)  
        cur_size = 0
        dict_map_common_idx = {}
        dict_eid_offset = {}
        for e_id in dict_U_nmf_downsampled.keys():
            list_idx_before_concat = np.arange(dict_U_nmf_downsampled[e_id][0].shape[0])
            list_idx_after_concat = [idx+cur_size for idx in list_idx_before_concat]
            dict_eid_offset[e_id] = cur_size
            cur_size+=dict_U_nmf_downsampled[e_id][0].shape[0]    
            dict_map_common_idx[e_id] = dict(zip(list_idx_before_concat, list_idx_after_concat))    

    #Get k NN for each batch separately
    dict_eid_neighidx = {}
    dict_eid_neighdist = {}
    for e_id in dict_U_nmf_downsampled.keys():
        dict_eid_neighdist[e_id] = []
        dict_eid_neighidx[e_id] = []
        for U_nmf in dict_U_nmf_downsampled[e_id]:
            temp_eid_neighdist, temp_eid_neighidx = get_k_neigh_ind(U_nmf,k=k)
            dict_eid_neighdist[e_id].append(temp_eid_neighdist)
            dict_eid_neighidx[e_id].append(temp_eid_neighidx)   
    #
    #foreach instance count the agreement counts between 
    #the separate graph & combined graph's 
    dict_eid_agree_score_list = {}
    for e_id in dict_U_nmf_downsampled.keys():
        dict_eid_agree_score_list[e_id] = []
        for j in np.arange(len(dict_eid_neighidx[e_id])):
            dict_idx_count = {}
            for i in np.arange(dict_eid_neighidx[e_id][j].shape[0]):
                # print("e_id: ",e_id)
                # print("i: ",i)
                # print("len(dict_map_common_idx[e_id]): ",len(dict_map_common_idx[e_id]))
                # print("dict_map_common_idx[e_id][i]",dict_map_common_idx[e_id][i])
                # print("len(neigh_ind_all): ",len(neigh_ind_all))
                cur_match_count = len(
                    set(
                        neigh_ind_all[dict_map_common_idx[e_id][i]]
                    ).intersection(
                        set(
                            dict_eid_neighidx[e_id][j][i]+dict_eid_offset[e_id]
                        )
                    )
                )
                dict_idx_count[i] = cur_match_count / float(k)
            dict_eid_agree_score_list[e_id].append(np.mean(list(dict_idx_count.values())))
    #
    #print("#")
    #print("dict_eid_agree_score_list: ")
    #print(dict_eid_agree_score_list)
    #print("#")
    dict_eid_agree_score = {}
    dict_eid_agree_score_std = {}
    for e_id in dict_eid_agree_score_list:
        dict_eid_agree_score[e_id] = np.mean(dict_eid_agree_score_list[e_id])
        dict_eid_agree_score_std[e_id] = np.std(dict_eid_agree_score_list[e_id])    
    #
    return dict_eid_agree_score, dict_eid_agree_score_std    

# def get_eid_agree_score(dict_U, k):
#     #Down sample all U's to the min of the entity sizes
#     dict_U_downsampled = get_dict_U_downsampled(dict_U)
#     #Give each matrix a batch id and get the list of batchids of all the instances
#     dict_U_batchids = get_dict_U_batchids(dict_U_downsampled)
#     #
#     #Get concatenated U and batchid list
#     list_U = []
#     list_batchids = []
#     for e_id in dict_U_downsampled.keys():
#         list_U.append(dict_U_downsampled[e_id])
#         list_batchids.append(dict_U_batchids[e_id])
#     #
#     U_all = np.concatenate(list_U)
#     batchids_all = np.concatenate(list_batchids)
#     #
#     print("U_all.shape: ",U_all.shape)
#     print("batchids_all.shape: ",batchids_all.shape)
#     #
#     # G_all
#     # Get k NN of all the entity instances
#     neigh_dist_all, neigh_ind_all = get_k_neigh_ind(U_all,k=k)
#     #
#     batchids_unique = np.unique(batchids_all)
#     assert len(batchids_unique) == len(list(dict_U.keys()))
#     print("batchids_unique: ",batchids_unique)
#     #for each entity U matrix that got concatenated to form U_all
#     #keep track of the idx before and after concatenation
#     cur_size = 0
#     dict_map_common_idx = {}
#     dict_eid_offset = {}
#     for e_id in dict_U_downsampled.keys():
#         list_idx_before_concat = np.arange(dict_U_downsampled[e_id].shape[0])
#         list_idx_after_concat = [idx+cur_size for idx in list_idx_before_concat]
#         dict_eid_offset[e_id] = cur_size
#         cur_size+=dict_U_downsampled[e_id].shape[0]    
#         dict_map_common_idx[e_id] = dict(zip(list_idx_before_concat, list_idx_after_concat))
#     #Get k NN for each batch separately
#     dict_eid_neighidx = {}
#     dict_eid_neighdist = {}
#     for e_id in dict_U_downsampled.keys():
#         dict_eid_neighdist[e_id], dict_eid_neighidx[e_id] = get_k_neigh_ind(dict_U_downsampled[e_id],k=k)
#     #foreach instance count the agreement counts between 
#     #the separate graph & combined graph's 
#     dict_eid_agree_score = {}
#     for e_id in dict_U_downsampled.keys():
#         dict_idx_count = {}
#         for i in np.arange(dict_eid_neighidx[e_id].shape[0]):
#             cur_match_count = len(set(neigh_ind_all[dict_map_common_idx[e_id][i]]).intersection(set(dict_eid_neighidx[e_id][i]+dict_eid_offset[e_id])))
#             dict_idx_count[i] = cur_match_count / float(k)
#         dict_eid_agree_score[e_id] = np.mean(list(dict_idx_count.values()))
#     #
#     return dict_eid_agree_score


def get_mean_entropy(dict_U, k):
    #Down sample all U's to the min of the entity sizes
    dict_U_downsampled = get_dict_U_downsampled(dict_U)
    #Give each matrix a batch id and get the list of batchids of all the instances
    dict_U_batchids = get_dict_U_batchids(dict_U_downsampled)
    #
    #Get concatenated U and batchid list
    list_U = []
    list_batchids = []
    for e_id in dict_U_downsampled.keys():
        list_U.append(dict_U_downsampled[e_id])
        list_batchids.append(dict_U_batchids[e_id])
    #
    U_all = np.concatenate(list_U)
    batchids_all = np.concatenate(list_batchids)
    #
    #print("U_all.shape: ",U_all.shape)
    #print("batchids_all.shape: ",batchids_all.shape)
    #
    # Get k NN of all the entity instances
    neigh_dist, neigh_ind = get_k_neigh_ind(U_all,k=k)
    #
    batchids_unique = np.unique(batchids_all)
    assert len(batchids_unique) == len(list(dict_U.keys()))
    #print("batchids_unique: ",batchids_unique)
    #
    #compute batch probabilities
    dict_idx_prob = {}
    for i in np.arange(neigh_ind.shape[0]):
        #
        dict_batchid_count = {}
        for cur_batch_id in batchids_unique:
            cur_nn_idx_list = neigh_ind[i]
            cur_batch_idx_list = batchids_all[cur_nn_idx_list]
            dict_batchid_count[cur_batch_id] = np.sum(cur_batch_idx_list == cur_batch_id)
        #sanity check
        assert k == np.sum(list(dict_batchid_count.values()))
        #
        dict_batchid_prob = {}
        for cur_batch_id in batchids_unique:
            dict_batchid_prob[cur_batch_id] = dict_batchid_count[cur_batch_id] / float(k)
        #
        dict_idx_prob[i] = dict_batchid_prob
    #
    #Compute entropy
    dict_idx_entropy = {}
    for idx in dict_idx_prob.keys():
        cur_dict_batchid_prob = dict_idx_prob[idx]
        #
        cur_ent = 0
        for cur_batch_id in cur_dict_batchid_prob.keys():
            p = cur_dict_batchid_prob[cur_batch_id]
            if p > 0:
                cur_ent+=(-p * np.log(p))
            else:
                cur_ent+=0
        #
        dict_idx_entropy[idx] = cur_ent
    #
    entropy_mean = np.mean(list(dict_idx_entropy.values()))
    #
    return entropy_mean