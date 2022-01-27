import numpy as np
import pickle as pkl
import pprint as pp

import sys
sys.path.append('./../')


from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import src.single_cell_metrics as scm

import community as community_louvain
import networkx as nx

from sklearn import preprocessing
from sklearn import metrics

from sklearn.neighbors import NearestNeighbors


class dcmtf_joint_clustering: 
    
    def __load_data(self):
        #self.dict_perf = pkl.load(open(fname_perf,"rb"))
        self.dict_u = pkl.load(open(self.fname_u,"rb"))
        self.dict_c = pkl.load(open(self.fname_c,"rb"))
        self.labels_pred_u = pkl.load(open(self.fname_u_pred_labels,"rb"))
        self.labels_pred_c = pkl.load(open(self.fname_c_pred_labels,"rb"))
        #
        self.U_c1 = self.dict_u["c1"]
        self.U_c2 = self.dict_u["c2"]
        self.C_c1 = self.dict_c["c1"]
        self.C_c2 = self.dict_c["c2"]

    def __print_params(self):
        pp.pprint("is_normalize: ",self.is_normalize)
        pp.pprint("k_nn_c1_graph : ",self.k_nn_c1_graph)
        pp.pprint("k_nn_c2_graph : ",self.k_nn_c2_graph)
        pp.pprint("knn_perf_metric : ",self.knn_perf_metric)
        pp.pprint("num_std_c1 : ",self.num_std_c1)
        pp.pprint("num_std_c2 : ",self.num_std_c2)
        pp.pprint("num_std_c12 : ",self.num_std_c12)
        pp.pprint("is_binary_adj_mat_c1 : ",self.is_binary_adj_mat_c1)
        pp.pprint("is_binary_adj_mat_c2 : ",self.is_binary_adj_mat_c2)
        pp.pprint("is_binary_adj_mat_c12 : ",self.is_binary_adj_mat_c12)
        pp.pprint("resolution_c1 : ",self.resolution_c1)
        pp.pprint("resolution_c2 : ",self.resolution_c2)
        pp.pprint("resolution_joint_c12 : ",self.resolution_joint_c12)
        pp.pprint("is_use_both_UC : ",self.is_use_both_UC)
        pp.pprint("is_use_quant_UC : ",self.is_use_quant_UC)
        pp.pprint("out_dir_name : ",self.out_dir_name)
        pp.pprint("data_dir_name : ",self.data_dir_name)
        pp.pprint("#")
        pp.pprint("fname_u: ",self.fname_u)
        pp.pprint("fname_c: ",self.fname_c)
        pp.pprint("fname_u_pred_labels: ",self.fname_u_pred_labels)
        pp.pprint("fname_c_pred_labels: ",self.fname_c_pred_labels)
        pp.pprint("#")
        pp.pprint("G : ",self.G)
        pp.pprint("X_data_bef_pp : ",self.X_data_bef_pp)
        pp.pprint("X_meta : ",self.X_meta)
        pp.pprint("y_val_dict : ",self.y_val_dict)
        pp.pprint("dict_num_clusters : ",self.dict_num_clusters)
        pp.pprint("dict_e_siz : ",self. dict_e_siz)
        pp.pprint("#")
        
    
    def __init__(self,\
                is_normalize, \
                k_nn_c1_graph, k_nn_c2_graph, knn_perf_metric, \
                num_std_c1, num_std_c2, num_std_c12, \
                is_binary_adj_mat_c1, is_binary_adj_mat_c2, is_binary_adj_mat_c12, \
                resolution_c1, resolution_c2, resolution_joint_c12, \
                is_use_both_UC, is_use_quant_UC, \
                out_dir_name, data_dir_name, \
                G, X_data_bef_pp, X_meta, y_val_dict,\
                dict_num_clusters, dict_e_size):
        #
        self.is_normalize = is_normalize
        #
        self.k_nn_c1 = k_nn_c1_graph
        self.k_nn_c2 = k_nn_c2_graph
        self.knn_perf = knn_perf_metric
        #
        self.num_std_c1 = num_std_c1 #for thres computation in connecting c1-c1 graph
        self.num_std_c2 = num_std_c2 #for thres computation in connecting c2-c2 graph
        self.num_std_c12 = num_std_c12 #for thres computation in connecting c1-c2 graph
        #
        self.is_binary_adj_mat_c1 = is_binary_adj_mat_c1
        self.is_binary_adj_mat_c2 = is_binary_adj_mat_c2
        self.is_binary_adj_mat_c12 = is_binary_adj_mat_c12
        #
        self.resolution_c1 = resolution_c1
        self.resolution_c2 = resolution_c2
        self.resolution_joint_c12 = resolution_joint_c12
        #
        self.is_use_both_UC = is_use_both_UC
        #
        self.is_use_quant_UC = is_use_quant_UC
        #
        self.out_dir_name = out_dir_name
        self.data_dir_name = data_dir_name
        #
        self.fname_perf = self.out_dir_name + "../dict_setting_perf.pkl"
        #
        if self.is_use_quant_UC:
            self.fname_u = self.out_dir_name + "dict_U_quantile_ass2.pkl"
        else:
            self.fname_u = self.out_dir_name + "dict_U.pkl"
        #
        if self.is_use_quant_UC:
            self.fname_c = self.out_dir_name + "dict_I_ortho_quantile_ass2.pkl"
        else:
            self.fname_c = self.out_dir_name + "dict_I_ortho.pkl"
        #
        self.fname_u_pred_labels = self.out_dir_name + "dict_u_clust_labels.pkl"
        self.fname_c_pred_labels = self.out_dir_name + "dict_c_clust_labels.pkl"
        #
        self.G = G
        self.X_data_bef_pp = X_data_bef_pp
        self.X_meta = X_meta
        self.y_val_dict = y_val_dict
        self.dict_e_size = dict_e_size
        self.dict_num_clusters = dict_num_clusters
        #
        self.__load_data()
    
    def __get_y_pred_kmeans(self, dict_num_clusters, e_id, dict_U):
        kmeans_e = KMeans(n_clusters= dict_num_clusters[e_id], random_state=0)
        kmeans_e.fit(dict_U[e_id])
        labels_pred = kmeans_e.labels_
        return labels_pred

    def __get_cur_performance(self):
        print("__get_cur_performance: ")
        print("#")
        #C
        dict_temp_c = {}
        dict_temp_c["c1"] = self.dict_c["c1"]
        dict_temp_c["c2"] = self.dict_c["c2"]
        #U
        dict_temp_u = {}
        dict_temp_u["c1"] = self.dict_u["c1"]
        dict_temp_u["c2"] = self.dict_u["c2"]
        #ARI
        #c1
        y_pred_c1 = self.__get_y_pred_kmeans(self.dict_num_clusters,"c1",self.dict_c)
        self.ari_c_c1_orig = adjusted_rand_score(self.y_val_dict["c1"],y_pred_c1)
        #c2
        y_pred_c2 = self.__get_y_pred_kmeans(self.dict_num_clusters,"c2",self.dict_c)
        self.ari_c_c2_orig = adjusted_rand_score(self.y_val_dict["c2"],y_pred_c2)
        #
        print("ARI c1: ",self.ari_c_c1_orig)
        print("ARI c2: ",self.ari_c_c2_orig)
        print("#")
        #BME
        self.bme_c_orig = scm.get_mean_entropy(dict_temp_c, self.knn_perf)
        print("BME: ",self.bme_c_orig)
        print("#")
        #ALS
        self.align_c_orig = scm.get_align_score(dict_temp_c, self.knn_perf)
        print("ALS: ",self.align_c_orig)
        print("#")
        #AGS
        # self.dict_agree_score_orig, self.dict_agree_std_orig = scm.get_eid_agree_score(dict_temp_u, self.knn_perf, \
        #                                                            self.G, self.X_data_bef_pp, \
        #                                                            self.X_meta, nmf_k=self.knn_perf)

        # print("AGS c1: ",self.dict_agree_score_orig["c1"])
        # print("AGS c2: ",self.dict_agree_score_orig["c2"])
        # print("#")

    def do_lovaine_clust(self, k_nn_c2, C_c2, e_c2_size, I_c2, resolution, y_true_c2, num_std_c2, is_binary_adj_mat):
        print("do_lovaine_clust: ")
        #k_nn_c2 = 100
        nbrs_c2 = NearestNeighbors(n_neighbors=k_nn_c2, algorithm='ball_tree').fit(C_c2)
        distances_c2, indices_c2 = nbrs_c2.kneighbors(C_c2)
        fn_vec_mat_c2 = np.zeros((e_c2_size, k_nn_c2))
        #
        for cur_cell_idx_c2 in np.arange(e_c2_size):
            for nn_idx in indices_c2[cur_cell_idx_c2]:
                cur_fn_idx = np.argmax(I_c2[nn_idx])
                fn_vec_mat_c2[cur_cell_idx_c2][cur_fn_idx]+=1
        #
        D_c2 = metrics.pairwise_distances(fn_vec_mat_c2,metric='manhattan')
        print("np.sum(D_c > 0): ",np.sum(D_c2 > 0))
        #
        thres_c2 = np.mean(D_c2) + (num_std_c2 * np.std(D_c2))
        D_c2_bin_thres = (D_c2 < thres_c2) + 0
        print("thres_c: ",thres_c2)
        print("np.sum(D_c_bin_thres > 0): ",np.sum(D_c2_bin_thres > 0))
        print("is_binary_adj_mat: ",is_binary_adj_mat)
        #
        if not is_binary_adj_mat:
            D_c2_bin_thres = D_c2_bin_thres * D_c2
        #
        print("resolution_c: ",resolution)
        G_c2 = nx.from_numpy_matrix(D_c2_bin_thres)
        partition_c2 = community_louvain.best_partition(G_c2, resolution=resolution)
        #
        print("c #partition: ",len(np.unique(list(partition_c2.values()))))
        #
        y_pred_c2 = list(partition_c2.values())
        print("c ARI: ",adjusted_rand_score(y_true_c2,y_pred_c2))
        return D_c2_bin_thres, fn_vec_mat_c2
    
    def jointly_cluster(self):
        self.__get_cur_performance()
        #
        if self.is_use_both_UC:
            if self.is_normalize:
                pp_scaler = StandardScaler()
                U_c1 = pp_scaler.fit_transform(U_c1)
                pp_scaler = StandardScaler()
                U_c2 = pp_scaler.fit_transform(U_c2)
        #
        if self.is_normalize:
            pp_scaler = StandardScaler()
            C_c1 = pp_scaler.fit_transform(C_c1)
            pp_scaler = StandardScaler()
            C_c2 = pp_scaler.fit_transform(C_c2)
        #
        #get indicator matrices of clustering 1
        print("Obtain clustering 1's indicator matrices")
        y_pred_c1_c = self.__get_y_pred_kmeans(self.dict_num_clusters,"c1",self.dict_c)
        y_pred_c2_c = self.__get_y_pred_kmeans(self.dict_num_clusters,"c2",self.dict_c)
        #
        lb_c1 = preprocessing.LabelBinarizer()
        self.I_c1 = lb_c1.fit_transform(y_pred_c1_c)
        #
        lb_c2 = preprocessing.LabelBinarizer()
        self.I_c2 = lb_c2.fit_transform(y_pred_c2_c)
        #
        print("I_c1.shape: ",self.I_c1.shape)
        print("I_c2.shape: ",self.I_c2.shape)
        #
        print("#")
        ############ c1 ############ 
        print("C1 - graph building")
        print("---")
        k_nn_c = self.k_nn_c1
        if self.is_use_both_UC:
            C = self.U_c1
            print("Using embd: U_c1")
        else:
            print("Using embd: C_c1")
            C = self.C_c1
        #    
        e_size = self.dict_e_size["c1"]
        I = self.I_c1
        resolution = self.resolution_c1
        y_true_c = self.y_val_dict["c1"]
        num_std_c = self.num_std_c1
        is_binary_adj_mat = self.is_binary_adj_mat_c1
        #
        D_c1_bin_thres, fn_vec_mat_c1 = self.do_lovaine_clust(\
            k_nn_c, C, e_size, I, resolution, y_true_c, num_std_c, is_binary_adj_mat)
        print("#")
        ############ c2 ############ 
        print("C2 - graph building")
        print("---")
        k_nn_c = self.k_nn_c2
        if self.is_use_both_UC:
            C = self.U_c2
            print("Using embd: U_c2")
        else:
            print("Using embd: C_c2")
            C = self.C_c2
        #    
        e_size = self.dict_e_size["c2"]
        I = self.I_c2
        resolution = self.resolution_c2
        y_true_c = self.y_val_dict["c2"]
        num_std_c = self.num_std_c2
        is_binary_adj_mat = self.is_binary_adj_mat_c2
        #
        D_c2_bin_thres, fn_vec_mat_c2 = self.do_lovaine_clust(\
            k_nn_c, C, e_size, I, resolution, y_true_c, num_std_c, is_binary_adj_mat)
        print("#")
        #########
        print("C1 & C2 joint graph building")
        print("---")
        print("D_c1_bin_thres.shape: ",D_c1_bin_thres.shape)
        print("D_c2_bin_thres.shape: ",D_c2_bin_thres.shape)
        print("#")
        ############## c1 & c2 ############
        #
        D_c1_c2 = metrics.pairwise_distances(fn_vec_mat_c1,fn_vec_mat_c2,metric='manhattan')
        #
        print("fn_vec_mat_c1.shape: ",fn_vec_mat_c1.shape)
        print("fn_vec_mat_c2.shape: ",fn_vec_mat_c2.shape)
        print("#")
        print("D_c1_c2.shape: ",D_c1_c2.shape)
        print("np.sum(D_c1_c2 > 0): ",np.sum(D_c1_c2 > 0))
        #
        thres_c = np.mean(D_c1_c2) + (self.num_std_c12 * np.std(D_c1_c2))
        D_c1_c2_bin_thres = (D_c1_c2 < thres_c) + 0
        print("thres_c: ", thres_c)
        print("np.sum(D_c1_c2_bin_thres > 0): ",np.sum(D_c1_c2_bin_thres > 0))
        print("#")
        #
        if not self.is_binary_adj_mat_c12:
            D_c1_c2_bin_thres = D_c1_c2_bin_thres * D_c1_c2
        #
        D_11_12 = np.hstack([D_c1_bin_thres,D_c1_c2_bin_thres])
        D_21_22 = np.hstack([D_c1_c2_bin_thres.T, D_c2_bin_thres])
        D_joint = np.vstack([D_11_12, D_21_22])
        #
        print("D_11_12.shape: ",D_11_12.shape)
        print("D_21_22.shape: ",D_21_22.shape)
        print("D_joint.shape: ",D_joint.shape)
        print("D_joint: ")
        print(D_joint)
        print("#")
        print("Jointly_clustering using lovaine: ")
        #
        print("Building the graph...")
        G_c = nx.from_numpy_matrix(D_joint)
        #
        print("Partitioning... ")
        partition_c = community_louvain.best_partition(G_c, resolution=self.resolution_joint_c12)
        print("resolution_joint_c12: ",self.resolution_joint_c12)
        print("c #partition: ",len(np.unique(list(partition_c.values()))))
        #
        y_pred_c = list(partition_c.values())
        print("len(y_pred_c): ",len(y_pred_c))
        #
        y_pred_c1_joint = y_pred_c[0:self.dict_e_size["c1"]]
        y_pred_c2_joint = y_pred_c[self.dict_e_size["c1"]:self.dict_e_size["c1"]+self.dict_e_size["c2"]]
        #
        y_true_c1 = self.y_val_dict["c1"]
        y_true_c2 = self.y_val_dict["c2"]
        print("len(y_true_c1): ",len(y_true_c1))
        print("len(y_true_c2): ",len(y_true_c2))
        print("#")
        print("clustering 1's performance: ")
        print("ARI c1 orig: ",self.ari_c_c1_orig)
        print("ARI c2 orig: ",self.ari_c_c2_orig)
        print("#")
        print("clustering 2's performance: ")
        print("ARI c1: ",adjusted_rand_score(y_true_c1,y_pred_c1_joint))
        print("ARI c2: ",adjusted_rand_score(y_true_c2,y_pred_c2_joint))
        print("#")