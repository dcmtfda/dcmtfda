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
from scipy.sparse import csr_matrix, vstack, hstack


class dcmtf_joint_clustering: 
    
    def __load_data(self):
        #self.dict_perf = pkl.load(open(fname_perf,"rb"))
        self.dict_u = pkl.load(open(self.fname_u,"rb"))
        self.dict_c = pkl.load(open(self.fname_c,"rb"))
        self.labels_pred_u = pkl.load(open(self.fname_u_pred_labels,"rb"))
        self.labels_pred_c = pkl.load(open(self.fname_c_pred_labels,"rb"))
        #
        # self.U_c1 = self.dict_u["c1"]
        # self.U_c2 = self.dict_u["c2"]
        # self.C_c1 = self.dict_c["c1"]
        # self.C_c2 = self.dict_c["c2"]

    def __print_params(self):
        pp.pprint("is_normalize: ",self.is_normalize)
        pp.pprint("dict_k_nn_c_graph : ",self.dict_k_nn_c_graph)
        #pp.pprint("k_nn_c2_graph : ",self.k_nn_c2_graph)
        pp.pprint("knn_perf_metric : ",self.knn_perf_metric)
        pp.pprint("dict_num_std_c : ",self.dict_num_std_c)
        #pp.pprint("num_std_c2 : ",self.num_std_c2)
        pp.pprint("dict_num_std_c_pairs : ",self.dict_num_std_c_pairs)
        pp.pprint("dict_is_binary_adj_mat_c : ",self.dict_is_binary_adj_mat_c)
        pp.pprint("dict_is_binary_adj_mat_c_pairs : ",self.dict_is_binary_adj_mat_c_pairs)
        #pp.pprint("is_binary_adj_mat_c2 : ",self.is_binary_adj_mat_c2)
        #pp.pprint("is_binary_adj_mat_c12 : ",self.is_binary_adj_mat_c12)
        pp.pprint("dict_resolution_c : ",self.dict_resolution_c)
        pp.pprint("resolution_c_joint : ",self.resolution_c_joint)
        #pp.pprint("resolution_joint_c12 : ",self.resolution_joint_c12)
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
                dict_k_nn_c_graph, knn_perf_metric, \
                dict_num_std_c, dict_num_std_c_pairs, \
                dict_is_binary_adj_mat_c, dict_is_binary_adj_mat_c_pairs, \
                dict_resolution_c, resolution_c_joint, \
                is_use_both_UC, is_use_quant_UC, \
                out_dir_name, data_dir_name, \
                G, X_data_bef_pp, X_meta, y_val_dict,\
                dict_num_clusters, dict_e_size):
        #
        self.is_normalize = is_normalize
        #
        self.dict_k_nn_c_graph = dict_k_nn_c_graph
        self.knn_perf = knn_perf_metric
        #
        self.dict_num_std_c = dict_num_std_c #for thres computation in connecting c1-c1 graph
        self.dict_num_std_c_pairs = dict_num_std_c_pairs #for thres computation in connecting c1-c2 graph
        #
        self.dict_is_binary_adj_mat_c = dict_is_binary_adj_mat_c
        self.dict_is_binary_adj_mat_c_pairs = dict_is_binary_adj_mat_c_pairs
        #
        self.dict_resolution_c = dict_resolution_c
        self.resolution_c_joint = resolution_c_joint
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
            #self.fname_u_pred_labels = self.out_dir_name + "dict_u_clust_labels_quantile_ass2.pkl"
        else:
            self.fname_u = self.out_dir_name + "dict_U.pkl"
            #self.fname_u_pred_labels = self.out_dir_name + "dict_u_clust_labels.pkl"
        #
        if self.is_use_quant_UC:
            self.fname_c = self.out_dir_name + "dict_I_ortho_quantile_ass2.pkl"
            #self.fname_c_pred_labels = self.out_dir_name + "dict_c_clust_labels_quantile_ass2.pkl"
        else:
            self.fname_c = self.out_dir_name + "dict_I_ortho.pkl"
            #self.fname_c_pred_labels = self.out_dir_name + "dict_c_clust_labels.pkl"
        #
        self.fname_c_pred_labels = self.out_dir_name + "dict_c_clust_labels.pkl"
        self.fname_u_pred_labels = self.out_dir_name + "dict_u_clust_labels.pkl"
        #
        
        #
        self.G = G
        self.X_data_bef_pp = X_data_bef_pp
        self.X_meta = X_meta
        self.y_val_dict = y_val_dict
        self.dict_e_size = dict_e_size
        self.dict_num_clusters = dict_num_clusters
        #
        self.dict_c_labels_jc = {}
        self.dict_c_labels_jc_fullg = {}
        self.dict_e_fn_mat = {}
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
        for e_id in self.y_val_dict.keys():
            dict_temp_c[e_id] = self.dict_c[e_id]
        print("dict_temp_c.keys(): ",list(dict_temp_c.keys()))
        #U
        dict_temp_u = {}
        for e_id in self.y_val_dict.keys():
            dict_temp_u[e_id] = self.dict_u[e_id]
        print("dict_temp_u.keys(): ",list(dict_temp_c.keys()))
        print("#")
        #
        #ARI
        self.dict_ari_orig = {}
        for e_id in self.y_val_dict.keys():
            y_pred_c = self.__get_y_pred_kmeans(self.dict_num_clusters,e_id, self.dict_c)
            self.dict_ari_orig[e_id] = adjusted_rand_score(self.y_val_dict[e_id],y_pred_c)
        #
        for e_id in self.y_val_dict.keys():
            print("ARI ",e_id," : ",self.dict_ari_orig[e_id])
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
        # for e_id in self.G.keys():
        #     print("AGS ",e_id," : ",self.dict_agree_score_orig[e_id])
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
        c2_ari = adjusted_rand_score(y_true_c2,y_pred_c2)
        print("c ARI: ",c2_ari)
        return csr_matrix(D_c2_bin_thres), csr_matrix(fn_vec_mat_c2), y_pred_c2, c2_ari
    
    def jointly_cluster(self):
        self.__get_cur_performance()
        #
        if self.is_use_both_UC:
            if self.is_normalize:
                for e_id in self.G.keys():
                    pp_scaler = StandardScaler()
                    self.dict_u[e_id] = pp_scaler.fit_transform(self.dict_u[e_id])
                
        #
        if self.is_normalize:
            for e_id in self.G.keys():
                pp_scaler = StandardScaler()
                self.dict_c[e_id] = pp_scaler.fit_transform(self.dict_c[e_id])
        #
        #get indicator matrices of clustering 1
        print("Obtain clustering 1's indicator matrices")
        self.dict_I_clust1 = {}
        dict_e_idx_start_end = {}
        start_idx = 0
        for e_id in self.y_val_dict.keys():
            #y_pred_c = self.__get_y_pred_kmeans(self.dict_num_clusters, e_id, self.dict_c) #<<<<  change back TBD: revert
            y_pred_c = self.labels_pred_c[e_id] 
            lb_c = preprocessing.LabelBinarizer()
            self.dict_I_clust1[e_id] = lb_c.fit_transform(y_pred_c)
            #
            end_idx = start_idx + self.dict_e_size[e_id]
            dict_e_idx_start_end[e_id] = {"start_idx":start_idx,"end_idx":end_idx}
            print("e_id: ",e_id," dict_I_clust1[e_id].shape: ",self.dict_I_clust1[e_id].shape, ", start_idx: ",start_idx," end_idx: ",end_idx)
            #
            start_idx = end_idx
        #
        print("#")

        #########
        ### Build graphs for each entity
        #########
        dict_e_G = {}
        #dict_e_fn_mat = {}
        dict_e_improved_ari_cc = {}
        for e_id in self.y_val_dict.keys():
            print("#")
            print("Building graph for entity: ",e_id)
            print("---")
            k_nn_c = self.dict_k_nn_c_graph[e_id]
            if self.is_use_both_UC:
                C = self.dict_u[e_id]
                print("Using embd: U_c for e_id: ",e_id)
            else:
                print("Using embd: C_c for e_id: ",e_id)
                C = self.dict_c[e_id]
            #    
            e_size = self.dict_e_size[e_id]
            I = self.dict_I_clust1[e_id]
            resolution = self.dict_resolution_c[e_id]
            y_true_c = self.y_val_dict[e_id]
            num_std_c = self.dict_num_std_c[e_id]
            is_binary_adj_mat = self.dict_is_binary_adj_mat_c[e_id]
            #
            D_c_bin_thres, fn_vec_mat_c, y_pred_c, c_ari = self.do_lovaine_clust(\
                k_nn_c, C, e_size, I, resolution, y_true_c, num_std_c, is_binary_adj_mat)
            dict_e_G[e_id] = D_c_bin_thres
            self.dict_e_fn_mat[e_id] = fn_vec_mat_c
            dict_e_improved_ari_cc[e_id] = c_ari
            self.dict_c_labels_jc[e_id] = y_pred_c
            print("#")

        print("#")
        print("clustering 1's performance: ")
        for e_id in self.y_val_dict.keys():
            print("ARI ",e_id,": ",self.dict_ari_orig[e_id])
        print("#")

        print("#")
        print("clustering 2's performance - only graph entity graph: ")
        for e_id in self.y_val_dict.keys():
            print("ARI ",e_id,": ",dict_e_improved_ari_cc[e_id])
        print("#")        

        #########
        ### Build graphs for each entity pairs
        #########
        dict_epair_G = {}
        for e_id1 in self.y_val_dict.keys():
            for e_id2 in self.y_val_dict.keys():
                if not e_id1 == e_id2:
                    print("#")
                    print("Building graph for entity pairs: ",e_id1," , ",e_id1)
                    print("---")
                    fn_vec_mat_c1 = self.dict_e_fn_mat[e_id1].todense()
                    fn_vec_mat_c2 = self.dict_e_fn_mat[e_id2].todense()
                    D_c1_c2 = metrics.pairwise_distances(fn_vec_mat_c1,fn_vec_mat_c2,metric='manhattan')
                    #
                    print("fn_vec_mat_c1.shape: ",fn_vec_mat_c1.shape)
                    print("fn_vec_mat_c2.shape: ",fn_vec_mat_c2.shape)
                    print("#")
                    print("D_c1_c2.shape: ",D_c1_c2.shape)
                    print("np.sum(D_c1_c2 > 0): ",np.sum(D_c1_c2 > 0))            
                    #
                    thres_c = np.mean(D_c1_c2) + (self.dict_num_std_c_pairs[(e_id1,e_id2)] * np.std(D_c1_c2))
                    D_c1_c2_bin_thres = (D_c1_c2 < thres_c) + 0
                    print("thres_c: ", thres_c)
                    print("np.sum(D_c1_c2_bin_thres > 0): ",np.sum(D_c1_c2_bin_thres > 0))
                    print("#")
                    #
                    if not self.dict_is_binary_adj_mat_c_pairs[(e_id1,e_id2)]:
                        D_c1_c2_bin_thres = D_c1_c2_bin_thres * D_c1_c2    
                    #
                    dict_epair_G[(e_id1,e_id2)] = csr_matrix(D_c1_c2_bin_thres)
                    dict_epair_G[(e_id2,e_id1)] = csr_matrix(D_c1_c2_bin_thres.T)
        print("#")
        print("dict_epair_G.keys: ",list(dict_epair_G.keys()))
        print("#")
        #
        #stitch them to a single big graph of all the distances
        print("Building the master graph: ")
        temp_col_list = []
        for e_id1 in self.G.keys():
            if e_id1.startswith("c"):
                temp_row_list = []
                for e_id2 in self.G.keys():
                    if e_id2.startswith("c"):
                        print("e_id1: ",e_id1)
                        print("e_id2: ",e_id2)
                        print("#")
                        if e_id1 == e_id2:
                            temp_row_list.append(dict_e_G[e_id1])
                        else:
                            temp_row_list.append(dict_epair_G[(e_id1,e_id2)])
                temp_col_list.append(hstack(temp_row_list))
        D_joint = vstack(temp_col_list)

        print("D_joint.shape: ",D_joint.shape)
        print("D_joint.nnz: ")
        print(D_joint.nnz)
        print("#")
        print("Jointly_clustering using lovaine: ")
        #
        print("Building the graph...")
        #G_c = nx.from_numpy_matrix(D_joint)
        G_c = nx.from_scipy_sparse_matrix(D_joint)
        #
        print("Partitioning... ")
        partition_c = community_louvain.best_partition(G_c, resolution=self.resolution_c_joint)
        print("resolution_c_joint: ",self.resolution_c_joint)
        print("c #partition: ",len(np.unique(list(partition_c.values()))))
        #
        y_pred_c = list(partition_c.values())
        print("len(y_pred_c): ",len(y_pred_c))
        #
        print("Splitting the predictions: ")
        dict_y_pred_joint = {}
        for e_id in self.y_val_dict.keys():
            start_idx = dict_e_idx_start_end[e_id]["start_idx"]
            end_idx = dict_e_idx_start_end[e_id]["end_idx"]
            dict_y_pred_joint[e_id] = y_pred_c[start_idx:end_idx]
            self.dict_c_labels_jc_fullg[e_id] = y_pred_c[start_idx:end_idx]
            print("e_id: ",e_id,", start_idx: ",start_idx,", end_idx: ",end_idx,", len(dict_y_pred_joint[e_id]): ",len(dict_y_pred_joint[e_id]))
        #
        print("#")
        print("clustering 1's performance: ")
        for e_id in self.y_val_dict.keys():
            print("ARI ",e_id,": ",self.dict_ari_orig[e_id])
        print("#")
        #
        print("#")
        print("clustering 2's performance - only graph entity graph: ")
        for e_id in self.y_val_dict.keys():
            print("ARI ",e_id,": ",dict_e_improved_ari_cc[e_id])
        print("#")          
        #    
        print("clustering 2's performance - full graph: ")
        for e_id in self.y_val_dict.keys():
            print("ARI ",e_id,": ",adjusted_rand_score(self.y_val_dict[e_id],dict_y_pred_joint[e_id]))