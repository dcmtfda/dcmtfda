import time
import os
import numpy as np
import collections
import sys
import pickle as pkl
#
import torch
from torch import nn

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score
#from sklearn.metrics.cluster import rand_score
from sklearn.metrics import silhouette_score

from scipy.spatial.distance import cdist
from scipy.special import comb

from src.vae import vae
from src.ffn_clust import ffnclust
from src.ffn_ucat import ffnucat
import src.single_cell_metrics as scm
import random

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

class dcmtf(nn.Module): #VAE

    def __input_transformation(self):
        #Construct:
        #dict_C 
        #dict_C = {}
        for e_id in self.G.keys():
            #print("#")
            #print("e_id: ",e_id)
            for x_id in self.G[e_id]:
                #print("x_id: ",x_id)
                cur_id = e_id + "_" + x_id
                #print(">>> ",cur_id)
                #print("#")
                #
                if self.X_meta[x_id][0] == e_id:
                    C = self.X_data[x_id]
                else:
                    C = self.X_data[x_id].transpose(1,0)
                #
                if self.X_dtype[x_id] == "real":
                    is_real = True
                elif self.X_dtype[x_id] == "binary":
                    is_real = False
                else:
                    assert False, "Unknown dtype: "+str(self.X_dtype[x_id])+" for x_id: "+str(x_id)
                #    
                self.dict_C[cur_id] = C

    def __network_construction(self):
        #Construct:
        #dict_vae
        #dict_ffnu_cat
        #dict_ffn_clust
        #
        #Create VAE for each of the possible (entity,X) pairs
        #
        print("is_train: ",self.is_train)
        print("is_load_init: ",self.is_load_init)
        if self.is_train:
            print("#######################")
            print("Creating VAEs: (X-> U)")
            print("#######################")
            #dict_vae = {}
            if self.is_load_init:
                print("is_load_init: ",self.is_load_init)
                print("Loading previous initialized VAEs: ")
                if os.path.exists(self.model_dir+"init"):
                    print("Can't load inits. Dir does not exists: ",self.model_dir+"init")
                else:
                    fname = self.model_dir+"init/dict_vae_init.pkl"
                    print("Loading: ",fname)
                    self.dict_vae = pkl.load(open(fname,"rb"))
                    if self.is_gpu:
                        for temp_key in self.dict_vae.keys():
                            self.dict_vae[temp_key].cuda()
                    print("self.dict_vae: ")
                    print(self.dict_vae)
                    print("#")
            else:
                print("Creating new VAEs: ")
                for e_id in self.G.keys():
                    print("#")
                    print("#")
                    print("#")
                    print("e_id: ",e_id)
                    for x_id in self.G[e_id]:
                        print("x_id: ",x_id)
                        cur_id = e_id + "_" + x_id
                        print(">>> ",cur_id)
                        print("#")
                        if self.X_dtype[x_id] == "real":
                            is_real = True
                        elif self.X_dtype[x_id] == "binary":
                            is_real = False
                        else:
                            assert False, "Unknown dtype: "+str(self.X_dtype[x_id])+" for x_id: "+str(x_id)
                        #    
                        cur_input_dim = self.dict_C[cur_id].shape[1]
                        cur_k_list = self.__get_k_list(cur_input_dim,self.k,self.kf,self.num_layers)
                        cur_actf_list = self.__get_actf_list(cur_k_list,self.e_actf)
                        vae_e = vae(cur_input_dim,cur_k_list,cur_actf_list,is_real)
                        if self.is_gpu:
                            vae_e.cuda()
                        self.dict_vae[cur_id] = vae_e
                #save current inits
                print("#")
                print("Saving the initializations")
                print("#")
                if not os.path.exists(self.model_dir+"init"):
                    os.makedirs(self.model_dir+"init")
                fname = self.model_dir+"init/dict_vae_init.pkl"
                print("Persisting: ",fname)
                pkl.dump(self.dict_vae,open(fname,"wb"))
        else:
            print("#######################")
            print("Loading VAEs: (X-> U)")
            print("#######################")
            if os.path.exists(self.model_dir):
                fname = self.model_dir+"dict_vae.pkl"
                print("Loading: ",fname)
                self.dict_vae = pkl.load(open(fname,"rb"))
                if self.is_gpu:
                    for temp_key in self.dict_vae.keys():
                        self.dict_vae[temp_key].cuda()
                print("self.dict_vae: ")
                print(self.dict_vae)
                print("#")
            else:
                assert False,"model_dir does not exists: "+str(self.model_dir)
        #
        #create FFN for each entity - to combine U_e_x1, U_e_x2,... U_e_xn to U_e
        #where n is the number of matrices associated with entity e  
        #
        if self.is_train:
            print("#########################################")
            print("Creating U concat networks: (Ucat -> U)")
            print("#########################################")
            #
            if self.is_load_init:
                print("is_load_init: ",self.is_load_init)
                print("Loading previous initialized VAEs: ")
                if os.path.exists(self.model_dir+"init"):
                    print("Can't load inits. Dir does not exists: ",self.model_dir+"init")
                else:
                    fname = self.model_dir+"init/dict_ffnu_cat_init.pkl"
                    print("Loading: ",fname)
                    self.dict_ffnu_cat = pkl.load(open(fname,"rb"))
                    if self.is_gpu:
                        for temp_key in self.dict_ffnu_cat.keys():
                            self.dict_ffnu_cat[temp_key].cuda()                    
                    print("self.dict_vae: ")
                    print(self.dict_ffnu_cat)
                    print("#")
            else:
                print("Creating new Ucat networks: ")            
                #dict_ffnu_cat = {}
                for e_id in self.G.keys():
                    print("#")
                    print("e_id:",e_id)
                    print("#")
                    #if len(self.G[e_id]) > 1:
                    if len(self.G[e_id]) > 0:
                        num_clusters = self.dict_num_clusters[e_id]
                        input_dim = int(len(self.G[e_id]) * (self.k/2.0)) #Note: / 2.0 since the mu and sigma layer's output dim are half of k (see: vae)
                        cur_ffn_cat = ffnucat(input_dim,self.k,self.e_actf,num_clusters)
                        if self.is_gpu:
                            cur_ffn_cat.cuda()
                        self.dict_ffnu_cat[e_id] = cur_ffn_cat
                    else:
                        print(">> No fusion needed for e_id: ",e_id, " because len(self.G[e_id]): ",len(self.G[e_id]))
                #save current inits
                print("#")
                print("Saving the initializations")
                print("#")
                if not os.path.exists(self.model_dir+"init"):
                    os.makedirs(self.model_dir+"init")
                fname = self.model_dir+"init/dict_ffnu_cat_init.pkl"
                print("Persisting: ",fname)
                pkl.dump(self.dict_ffnu_cat,open(fname,"wb"))        
        else:
            print("#########################################")
            print("Loading U concat networks: (Ucat -> U)")
            print("#########################################")
            if os.path.exists(self.model_dir):
                fname = self.model_dir+"dict_ffnu_cat.pkl"
                print("Loading: ",fname)
                self.dict_ffnu_cat = pkl.load(open(fname,"rb"))
                if self.is_gpu:
                    for temp_key in self.dict_ffnu_cat.keys():
                        self.dict_ffnu_cat[temp_key].cuda()                 
                print("self.dict_ffnu_cat: ")
                print(self.dict_ffnu_cat)
                print("#")
            else:
                assert False,"model_dir does not exists: "+str(self.model_dir)
        #
        if self.is_train:
            print("##################################################")
            print("Creating Clustering networks: (U-> I, I_ortho)")
            print("##################################################")
            #
            if self.is_load_init:
                print("is_load_init: ",self.is_load_init)
                print("Loading previous initialized VAEs: ")
                if os.path.exists(self.model_dir+"init"):
                    print("Can't load inits. Dir does not exists: ",self.model_dir+"init")
                else:
                    fname = self.model_dir+"init/dict_ffn_clust_init.pkl"
                    print("Loading: ",fname)
                    self.dict_ffn_clust = pkl.load(open(fname,"rb"))
                    if self.is_gpu:
                        for temp_key in self.dict_ffn_clust.keys():
                            self.dict_ffn_clust[temp_key].cuda()                     
                    print("self.dict_ffn_clust: ")
                    print(self.dict_ffn_clust)
                    print("#")
            else:
                print("Creating new clustering networks: ")             
                #dict_ffn_clust = {}
                for e_id in self.G.keys():
                    print("#")
                    print("e_id:",e_id)
                    print("#")
                    input_dim = int(self.k/2.0) #Note: / 2.0 since the mu and sigma layer's output dim are half of k (see: vae)
                    num_clusters = self.dict_num_clusters[e_id]
                    cur_ffn_clust = ffnclust(input_dim,num_clusters,self.k)
                    if self.is_gpu:
                        cur_ffn_clust.cuda()
                    self.dict_ffn_clust[e_id] = cur_ffn_clust
                #save current inits
                print("#")
                print("Saving the initializations")
                print("#")
                if not os.path.exists(self.model_dir+"init"):
                    os.makedirs(self.model_dir+"init")
                fname = self.model_dir+"init/dict_ffn_clust_init.pkl"
                print("Persisting: ",fname)
                pkl.dump(self.dict_ffn_clust,open(fname,"wb"))         
        else:
            print("##################################################")
            print("Loading Clustering networks: (U-> I, I_ortho)")
            print("##################################################")
            if os.path.exists(self.model_dir):
                fname = self.model_dir+"dict_ffn_clust.pkl"
                print("Loading: ",fname)
                self.dict_ffn_clust = pkl.load(open(fname,"rb"))
                if self.is_gpu:
                    for temp_key in self.dict_ffn_clust.keys():
                        self.dict_ffn_clust[temp_key].cuda()                  
                print("self.dict_ffn_clust: ")
                print(self.dict_ffn_clust)
                print("#")
            else:
                assert False,"model_dir does not exists: "+str(self.model_dir)


    def __get_k_list(self, n, k, kf, num_layers):
        k_list = []
        if kf != None:
            while True:
                k1 = int(n * kf)
                if k1 > k:
                    k_list.append(k1)
                    n = k1
                else:
                    k_list.append(k)
                    break
            if num_layers != None:
                print("WARNING: Using kf: "+str(kf)+" to construct the layers. Though num_layers it not None. num_layers: "+str(num_layers))
        elif (num_layers != None) and (num_layers >= 0) and (num_layers < 3):
            if num_layers == 1:
                k_list.append(int(n/2.0))
            elif num_layers == 2:
                k_list.append(int(n/2.0))
                k_list.append(int(n/3.0))
            elif num_layers == 0:
                pass
            else:
                assert False,"Unexpected values for num_layers: "+str(num_layers)+". Expected; 1 or 2"
            k_list.append(k)
        else:
            assert False,"Unexpected values for kf: "+str(kf)+", num_layers: "+str(num_layers)
        return k_list

    def __get_actf_list(self, k_list, e_actf):
        actf_list_e = []
        for k in k_list:
            actf_list_e.append(e_actf) 
        actf_list_e.append(e_actf)  #for the extra decoding layer
        return actf_list_e

    def __get_y_pred_kmeans(self,X,num_clusters):
        kmeans_e = KMeans(n_clusters=num_clusters, random_state=0)
        kmeans_e.fit(X.data.cpu().numpy())
        labels_pred = kmeans_e.labels_
        return labels_pred

    def __get_rigorous_C(self, I_o, num_clusters):
        assert I_o.shape[1] == num_clusters
        #
        labels_pred = self.__get_y_pred_kmeans(I_o,num_clusters)
        I = np.zeros((len(labels_pred),num_clusters))
        for i in np.arange(I.shape[0]):
            j = labels_pred[i]
            I[i,j] = 1
        C = I / np.sqrt(I.sum(axis=0))
        return C        



    # No tensor round to x decimal places available readily apparently
    def __custom_round(self, X, n_digits):
            #from https://discuss.pytorch.org/t/round-tensor-to-x-decimal-places/25832/4
            return (X * 10**n_digits).round() / (10**n_digits)
            #return torch.round(X)

    def __get_vae_loss(self, recon_x, x, mu, logvar, dim, is_real):
        beta = 1
        #print("recon_x.shape: ",recon_x.shape)
        #print("x.shape: ",x.shape)
        mse_criterion = torch.nn.MSELoss(reduction="none")
        #mse_criterion = torch.nn.MSELoss()
        if is_real:
            recons_loss = torch.sum(mse_criterion(recon_x,x),dim=dim)
            #recons_loss = 1.0 * mse_criterion(recon_x,x)
        else:
            recons_loss = torch.sum(torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='none'),dim=dim)
            #recons_loss = 1.0 * torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='mean')
        # from https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=dim)
        #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #
        #print("recons_loss: ",recons_loss.shape," KLD.shape: ",KLD.shape)
        #return torch.sum(recons_loss + KLD), recons_loss, KLD
        return torch.mean(recons_loss + (beta*KLD))
        #return torch.sum(recons_loss + (beta*KLD))
        #return torch.sum(recons_loss + KLD)
        #return recons_loss + KLD
        #return torch.sum(torch.sum(recons_loss) + torch.sum(KLD))


    # Converts dot product i.e distance matrix M_(nxn) = f(X,C) = (X^T.C).(X^T.C)^T = P.P^T to
    # similarity matrix S = [diag(P.P^T)_(nx1)]^repeat_nrows + [diag(P.P^T)_(1xn)]^repeat_ncols - 2M
    # Ref regd conversion: 
    # https://en.wikipedia.org/wiki/Matrix_norm#Frobenius_norm
    # https://en.wikipedia.org/wiki/Frobenius_inner_product
    def __get_fro_from_dot(self,P,M_dot):
        #print("M_dot[M_dot < 0]: ")
        #print(M_dot[M_dot < 0])
        #assert torch.sum(M_dot < 0) == 0
        # print("torch.mm(P,P.transpose(1,0)): ")
        # print(torch.mm(P,P.transpose(1,0)))
        # print("#")
        # print("M_dot: ")
        # print(M_dot)
        # print("#")
        # print("P: ")
        # print(P)
        #assert torch.sum(torch.mm(P,P.transpose(1,0)) == M_dot) == M_dot.shape[0]*M_dot.shape[1]
        P_row_norm_sq = torch.diag(torch.mm(P,P.transpose(1,0)))
        assert P_row_norm_sq.shape[0] == P.shape[0]
        M_fro_sq = P_row_norm_sq.view(-1, 1) + P_row_norm_sq.view(1, -1) - (2.0 * M_dot)
        #print("torch.diag(M_fro_sq): ",torch.diag(M_fro_sq))
        ###assert torch.sum(torch.diag(self.__custom_round(M_fro_sq,1))) == 0," torch.sum(torch.diag(M_fro_sq)): "+str(torch.sum(torch.diag(M_fro_sq)))+" M_fro_sq: "+str(M_fro_sq)
        #M_fro_sq = M_fro_sq - torch.diag(torch.diag(M_fro_sq))
        #M_fro_sq[M_fro_sq < 0] = 0
        #print("M_fro_sq[M_fro_sq < 0]: ")
        #print(M_fro_sq[M_fro_sq < 0])
        #assert torch.sum(M_fro_sq < 0) == 0
        return M_fro_sq

    #Calculating M as in Eqn (8) and (9) of CFRM 
    #http://www.cs.binghamton.edu/~blong/publications/icml06.pdf
    def __get_M_fro(self, e_id, dict_cur_batch_idx_list):
        # print("#")
        # print("Calculating M...")
        # print("e_id: ",e_id, ", size: ",self.dict_e_size[e_id]," cur_batch_size: ",self.dict_e_size_cur_batch[e_id])
        # print("#")
        #e_size = self.dict_mini_batch_size[e_id]
        e_size = self.dict_e_size_cur_batch[e_id]
        M_fro = torch.zeros(e_size,e_size)#.cuda()
        if self.is_gpu:
            M_fro = M_fro.cuda()
        #
        for x_id in self.G[e_id]:
            #print("processing: ",x_id)
            row_e_id = self.X_meta[x_id][0]
            col_e_id = self.X_meta[x_id][1]
            # print("#")
            # print("self.dict_U[row_e_id].shape: ",self.dict_U[row_e_id].shape)
            # print("self.dict_U[row_e_id]: ",self.dict_U[row_e_id])
            # print("#")
            # print("self.dict_U[col_e_id].shape: ",self.dict_U[col_e_id].shape)
            # print("self.dict_U[col_e_id]: ",self.dict_U[col_e_id])
            # print("#")
            #R = X_data[x_id][dict_cur_batch_idx_list[row_e_id],:][:,dict_cur_batch_idx_list[col_e_id]]
            R = torch.mm(self.dict_U[row_e_id],self.dict_U[col_e_id].transpose(1,0)) ##[dict_cur_batch_idx_list[row_e_id],:][:,dict_cur_batch_idx_list[col_e_id]]
            if e_id == self.X_meta[x_id][0]:
                # print("R.shape: ",R.shape)
                # print("dict_I_ortho[X_meta[x_id][1]].shape: ",self.dict_I_ortho[self.X_meta[x_id][1]].shape)
                # print("#")
                # print("R: ",R)
                # print("dict_I_ortho[X_meta[x_id][1]]: ",self.dict_I_ortho[self.X_meta[x_id][1]])
                # print("#")
                M_dot = self.__custom_round(                   
                                    torch.mm(                            
                                        torch.mm(                                     
                                            torch.mm(                                              
                                                R,self.dict_I_ortho[self.X_meta[x_id][1]]

                                            ),\
                                            self.dict_I_ortho[self.X_meta[x_id][1]].transpose(1,0)
                                        ),\
                                        R.transpose(1,0)\
                                    )\
                  ,4)
                P = torch.mm(R,self.dict_I_ortho[self.X_meta[x_id][1]])
            elif e_id == self.X_meta[x_id][1]:
                # print("R.shape: ",R.shape)
                # print("dict_I_ortho[X_meta[x_id][0]].shape: ",self.dict_I_ortho[self.X_meta[x_id][0]].shape)
                # print("#")
                # print("R: ",R)
                # print("dict_I_ortho[X_meta[x_id][0]]: ",self.dict_I_ortho[self.X_meta[x_id][0]])
                # print("#")
                R = R.transpose(1,0)
                M_dot = self.__custom_round(                   
                                    torch.mm(                            
                                        torch.mm(                                     
                                            torch.mm(                                              
                                                R,self.dict_I_ortho[self.X_meta[x_id][0]]
                                            ),\
                                            self.dict_I_ortho[self.X_meta[x_id][0]].transpose(1,0)
                                        ),\
                                        R.transpose(1,0)\
                                    )\
                  ,4)
                P = torch.mm(R,self.dict_I_ortho[self.X_meta[x_id][0]])
            else:
                assert False
            assert M_dot.shape[0] <= self.dict_mini_batch_size[e_id]
            assert M_dot.shape[1] <= self.dict_mini_batch_size[e_id]
            M_fro_cur_x = self.__get_fro_from_dot(P,M_dot)
            if self.is_gpu:
                M_fro_cur_x = M_fro_cur_x.cuda()
            
            #print("M_fro_cur_x.shape: ",M_fro_cur_x.shape)
            #print("M_fro_cur_x: ",M_fro_cur_x)
            #print("#")
            #print("M_fro.shape: ",M_fro.shape)
            #print("M_fro: ",M_fro)
            #print("#")
            M_fro = M_fro + M_fro_cur_x #TODO: Check if this is ok?
        return M_fro

    def __is_converged(self, prev_cost,cost,convg_thres,epoch):
        is_converged = False
        #diff = (prev_cost - cost)
        diff = (abs(prev_cost) - abs(cost))
        if abs(diff) < convg_thres:
            print("epoch: ",epoch,", prev_cost: ",prev_cost,", cost: ",cost,", convg_thres: ",convg_thres,", abs(diff): ",abs(diff))
            is_converged = True
        # if np.isnan(cost.cpu().numpy()):
        #     return True
        return is_converged

    def __pretrain(self, e_id, x_id):
        #
        cur_id = e_id + "_" + x_id
        #
        if self.X_dtype[x_id] == "real":
            is_real = True
        elif self.X_dtype[x_id] == "binary":
            is_real = False
        else:
            assert False, "Unknown dtype: "+str(self.X_dtype[x_id])+" for x_id: "+str(x_id)
        #
        optimizer = torch.optim.Adam(self.dict_vae[cur_id].parameters(),\
                                     lr = self.learning_rate_pretrain,\
                                     weight_decay = self.weight_decay_pretrain)
        #    
        print("###")
        print("cur_id: ",cur_id)
        print("#")
        print("Pre-train started.")
        epoch = 1
        dict_loss_metadata = {"start":0,"end":0}
        prev_loss_epoch = 0
        dict_loss = {}
        while True:
            batch_size = self.dict_batch_size[e_id]
            mini_batch_size = self.dict_mini_batch_size[e_id]
            if self.mini_batch_size_frac < 1.0:
                #with batching
                cur_batch_idx_list = torch.randperm(batch_size)[:mini_batch_size]
            elif self.mini_batch_size_frac == 1.0:
                #without batching
                cur_batch_idx_list = np.arange(mini_batch_size)
            else:
                assert False," Unexpected mini_batch_size_frac: "+str(self.mini_batch_size_frac)
            cur_batch_idx_list = torch.randperm(batch_size)[:mini_batch_size]
            cur_C = self.dict_C[cur_id][cur_batch_idx_list,:]
            if epoch > self.max_epochs_pretrain:
                break
            mse_criterion = torch.nn.MSELoss()
            #returns x_enc, mu, logvar, x_dec
            x_enc, mu, logvar, pred_C = self.dict_vae[cur_id](cur_C)  
            #
            #loss, recons_loss, kld_loss = self.__get_vae_loss(pred_C, cur_C, mu, logvar, dim=self.dim, is_real=self.is_real)
            loss_epoch = 1.0 * self.__get_vae_loss(pred_C, cur_C, mu, logvar, dim=1, is_real=is_real)
            dict_loss[epoch] = loss_epoch
            dict_loss[epoch].backward()
            #
            optimizer.step()
            optimizer.zero_grad()
            #
            print("pre-train epoch: ",epoch,", loss: ",round(dict_loss[epoch].item(),4))
            #
            if epoch == 1:
                dict_loss_metadata["start"] = dict_loss[epoch].item()
            epoch+=1
            #
            if self.convg_thres_pretrain != None:
                if self.__is_converged(prev_loss_epoch,loss_epoch,self.convg_thres_pretrain,epoch):
                    print("**Pre-train converged**")
                    break
                prev_loss_epoch = loss_epoch
        print("pre-train ended.")
        print("#")
        dict_loss_metadata["end"] = dict_loss[epoch-1].item()
        print("pre-train loss summary: ")
        print(dict_loss_metadata)
        print("###")

    def __print_input(self):
        print("k: ",self.k)
        print("kf: ",self.kf)
        print("num_layers: ",self.num_layers)
        print("e_actf: ",self.e_actf)
        print("dict_num_clusters: ",self.dict_num_clusters)
        print("learning_rate: ",self.learning_rate)
        print("weight_decay: ",self.weight_decay)
        print("convg_thres: ",self.convg_thres)
        print("max_epochs: ",self.max_epochs)
        print("is_pretrain: ",self.is_pretrain)
        print("learning_rate_pretrain: ",self.learning_rate_pretrain)
        print("weight_decay_pretrain: ",self.weight_decay_pretrain)
        print("convg_thres_pretrain: ",self.convg_thres_pretrain)
        print("max_epochs_pretrain: ",self.max_epochs_pretrain)
        print("mini_batch_size_frac: ",self.mini_batch_size_frac)
        print("dict_e_loss_weight: ",self.dict_e_loss_weight)
        print("dict_loss_weight",self.dict_loss_weight)
        print("dict_e_size: ",self.dict_e_size)
        print("dict_batch_size: ",self.dict_batch_size)
        print("dict_mini_batch_size: ",self.dict_mini_batch_size)
        print("#")
        print("G: ")
        print(self.G)
        print("#")
        print("X_data: ")
        print(self.X_data)
        print("#")
        print("X_meta: ")
        print(self.X_meta)
        print("#")
        print("X_dtype: ")
        print(self.X_dtype)
        print("#")
        print("y_val_dict",)
        print(self.y_val_dict)
        print("",)

    def __copy_params(self):
        self.out_params["k"] = self.k
        self.out_params["kf"] =self.kf
        self.out_params["num_layers"] = self.num_layers
        self.out_params["e_actf"] = self.e_actf
        self.out_params["dict_num_clusters"] = self.dict_num_clusters
        self.out_params["learning_rate"] = self.learning_rate
        self.out_params["weight_decay"] = self.weight_decay
        self.out_params["convg_thres"] = self.convg_thres
        self.out_params["max_epochs"] = self.max_epochs
        self.out_params["is_pretrain"] = self.is_pretrain
        self.out_params["learning_rate_pretrain"] = self.learning_rate_pretrain
        self.out_params["weight_decay_pretrain"] = self.weight_decay_pretrain
        self.out_params["convg_thres_pretrain"] = self.convg_thres_pretrain
        self.out_params["max_epochs_pretrain"] = self.max_epochs_pretrain
        self.out_params["mini_batch_size_frac"] = self.mini_batch_size_frac
        self.out_params["dict_e_loss_weight"] = self.dict_e_loss_weight
        self.out_params["dict_loss_weight"] =self.dict_loss_weight
        self.out_params["dict_e_size"] = self.dict_e_size
        self.out_params["dict_batch_size"] = self.dict_batch_size
        self.out_params["dict_mini_batch_size"] = self.dict_mini_batch_size
        self.out_params["G"] = self.G
        self.out_params["X_meta"] = self.X_meta
        self.out_params["X_dtype"] = self.X_dtype
        self.out_params["loss"] = self.loss

    def __init__(self,\
        G, X_data, X_data_bef_pp, X_data_size_fac, X_meta, X_dtype,\
        k, kf, num_layers, e_actf, dict_num_clusters,\
        learning_rate, weight_decay, convg_thres, max_epochs,\
        is_pretrain, learning_rate_pretrain, weight_decay_pretrain, convg_thres_pretrain, max_epochs_pretrain,\
        mini_batch_size_frac, num_batches, dict_e_loss_weight, dict_loss_weight,\
        dict_e_size, y_val_dict,\
        is_gpu, is_train, is_load_init, is_rand_batch_train, \
        model_dir):
        super(dcmtf, self).__init__()
        self.G = G
        self.X_data = X_data
        self.X_meta  = X_meta
        self.X_dtype = X_dtype
        self.X_data_bef_pp = X_data_bef_pp
        self.X_data_size_fac = X_data_size_fac
        self.k = k
        self.kf = kf
        self.num_layers = num_layers
        self.e_actf = e_actf
        self.dict_num_clusters = dict_num_clusters
        self.learning_rate = learning_rate
        self.weight_decay =  weight_decay
        self.convg_thres =  convg_thres
        self.max_epochs = max_epochs
        self.is_pretrain = is_pretrain
        self.learning_rate_pretrain = learning_rate_pretrain
        self.weight_decay_pretrain = weight_decay_pretrain
        self.convg_thres_pretrain =  convg_thres_pretrain
        self.max_epochs_pretrain = max_epochs_pretrain
        self.mini_batch_size_frac = mini_batch_size_frac
        self.dict_e_loss_weight = dict_e_loss_weight
        self.dict_loss_weight = dict_loss_weight
        self.dict_e_size = dict_e_size
        self.y_val_dict = y_val_dict
        self.is_gpu = is_gpu
        self.is_rand_batch_train = is_rand_batch_train
        #
        self.is_train = is_train
        self.is_load_init = is_load_init
        self.model_dir = model_dir
        #
        self.dict_batch_size = {}
        #self.dict_mini_batch_size = {}
        for e_id in self.G.keys():
            self.dict_batch_size[e_id] = self.dict_e_size[e_id]
            #self.dict_mini_batch_size[e_id] = int(self.dict_batch_size[e_id] * self.mini_batch_size_frac)
        #
        self.loss = -1
        self.dict_vae = {}
        self.dict_ffnu_cat = {}
        self.dict_ffn_clust = {}
        self.dict_C = {}
        self.dict_U = {}
        self.dict_mu = {}
        self.dict_I_ortho = {}
        self.dict_recons_X = {}
        self.dict_A = {}
        self.out_params = {}
        self.dict_C_dec = {}
        #
        self.num_batches = num_batches
        self.dict_mini_batch_size = {}
        for e_id in self.G.keys():
            self.dict_mini_batch_size[e_id] = int(np.ceil(self.dict_batch_size[e_id] / self.num_batches))
        self.dict_e_idx = {}
        for e_id in self.G.keys():
            #self.dict_e_idx[e_id] = np.random.permutation(self.dict_batch_size[e_id])
            self.dict_e_idx[e_id] = torch.randperm(self.dict_batch_size[e_id])
        self.dict_e_size_cur_batch = {}
        #
        self.__print_input()
        #
        self.__input_transformation()
        self.__network_construction()

    def __get_y_pred_kmeans_c(self, e_id):
        kmeans_e = KMeans(n_clusters=self.dict_num_clusters[e_id], random_state=0)
        kmeans_e.fit(self.dict_I_ortho[e_id].data.cpu().numpy())
        labels_pred = kmeans_e.labels_
        return labels_pred

    def calc_kmeans_sil_score(self):
        print("#")
        print("calc_kmeans_sil_score: ")
        print("#")
        dict_sil = {}
        for e_id in self.G.keys():
            if e_id == "g":
                y_pred_e = self.__get_y_pred_kmeans_c(e_id)
                y_pred_e_sel = y_pred_e[np.array(self.y_val_dict[e_id])>= 0]
                #
                I_ortho_temp = self.dict_I_ortho[e_id].data.cpu().numpy()
                I_ortho_temp_sel = I_ortho_temp[np.array(self.y_val_dict[e_id]) >= 0]
                #
                #cur_sil = silhouette_score(I_ortho_temp_sel, y_pred_e_sel, metric='euclidean') 
                U_dist = cdist(I_ortho_temp_sel, I_ortho_temp_sel, metric = "euclidean")
                cur_sil = silhouette_score(U_dist, y_pred_e_sel, metric="precomputed") 
                
                dict_sil[e_id] = round(cur_sil,4)
            else:
                y_pred_e = self.__get_y_pred_kmeans_c(e_id)
                I_ortho_temp = self.dict_I_ortho[e_id].data.cpu().numpy()
                U_dist = cdist(I_ortho_temp, I_ortho_temp, metric = "euclidean")
                cur_sil = silhouette_score(U_dist, y_pred_e, metric='precomputed') 
                dict_sil[e_id] = round(cur_sil,4)
        return dict_sil

    #https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation
    def __rand_index_score(self, clusters, classes):
        tp_plus_fp = comb(np.bincount(clusters), 2).sum()
        tp_plus_fn = comb(np.bincount(classes), 2).sum()
        A = np.c_[(clusters, classes)]
        tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
                 for i in set(clusters))
        fp = tp_plus_fp - tp
        fn = tp_plus_fn - tp
        tn = comb(len(A), 2) - tp - fp - fn
        return (tp + tn) / (tp + fp + fn + tn)

    def calc_kmeans_ari_c(self):
        print("#")
        print("calc_kmeans_ari_c: ")
        print("#")
        dict_ari = {}
        for e_id in self.y_val_dict.keys():
            print("e_id: ",e_id)
            if len(self.y_val_dict[e_id]) > 0:
                # if e_id == "g":
                #     y_e_filtered = np.array(self.y_val_dict[e_id])[np.array(self.y_val_dict[e_id]) >= 0]
                #     y_pred_e = self.__get_y_pred_kmeans_c(e_id)
                #     y_pred_e_sel = y_pred_e[np.array(self.y_val_dict[e_id])>= 0]
                #     cur_ari = adjusted_rand_score(y_e_filtered,y_pred_e_sel)
                #     dict_ari[e_id] = round(cur_ari,4)
                # else:
                if np.sum(self.y_val_dict[e_id] < 0) > 0: #labels contain -ve values => ignore them from performance computation
                    y_true_e_all = self.y_val_dict[e_id]
                    y_pred_e_all = self.__get_y_pred_kmeans_c(e_id)
                    #filter ytrue -> only non-negative values
                    y_true_e = np.array(y_true_e_all)[np.array(y_true_e_all) >= 0]
                    y_pred_e = y_pred_e_all[np.array(y_true_e_all)>= 0]
                    assert y_true_e.shape[0] == y_pred_e.shape[0]
                    print("#")
                    print("Ground truth labels contain -ve values. Considering only +ve labels for performance computation.")
                    print("Reduced len(y_true_e_all): ",y_true_e_all.shape[0]," to len(y_true_e): ",y_true_e.shape[0])
                    print("Reduced len(y_pred_e_all): ",y_pred_e_all.shape[0]," to len(y_pred_e): ",y_pred_e.shape[0])
                    print("#")
                else:
                    y_true_e = self.y_val_dict[e_id]
                    y_pred_e = self.__get_y_pred_kmeans_c(e_id)
                    #
                    y_true_e_all = self.y_val_dict[e_id]
                    y_pred_e_all = self.__get_y_pred_kmeans_c(e_id)
                #
                cur_ari = adjusted_rand_score(y_true_e, y_pred_e)
                cur_ami = adjusted_mutual_info_score(y_pred_e, y_true_e)
                cur_nmi = normalized_mutual_info_score(y_pred_e, y_true_e)
                cur_ri = self.__rand_index_score(y_true_e, y_pred_e) 
                # print("y_pred_e: ")
                # print(list(y_pred_e))
                # print("y_true_e: ")
                # print(y_true_e)
                dict_ari[e_id] = round(cur_ari,4)
                #print("dict_ari[e_id]: ",dict_ari[e_id])
                print("###")
                print("Clustering Results with C: ")
                print("###")
                print("ARI: ",np.round(cur_ari,6))
                print("AMI: ",np.round(cur_ami,6))
                print("NMI: ",np.round(cur_nmi,6))
                print("RI: ",np.round(cur_ri,6))
                print("#")
                # print("Cluster Size: ")
                # print("y_true_e: ")
                # print(collections.Counter(y_true_e_all))
                # print(collections.Counter(y_true_e))
                # print("y_pred_e: ")
                # print(collections.Counter(y_pred_e_all))
                # print(collections.Counter(y_pred_e))
                # print("#")
            else:
                dict_ari[e_id] = 0.0
        return dict_ari

    def __get_y_pred_kmeans_u(self, e_id):
        kmeans_e = KMeans(n_clusters=self.dict_num_clusters[e_id], random_state=0)
        kmeans_e.fit(self.dict_U[e_id].data.cpu().numpy())
        labels_pred = kmeans_e.labels_
        return labels_pred

    def calc_kmeans_ari_u(self):
        print("#")
        print("calc_kmeans_ari_u: ")
        print("#")
        dict_ari = {}
        for e_id in self.y_val_dict.keys():
            print("e_id: ",e_id)
            if len(self.y_val_dict[e_id]) > 0:
                # if e_id == "g":
                #     y_e_filtered = np.array(self.y_val_dict[e_id])[np.array(self.y_val_dict[e_id]) >= 0]
                #     y_pred_e = self.__get_y_pred_kmeans_u(e_id)
                #     y_pred_e_sel = y_pred_e[np.array(self.y_val_dict[e_id]) >= 0]
                #     # print("self.y_val_dict[e_id]: ",self.y_val_dict[e_id])
                #     # print("y_e_filtered: ",y_e_filtered)
                #     # print("y_pred_e_sel: ",y_pred_e_sel)
                #     cur_ari = adjusted_rand_score(y_e_filtered,y_pred_e_sel)
                #     dict_ari[e_id] = round(cur_ari,4)
                # else:
                if np.sum(self.y_val_dict[e_id] < 0) > 0: #labels contain -ve values => ignore them from performance computation
                    y_true_e_all = self.y_val_dict[e_id]
                    y_pred_e_all = self.__get_y_pred_kmeans_u(e_id)
                    #filter ytrue -> only non-negative values
                    y_true_e = np.array(y_true_e_all)[np.array(y_true_e_all) >= 0]
                    y_pred_e = y_pred_e_all[np.array(y_true_e_all)>= 0]
                    assert y_true_e.shape[0] == y_pred_e.shape[0]
                    print("#")
                    print("Ground truth labels contain -ve values. Considering only +ve labels for performance computation.")
                    print("Reduced len(y_true_e_all): ",y_true_e_all.shape[0]," to len(y_true_e): ",y_true_e.shape[0])
                    print("Reduced len(y_pred_e_all): ",y_pred_e_all.shape[0]," to len(y_pred_e): ",y_pred_e.shape[0])
                    print("#")
                else:
                    y_true_e = self.y_val_dict[e_id]
                    y_pred_e = self.__get_y_pred_kmeans_u(e_id)
                    #
                    y_true_e_all = self.y_val_dict[e_id]
                    y_pred_e_all = self.__get_y_pred_kmeans_c(e_id)

                #y_pred_e = self.__get_y_pred_kmeans_u(e_id)                                        
                cur_ari = adjusted_rand_score(y_true_e, y_pred_e)
                cur_ami = adjusted_mutual_info_score(y_pred_e, y_true_e)
                cur_nmi = normalized_mutual_info_score(y_pred_e, y_true_e)
                cur_ri = self.__rand_index_score(y_true_e, y_pred_e) 
                # print("y_pred_e: ")
                # print(y_pred_e)
                # print("y_true_e: ")
                # print(y_true_e)
                dict_ari[e_id] = round(cur_ari,4)
                #print("dict_ari[e_id]: ",dict_ari[e_id])
                print("###")
                print("Clustering Results with U: ")
                print("###")
                print("ARI: ",np.round(cur_ari,6))
                print("AMI: ",np.round(cur_ami,6))
                print("NMI: ",np.round(cur_nmi,6))
                print("RI: ",np.round(cur_ri,6))
                print("#")
                # print("Cluster Size: ")
                # print("y_true_e: ")
                # print(collections.Counter(y_true_e_all))
                # print(collections.Counter(y_true_e))
                # print("y_pred_e: ")
                # print(collections.Counter(y_pred_e_all))
                # print(collections.Counter(y_pred_e))
                # print("#")
            else:
                dict_ari[e_id] = 0.0                
        return dict_ari

    def get_u_clust_labels(self):
        print("#")
        print("get_u_clust_labels: ")
        print("#")
        dict_labels_u = {}
        for e_id in self.y_val_dict.keys():
            y_pred_e = self.__get_y_pred_kmeans_u(e_id)
            dict_labels_u[e_id] = y_pred_e
        return dict_labels_u

    def get_c_clust_labels(self):
        print("#")
        print("get_c_clust_labels: ")
        print("#")
        dict_labels_c = {}
        for e_id in self.y_val_dict.keys():
            y_pred_e = self.__get_y_pred_kmeans_c(e_id)
            dict_labels_c[e_id] = y_pred_e
        return dict_labels_c

    def calc_batchmix_entropy(self,list_eid,k):
        print("#")
        print("calc_batchmix_entropy: ")
        print("#")
        dict_U_temp = {} 
        for e_id in list_eid:
            dict_U_temp[e_id] = self.dict_U[e_id].cpu().data.numpy()
        bme = scm.get_mean_entropy(dict_U_temp, k)
        return bme
    
    def calc_batchmix_entropy_c(self,list_eid,k):
        print("#")
        print("calc_batchmix_entropy C: ")
        print("#")
        dict_U_temp = {} 
        for e_id in list_eid:
            dict_U_temp[e_id] = self.dict_I_ortho[e_id].cpu().data.numpy()
        bme = scm.get_mean_entropy(dict_U_temp, k)
        return bme

    def calc_align_score(self,list_eid,k):
        print("#")
        print("calc_align_score: ")
        print("#")
        dict_U_temp = {} 
        for e_id in list_eid:
            dict_U_temp[e_id] = self.dict_U[e_id].cpu().data.numpy()
        align_score = scm.get_align_score(dict_U_temp, k)
        return align_score

    def calc_align_score_c(self,list_eid,k): 
        print("#")
        print("calc_align_score C: ")
        print("#")
        dict_U_temp = {} 
        for e_id in list_eid:
            dict_U_temp[e_id] = self.dict_I_ortho[e_id].cpu().data.numpy()
        align_score = scm.get_align_score(dict_U_temp, k)
        return align_score

    def calc_agree_score_u(self,list_eid,knn_k):
        print("#")
        print("calc_agree_score: ")
        print("#")
        dict_U_temp = {} 
        for e_id in list_eid:
            dict_U_temp[e_id] = self.dict_U[e_id].cpu().data.numpy()
        dict_agree_score, dict_agree_std = scm.get_eid_agree_score(dict_U_temp, knn_k, self.G, self.X_data_bef_pp, self.X_meta, nmf_k=self.k)
        return dict_agree_score

    def calc_agree_score_c(self,list_eid,knn_k):
        print("#")
        print("calc_agree_score: ")
        print("#")
        dict_U_temp = {} 
        for e_id in list_eid:
            dict_U_temp[e_id] = self.dict_I_ortho[e_id].cpu().data.numpy()
        dict_agree_score, dict_agree_std = scm.get_eid_agree_score(dict_U_temp, knn_k, self.G, self.X_data_bef_pp, self.X_meta, nmf_k=self.k)
        return dict_agree_score

    def persist_out(self,out_dir):
        print("#")
        print("Persisting model and outputs: ")
        print("#")
        fname = out_dir+"dict_vae.pkl"
        print("Persisting: ",fname)
        pkl.dump(self.dict_vae,open(fname,"wb"))

        fname = out_dir+"dict_ffnu_cat.pkl"
        print("Persisting: ",fname)
        pkl.dump(self.dict_ffnu_cat,open(fname,"wb"))

        fname = out_dir+"dict_ffn_clust.pkl"
        print("Persisting: ",fname)
        pkl.dump(self.dict_ffn_clust,open(fname,"wb"))

        fname = out_dir+"dict_A.pkl"
        print("Persisting: ",fname)
        dict_temp = self.dict_A
        dict_temp_np = {}
        for temp_key in dict_temp.keys():
            dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
        pkl.dump(dict_temp_np,open(fname,"wb"))

        fname = out_dir+"dict_u_clust_labels.pkl"
        print("Persisting: ",fname)
        dict_u_clust_labels = self.get_u_clust_labels()
        pkl.dump(dict_u_clust_labels,open(fname,"wb"))

        fname = out_dir+"dict_c_clust_labels.pkl"
        print("Persisting: ",fname)
        dict_c_clust_labels = self.get_c_clust_labels()
        pkl.dump(dict_c_clust_labels,open(fname,"wb"))

        fname = out_dir+"dict_recons_X.pkl"
        print("Persisting: ",fname)
        dict_temp = self.dict_recons_X
        dict_temp_np = {}
        for temp_key in dict_temp.keys():
            dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
        pkl.dump(dict_temp_np,open(fname,"wb"))

        fname = out_dir+"dict_recons_Y.pkl"
        print("Persisting: ",fname)
        dict_temp = self.dict_C_dec
        dict_temp_np = {}
        for temp_key in dict_temp.keys():
            dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
        pkl.dump(dict_temp_np,open(fname,"wb"))

        fname = out_dir+"dict_I_ortho.pkl"
        print("Persisting: ",fname)
        dict_temp = self.dict_I_ortho
        dict_temp_np = {}
        for temp_key in dict_temp.keys():
            dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
        pkl.dump(dict_temp_np,open(fname,"wb"))

        fname = out_dir+"dict_U.pkl"
        print("Persisting: ",fname)
        dict_temp = self.dict_U
        dict_temp_np = {}
        for temp_key in dict_temp.keys():
            dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
        pkl.dump(dict_temp_np,open(fname,"wb"))

        fname = out_dir+"dict_mu.pkl"
        print("Persisting: ",fname)
        dict_temp = self.dict_mu
        dict_temp_np = {}
        for temp_key in dict_temp.keys():
            dict_temp_np[temp_key] = dict_temp[temp_key].data.cpu().numpy()
        pkl.dump(dict_temp_np,open(fname,"wb"))

        fname = out_dir+"dict_out_params.pkl"
        print("Persisting: ",fname)
        pkl.dump(self.out_params,open(fname,"wb"))
        print("#")


    def fit(self):
        #
        if self.is_pretrain:
            for e_id in self.G.keys():
                for x_id in self.G[e_id]:
                    self.__pretrain(e_id, x_id)
        #
        # Train prep
        params_list_data = []
        params_list_clust = []
        #
        #data params
        #VAE: X -> [U,U...]
        for e_id in self.G.keys():
            for x_id in self.G[e_id]:
                cur_id = e_id + "_" + x_id
                params_list_data+=list(self.dict_vae[cur_id].parameters())
        #ffn-cat: [U,U...] -> U
        for e_id in self.G.keys():
            #if len(self.G[e_id]) > 1:
            if len(self.G[e_id]) > 0:
                params_list_data+=list(self.dict_ffnu_cat[e_id].parameters())
        #
        #clust params: all of data params
        params_list_clust+=params_list_data
        #plus ffn-clust: U -> I_ortho 
        for e_id in self.G.keys():
            params_list_clust+=list(self.dict_ffn_clust[e_id].parameters())
        #
        optimizer_data = torch.optim.Adam(params_list_data, lr=self.learning_rate, weight_decay=self.weight_decay)
        optimizer_clust = torch.optim.Adam(params_list_clust, lr=self.learning_rate, weight_decay=self.weight_decay)
        #
        #debug - plot
        dict_loss_recons_debug = {}
        dict_loss_aec_pass1_debug = {}
        dict_loss_aec_pass2_debug = {}
        dict_loss_regl_debug = {}
        dict_loss_epoch_pass1_debug = {}
        dict_loss_epoch_pass2_debug = {}
        #
        # debug metadata - start and end losses
        dict_loss_aec_metadata_pass1 = {}
        dict_loss_aec_metadata_pass2 = {}
        dict_loss_recons_metadata = {}
        dict_loss_regl_metadata = {}
        #
        for x_id in self.X_meta.keys():
            dict_loss_recons_metadata[x_id] = {"start":0, "end":0}
        #
        for e_id in self.G.keys():
            dict_loss_regl_metadata[e_id] = {"start":0, "end":0}
        #
        for e_id in self.G.keys():
            for x_id in self.G[e_id]:
                cur_id = e_id + "_" + x_id
                dict_loss_aec_metadata_pass1[cur_id] = {"start":0, "end":0}
        for e_id in self.G.keys():
            for x_id in self.G[e_id]:
                cur_id = e_id + "_" + x_id
                dict_loss_aec_metadata_pass2[cur_id] = {"start":0, "end":0}        
        #ts
        epoch = 1
        prev_loss_epoch = 0

        # try - start
        try:
            #while - start
            while True:
                epoch_start_time = time.time()
                #
                if epoch > self.max_epochs:
                    break
                #
                loss_recons = 0
                loss_aec_pass1 = 0
                loss_aec_pass2 = 0
                loss_regl = 0
                loss_epoch_pass1 = 0
                loss_epoch_pass2 = 0
                #
                if self.is_rand_batch_train:
                    for e_id in self.G.keys():
                        self.dict_e_idx[e_id] = torch.randperm(self.dict_batch_size[e_id])
                #
                #batching - step 1
                dict_e_batch_idx = {}
                for e_id in self.G.keys():
                    #print("e_id: ",e_id)
                    dict_temp = {}
                    for cur_batch in np.arange(self.num_batches):
                        #print("cur_batch: ",cur_batch)
                        start_idx = cur_batch * self.dict_mini_batch_size[e_id]
                        end_idx = (cur_batch+1) * self.dict_mini_batch_size[e_id]        
                        #print("start_idx: ",start_idx,", end_idx: ",end_idx)
                        #print("dict_e_idx[e_id]: ",dict_e_idx[e_id])
                        dict_temp[cur_batch] = self.dict_e_idx[e_id][start_idx:end_idx]
                    # print("dict_temp: ")
                    # print(dict_temp)
                    # print("#")
                    dict_e_batch_idx[e_id] = dict_temp
                #batching - step 2
                if self.is_rand_batch_train:
                    list_batch_ids = np.random.choice(self.num_batches,1)
                else:
                    list_batch_ids = np.arange(self.num_batches)
                #
                #print("list_batch_ids: ",list_batch_ids)
                #
                for cur_batch in list_batch_ids:
                    dict_cur_batch_idx_list = {}
                    for e_id in self.G.keys():
                        dict_cur_batch_idx_list[e_id] = dict_e_batch_idx[e_id][cur_batch]
                        self.dict_e_size_cur_batch[e_id] = len(dict_e_batch_idx[e_id][cur_batch])

                        # if self.mini_batch_size_frac < 1.0:
                        #     #with batching
                        #     dict_cur_batch_idx_list[e_id] = torch.randperm(self.dict_batch_size[e_id])[:self.dict_mini_batch_size[e_id]]
                        # elif self.mini_batch_size_frac == 1.0:
                        #     #without batching
                        #     dict_cur_batch_idx_list[e_id] = np.arange(self.dict_batch_size[e_id])
                        # else:
                        #     assert False," Unexpected mini_batch_size_frac: "+str(self.mini_batch_size_frac)
                    #
                    dict_cur_batch_X = {}
                    for x_id in self.X_meta.keys():
                        row_e_id = self.X_meta[x_id][0]
                        col_e_id = self.X_meta[x_id][1]
                        dict_cur_batch_X[x_id] = self.X_data[x_id][dict_cur_batch_idx_list[row_e_id],:][:,dict_cur_batch_idx_list[col_e_id]]
                    #############
                    # pass 1 - U
                    #############
                    # (1) Forward pass - VAEs
                    dict_C_enc = {}
                    dict_C_dec = {}
                    dict_mu = {}
                    dict_logvar = {}
                    dict_loss_aec_pass1 = {}
                    dict_loss_aec_pass2 = {}
                    for e_id in self.G.keys():
                        #print("#")
                        #print("e_id: ",e_id)
                        for x_id in self.G[e_id]:
                            #print("x_id: ",x_id)
                            cur_id = e_id + "_" + x_id
                            #print(">>> ",cur_id)
                            #print("#")
                            #
                            if self.X_dtype[x_id] == "real":
                                is_real = True
                            elif self.X_dtype[x_id] == "binary":
                                is_real = False
                            else:
                                assert False, "Unknown dtype: "+str(self.X_dtype[x_id])+" for x_id: "+str(x_id)
                            #    
                            C_batch = self.dict_C[cur_id][dict_cur_batch_idx_list[e_id],:]
                            enc_C, mu, logvar, dec_C = self.dict_vae[cur_id](C_batch)
                            #
                            dict_C_enc[cur_id] = enc_C
                            dict_mu[cur_id] = mu
                            dict_logvar[cur_id] = logvar
                            dict_C_dec[cur_id] = dec_C
                            # Note: dim = 1 i.e. always rows because we construct C by transposing if the entity is col (see dict_C construction)
                            dict_loss_aec_pass1[cur_id] = self.dict_e_loss_weight[e_id] * \
                                                    self.dict_loss_weight["aec"] * \
                                                    self.__get_vae_loss(dec_C, C_batch, mu, logvar, dim=1, is_real=is_real)
                            dict_loss_aec_pass1[cur_id].backward(retain_graph=True)
                            if epoch == 1:
                                dict_loss_aec_metadata_pass1[cur_id]["start"] = dict_loss_aec_pass1[cur_id].item()
                    #
                    # (2) forward pass U network
                    #construct input - Ucat
                    dict_U_cat = {}
                    for e_id in self.G.keys():
                        U_parts_list = []
                        for x_id in self.G[e_id]:
                            cur_id = e_id + "_" + x_id
                            U_parts_list.append(dict_mu[cur_id])
                        dict_U_cat[e_id] = torch.cat(U_parts_list,dim=1)
                    #
                    #do forward pass - Ucat to U
                    #dict_U = {}
                    for e_id in self.G.keys():
                        #if len(self.G[e_id]) > 1:
                        if len(self.G[e_id]) > 0:
                            #print("#")
                            #print("e_id:",e_id,", dict_U_cat[e_id].shape: ",dict_U_cat[e_id].shape)
                            #print("#")
                            U = self.dict_ffnu_cat[e_id](dict_U_cat[e_id])
                            self.dict_U[e_id] = U
                        else:
                            assert dict_U_cat[e_id].shape[1] == int(self.k/2.0),"Incorrect embedding dim for e_id: "+str(e_id)+", dict_U_cat[e_id].shape[1]: "+str(dict_U_cat[e_id].shape[1])+", k: "+str(self.k)
                            self.dict_U[e_id] = dict_U_cat[e_id]
                    #
                    #mat recons
                    dict_loss_mat = {}
                    for x_id in self.X_meta.keys():
                        row_e_id = self.X_meta[x_id][0]
                        col_e_id = self.X_meta[x_id][1]
                        #
                        if self.X_dtype[x_id] == "real":
                            is_real_cur_x = True
                        elif self.X_dtype[x_id] == "binary":
                            is_real_cur_x = False
                        else:
                            assert False, "Unknown dtype: "+str(self.X_dtype[x_id])+" for x_id: "+str(x_id)        
                        #
                        recons_X = torch.mm(self.dict_U[row_e_id],self.dict_U[col_e_id].transpose(1,0))
                        if is_real_cur_x:
                            #mse_criterion = torch.nn.MSELoss(reduction="mean")
                            #dict_loss_mat[x_id] = dict_loss_weight["mat"] * mse_criterion(recons_X,dict_cur_batch_X[x_id])
                            dict_loss_mat[x_id] = self.dict_loss_weight["mat"] * torch.norm(recons_X-dict_cur_batch_X[x_id])
                        else:
                            bce_loss = torch.nn.BCEWithLogitsLoss(reduction="mean")
                            dict_loss_mat[x_id] = self.dict_loss_weight["mat"] * bce_loss(recons_X,dict_cur_batch_X[x_id])
                        #
                        dict_loss_mat[x_id].backward(retain_graph=True)
                        if epoch == 1:
                            dict_loss_recons_metadata[x_id]["start"] = dict_loss_mat[x_id].item()

                    #opt data
                    optimizer_data.step()
                    optimizer_data.zero_grad()
                    #################
                    # pass 2 - Clust
                    #################
                    # (1) Forward pass - VAEs
                    # TODO:
                    # 2nd pass only thru encoders or both enc & dec?
                    dict_C_enc = {}
                    dict_C_dec = {}
                    dict_mu = {}
                    dict_logvar = {}
                    for e_id in self.G.keys():
                        #print("#")
                        #print("e_id: ",e_id)
                        for x_id in self.G[e_id]:
                            #print("x_id: ",x_id)
                            cur_id = e_id + "_" + x_id
                            #print(">>> ",cur_id)
                            #print("#")
                            #
                            if self.X_dtype[x_id] == "real":
                                is_real = True
                            elif self.X_dtype[x_id] == "binary":
                                is_real = False
                            else:
                                assert False, "Unknown dtype: "+str(self.X_dtype[x_id])+" for x_id: "+str(x_id)
                            #    
                            C_batch = self.dict_C[cur_id][dict_cur_batch_idx_list[e_id],:]
                            # Both enc and dec
                            enc_C, mu, logvar, dec_C = self.dict_vae[cur_id](C_batch)
                            # Only enc
                            #enc_C, mu, logvar, dec_C = dict_vae[cur_id](C_batch,is_decoder=False)
                            #assert dec_C == None
                            #
                            dict_C_enc[cur_id] = enc_C
                            dict_mu[cur_id] = mu
                            dict_logvar[cur_id] = logvar
                            dict_C_dec[cur_id] = dec_C
                            # Note: dim = 1 i.e. always rows because we construct C by transposing if the entity is col (see dict_C construction)
                            #dict_loss_aec_pass2[cur_id] = torch.tensor(0).float()
                            dict_loss_aec_pass2[cur_id] = self.dict_e_loss_weight[e_id] *\
                                                    self.dict_loss_weight["aec"] *\
                                                    self.__get_vae_loss(dec_C, C_batch, mu, logvar, dim=1, is_real=is_real)
                            dict_loss_aec_pass2[cur_id].backward(retain_graph=True)
                            if epoch == 1:
                                dict_loss_aec_metadata_pass2[cur_id]["start"] = dict_loss_aec_pass2[cur_id].item()
                    #
                    #forward pass U network
                    #construct input - Ucat
                    dict_U_cat = {}
                    for e_id in self.G.keys():
                        U_parts_list = []
                        for x_id in self.G[e_id]:
                            cur_id = e_id + "_" + x_id
                            U_parts_list.append(dict_mu[cur_id])
                        dict_U_cat[e_id] = torch.cat(U_parts_list,dim=1)
                    #
                    #forward pass - Ucat to U
                    #dict_U = {}
                    for e_id in self.G.keys():
                        #if len(self.G[e_id]) > 1:
                        if len(self.G[e_id]) > 0:
                            #print("#")
                            #print("e_id:",e_id,", dict_U_cat[e_id].shape: ",dict_U_cat[e_id].shape)
                            #print("#")
                            U = self.dict_ffnu_cat[e_id](dict_U_cat[e_id])
                            self.dict_U[e_id] = U
                        else:
                            assert dict_U_cat[e_id].shape[1] == int(self.k/2.0),"Incorrect embedding dim for e_id: "+str(e_id)+", dict_U_cat[e_id].shape[1]: "+str(dict_U_cat[e_id].shape[1])+", k: "+str(self.k)
                            self.dict_U[e_id] = dict_U_cat[e_id]
                    #
                    #forward pass - U to I,I_ortho
                    dict_I = {}
                    #dict_I_ortho = {}
                    for e_id in self.G.keys():
                        #print("#")
                        #print("e_id:",e_id,", dict_U[e_id].shape: ",dict_U[e_id].shape)
                        #print("#")
                        I, I_ortho = self.dict_ffn_clust[e_id](self.dict_U[e_id])
                        dict_I[e_id] = I
                        self.dict_I_ortho[e_id] = I_ortho
                    #
                    # calculating regl or clustering loss
                    #
                    dict_loss_regl = {}
                    #TODO: exp range and domain, can input be neg? can it be 0 or do we need eps?
                    # eps = 1e-10
                    for e_id in self.G.keys():
                        M_fro =  self.__get_M_fro(e_id,dict_cur_batch_idx_list) #+ eps
                        #print("M_fro.shape: ",M_fro.shape)
                        #print("M_fro: ",M_fro)
                        #print("#")
                        W = torch.exp(-0.5*M_fro) #Eqn (6) of spectralnet
                        #print("W: ",W)
                        D = torch.diag(torch.sum(W,dim=0))
                        #print("D: ",D)
                        L = D - W
                        #print("D.shape:",D.shape)
                        #print("D:",D)
                        #print("#")
                        #print("W.shape:",W.shape)
                        #print("W:",W)
                        #print("#")
                        #print("L.shape:",L.shape)
                        #print("L:",L)
                        #print("#")
                        #print("dict_I_ortho[e_id].shape: ",self.dict_I_ortho[e_id].shape)
                        #print("dict_I_ortho[e_id]: ",self.dict_I_ortho[e_id])
                        #print("#")
                        regl1 = self.dict_loss_weight["clust"] *\
                                torch.trace(torch.mm(torch.mm(self.dict_I_ortho[e_id].transpose(1,0),L),\
                                                                                 self.dict_I_ortho[e_id]))
                        dict_loss_regl[e_id] = regl1 
                        dict_loss_regl[e_id].backward(retain_graph=True)
                        if epoch == 1:
                            dict_loss_regl_metadata[e_id]["start"] = dict_loss_regl[e_id].item()
                    #opt clust
                    optimizer_clust.step()
                    optimizer_clust.zero_grad()
                    # debug - print
                    loss_recons += round(np.sum(list(dict_loss_mat.values())).item(),4)
                    loss_aec_pass1 += round(np.sum(list(dict_loss_aec_pass1.values())).item(),4)
                    loss_aec_pass2 += round(np.sum(list(dict_loss_aec_pass2.values())).item(),4)
                    loss_regl += round(np.sum(list(dict_loss_regl.values())).item(),4)
                    loss_epoch_pass1 += loss_aec_pass1 + loss_recons
                    loss_epoch_pass2 += loss_aec_pass2 + loss_regl
                epoch_end_time = time.time()
                # debug - plot
                dict_loss_recons_debug[epoch] = round(loss_recons/self.num_batches,4)
                dict_loss_aec_pass1_debug[epoch] = round(loss_aec_pass1/self.num_batches,4)
                dict_loss_aec_pass2_debug[epoch] = round(loss_aec_pass2/self.num_batches,4)
                dict_loss_regl_debug[epoch] = round(loss_regl/self.num_batches,4)
                dict_loss_epoch_pass1_debug[epoch] = round(loss_epoch_pass1/self.num_batches,4)
                dict_loss_epoch_pass2_debug[epoch] = round(loss_epoch_pass2/self.num_batches,4)
                # debug - print
                out_str_aec1 = ""
                for e_id in self.G.keys():
                    for x_id in self.G[e_id]:
                        cur_id = e_id + "_" + x_id
                        out_str_aec1+= str(", ")
                        out_str_aec1+= str(cur_id)
                        out_str_aec1+= str(":")
                        out_str_aec1+= str(round(dict_loss_aec_pass1[cur_id].item(),4))
                out_str_aec2 = ""
                for e_id in self.G.keys():
                    for x_id in self.G[e_id]:
                        cur_id = e_id + "_" + x_id
                        out_str_aec2+= str(", ")
                        out_str_aec2+= str(cur_id)
                        out_str_aec2+= str(":")
                        out_str_aec2+= str(round(dict_loss_aec_pass2[cur_id].item(),4))        
                out_str_recons = ""
                for x_id in self.X_meta.keys():
                    out_str_recons+= str(", ")
                    out_str_recons+= str(x_id)
                    out_str_recons+= str(":")
                    ##out_str_recons+= str(round(dict_loss_recons[x_id].item(),4))
                    out_str_recons+= str(round(dict_loss_mat[x_id].item(),4))
                out_str_regl = ""
                for e_id in self.G.keys():
                    out_str_regl+= str(", ")
                    out_str_regl+= str(e_id)
                    out_str_regl+= str(":")
                    out_str_regl+= str(round(dict_loss_regl[e_id].item(),4))    
                #
                #self.loss = loss_epoch_pass1 + loss_epoch_pass2
                self.loss = round((loss_epoch_pass1/self.num_batches) + (loss_epoch_pass2/self.num_batches),4) 
                #
                if epoch % 100 == 0 or epoch == 1:
                    #print("epoch:",epoch,", loss: ",round(self.loss,4),", lp1: ",loss_epoch_pass1,", lp2: ",loss_epoch_pass2)
                    print("epoch:",epoch,", loss: ",round(self.loss,4),"| aec1: ",round(loss_aec_pass1/self.num_batches,4),", rec: ",round(loss_recons/self.num_batches,4),"| aec2: ",round(loss_aec_pass2/self.num_batches,4),", reg: ",round(loss_regl/self.num_batches,4), ". Took ",round((epoch_end_time-epoch_start_time)/60.0,4)," mins.")
                # print("epoch:",epoch,", loss1: ",round(loss_epoch_pass1,4),\
                #       ", loss_aec_pass1:",loss_aec_pass1,", loss_recons:",loss_recons)
                # print("epoch:",epoch,", loss2: ",round(loss_epoch_pass2,4),\
                #       ", loss_aec_pass2:",loss_aec_pass2,", loss_regl: ",loss_regl)
                # print("epoch:",epoch," |recons: ",out_str_recons," |aec1: ",out_str_aec1," |aec2: ",out_str_aec2," |regl: ",out_str_regl)
                # print("#")
                #
                epoch+=1
                if self.convg_thres != None:
                    if self.__is_converged(prev_loss_epoch,self.loss,self.convg_thres, epoch):
                        print("**train converged**")
                        break
                    prev_loss_epoch = self.loss
                #while - end
            self.__copy_params()
            #
            # Final pass after training to obtain U for the whole dataset
            # and X prime
            #1
            dict_C_enc = {}
            #dict_C_dec = {}
            #dict_mu = {}
            dict_logvar = {}
            for e_id in self.G.keys():
                #print("#")
                #print("e_id: ",e_id)
                for x_id in self.G[e_id]:
                    #print("x_id: ",x_id)
                    cur_id = e_id + "_" + x_id
                    enc_C, mu, logvar, dec_C = self.dict_vae[cur_id](self.dict_C[cur_id])
                    dict_C_enc[cur_id] = enc_C
                    self.dict_mu[cur_id] = mu
                    dict_logvar[cur_id] = logvar
                    self.dict_C_dec[cur_id] = dec_C
            #2
            dict_U_cat = {}
            for e_id in self.G.keys():
                U_parts_list = []
                for x_id in self.G[e_id]:
                    cur_id = e_id + "_" + x_id
                    U_parts_list.append(self.dict_mu[cur_id])
                dict_U_cat[e_id] = torch.cat(U_parts_list,dim=1)

            #do forward pass - Ucat to I
            #dict_U = {}
            for e_id in self.G.keys():
                #if len(self.G[e_id]) > 1:
                if len(self.G[e_id]) > 0:
                    #print("#")
                    #print("e_id:",e_id,", dict_U_cat[e_id].shape: ",dict_U_cat[e_id].shape)
                    #print("#")
                    U = self.dict_ffnu_cat[e_id](dict_U_cat[e_id])
                    self.dict_U[e_id] = U
                else:
                    assert dict_U_cat[e_id].shape[1] == int(self.k/2.0),"Incorrect embedding dim for e_id: "+str(e_id)+", dict_U_cat[e_id].shape[1]: "+str(dict_U_cat[e_id].shape[1])+", k: "+str(self.k)
                    self.dict_U[e_id] = dict_U_cat[e_id]    
            #3
            dict_I = {}
            #dict_I_ortho = {}
            for e_id in self.G.keys():
                #print("#")
                #print("e_id:",e_id,", dict_U[e_id].shape: ",dict_U[e_id].shape)
                #print("#")
                I, I_ortho = self.dict_ffn_clust[e_id](self.dict_U[e_id])
                dict_I[e_id] = I
                self.dict_I_ortho[e_id] = I_ortho
            #4
            #dict_recons_X = {}
            #dict_A = {}
            for x_id in self.X_meta.keys():
                row_e_id = self.X_meta[x_id][0]
                col_e_id = self.X_meta[x_id][1]
                #1
                # recons_X_prev = dict_IW[row_e_id].mm(dict_IW[col_e_id].transpose(1,0))
                # X_mu_, X_theta, X_pi = N_ffn_zinb_dict[x_id](recons_X_prev)
                # X_mu = torch.mm(X_size_fac[x_id],X_mu_)
                # recons_X = X_mu
                # A = dict_W[row_e_id].mm(dict_W[col_e_id].transpose(1,0))
                #2
                #
                C_row_e_id = torch.from_numpy(self.__get_rigorous_C(self.dict_I_ortho[row_e_id], self.dict_num_clusters[row_e_id])).float().cuda()
                C_col_e_id = torch.from_numpy(self.__get_rigorous_C(self.dict_I_ortho[col_e_id], self.dict_num_clusters[col_e_id])).float().cuda()
                A = torch.mm(torch.mm(C_row_e_id.transpose(1,0),\
                                    self.X_data[x_id]), C_col_e_id)
                #
                #A = torch.mm(torch.mm(self.dict_I_ortho[row_e_id].transpose(1,0),\
                #                    self.X_data[x_id]),self.dict_I_ortho[col_e_id])
                #A = torch.mm(torch.mm(torch.pinverse(self.dict_I_ortho[row_e_id]),\
                #                    self.X_data[x_id]),torch.pinverse(self.dict_I_ortho[col_e_id]).transpose(1,0))
                recons_X = torch.mm(torch.mm(self.dict_I_ortho[row_e_id],A),\
                                  self.dict_I_ortho[col_e_id].transpose(1,0))
                # X_mu_, X_theta, X_pi = N_ffn_zinb_dict[x_id](recons_X_prev)
                # X_mu = torch.mm(X_size_fac[x_id],X_mu_)
                # recons_X = X_mu
                self.dict_recons_X[x_id] = recons_X
                self.dict_A[x_id] = A
                #        
        # try - end                    
        except KeyboardInterrupt as err:
            print("#")
            print("err:")
            print(err)
            print(err.args)
            print("#")
            #
            self.__copy_params()
            #
            #pass
            #
            # Final pass after training to obtain U for the whole dataset
            # and X prime
            #
            #1
            dict_C_enc = {}
            #dict_C_dec = {}
            #dict_mu = {}
            dict_logvar = {}
            for e_id in self.G.keys():
                #print("#")
                #print("e_id: ",e_id)
                for x_id in self.G[e_id]:
                    #print("x_id: ",x_id)
                    cur_id = e_id + "_" + x_id
                    enc_C, mu, logvar, dec_C = self.dict_vae[cur_id](self.dict_C[cur_id])
                    dict_C_enc[cur_id] = enc_C
                    self.dict_mu[cur_id] = mu
                    dict_logvar[cur_id] = logvar
                    self.dict_C_dec[cur_id] = dec_C
            #2
            dict_U_cat = {}
            for e_id in self.G.keys():
                U_parts_list = []
                for x_id in self.G[e_id]:
                    cur_id = e_id + "_" + x_id
                    U_parts_list.append(self.dict_mu[cur_id])
                dict_U_cat[e_id] = torch.cat(U_parts_list,dim=1)

            #do forward pass - Ucat to I
            #dict_U = {}
            for e_id in self.G.keys():
                #if len(self.G[e_id]) > 1:
                if len(self.G[e_id]) > 0:
                    #print("#")
                    #print("e_id:",e_id,", dict_U_cat[e_id].shape: ",dict_U_cat[e_id].shape)
                    #print("#")
                    U = self.dict_ffnu_cat[e_id](dict_U_cat[e_id])
                    self.dict_U[e_id] = U
                else:
                    assert dict_U_cat[e_id].shape[1] == int(self.k/2.0),"Incorrect embedding dim for e_id: "+str(e_id)+", dict_U_cat[e_id].shape[1]: "+str(dict_U_cat[e_id].shape[1])+", k: "+str(self.k)
                    self.dict_U[e_id] = dict_U_cat[e_id]
                # #print("#")
                # #print("e_id:",e_id,", dict_U_cat[e_id].shape: ",dict_U_cat[e_id].shape)
                # #print("#")
                # U = self.dict_ffnu_cat[e_id](dict_U_cat[e_id])
                # self.dict_U[e_id] = U     
            #3
            dict_I = {}
            #dict_I_ortho = {}
            for e_id in self.G.keys():
                #print("#")
                #print("e_id:",e_id,", dict_U[e_id].shape: ",dict_U[e_id].shape)
                #print("#")
                I, I_ortho = self.dict_ffn_clust[e_id](self.dict_U[e_id])
                dict_I[e_id] = I
                self.dict_I_ortho[e_id] = I_ortho

            #4
            #dict_recons_X = {}
            #dict_A = {}
            for x_id in self.X_meta.keys():
                row_e_id = self.X_meta[x_id][0]
                col_e_id = self.X_meta[x_id][1]
                #1
                # recons_X_prev = dict_IW[row_e_id].mm(dict_IW[col_e_id].transpose(1,0))
                # X_mu_, X_theta, X_pi = N_ffn_zinb_dict[x_id](recons_X_prev)
                # X_mu = torch.mm(X_size_fac[x_id],X_mu_)
                # recons_X = X_mu
                # A = dict_W[row_e_id].mm(dict_W[col_e_id].transpose(1,0))
                #2
                #
                C_row_e_id = torch.from_numpy(self.__get_rigorous_C(self.dict_I_ortho[row_e_id], self.dict_num_clusters[row_e_id])).float().cuda()
                C_col_e_id = torch.from_numpy(self.__get_rigorous_C(self.dict_I_ortho[col_e_id], self.dict_num_clusters[col_e_id])).float().cuda()
                A = torch.mm(torch.mm(C_row_e_id.transpose(1,0),\
                                    self.X_data[x_id]), C_col_e_id)
                #
                #A = torch.mm(torch.mm(self.dict_I_ortho[row_e_id].transpose(1,0),\
                #                    self.X_data[x_id]),self.dict_I_ortho[col_e_id])
                #A = torch.mm(torch.mm(torch.pinverse(self.dict_I_ortho[row_e_id]),\
                #                    self.X_data[x_id]),torch.pinverse(self.dict_I_ortho[col_e_id]).transpose(1,0))
                recons_X = torch.mm(torch.mm(self.dict_I_ortho[row_e_id],A),\
                                  self.dict_I_ortho[col_e_id].transpose(1,0))
                # X_mu_, X_theta, X_pi = N_ffn_zinb_dict[x_id](recons_X_prev)
                # X_mu = torch.mm(X_size_fac[x_id],X_mu_)
                # recons_X = X_mu
                self.dict_recons_X[x_id] = recons_X
                self.dict_A[x_id] = A
            pass        
