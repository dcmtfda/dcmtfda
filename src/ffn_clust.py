import numpy as np
import collections
import sys
#
import torch
from torch import nn

import random

np.random.seed(0)
torch.manual_seed(0)
random.seed(0)

class ffnclust(nn.Module): #VAE

    is_bn = False
  
    def get_actf(self,actf_name):
        if actf_name == "relu":
            A = nn.ReLU()
        elif actf_name == "sigma":
            A = nn.Sigmoid()
        elif actf_name == "tanh":
            A = nn.Tanh()
        elif actf_name == "lrelu":
            A = nn.LeakyReLU()
        elif actf_name == "softmax":
            A = nn.Softmax(dim=1)
        else:
            print("Unknown activation function: ",actf_name)
            sys.exit(1)
        return A

    def __init__(self,input_dim, num_clusters, k):
        super(ffnclust, self).__init__()
        #
        clust_layers_dict1 = collections.OrderedDict()
        #
        #Converts reprn of dim k to dim num_clusters
        if self.is_bn:
            clust_layers_dict1["clust1"] = nn.Linear(int(input_dim), int(num_clusters),bias=False)
            clust_layers_dict1["clust1-bn"] = nn.BatchNorm1d(num_features=num_clusters)
        else:
            clust_layers_dict1["clust1"] = nn.Linear(int(input_dim), int(num_clusters),bias=True)
        #clust_layers_dict1["clust1-softmax"] = self.get_actf("softmax")
        clust_layers_dict1["clust1-actf"] = self.get_actf("tanh")
        #
        self.i1 =  nn.Sequential(clust_layers_dict1)
        print("I:")
        print(self.i1)
        print("#")

    #
    # Based on SpectralNet, Shahman et. al 
    # https://arxiv.org/abs/1801.01587
    # https://github.com/KlugerLab/SpectralNet/blob/master/src/core/layer.py
    #
    def get_ortho_I(self, x,epsilon = 1e-10):
        try:
            x_2 = torch.mm(x.transpose(1,0), x)
            I = torch.eye(x.shape[1])
            if x_2.is_cuda:
                I = I.cuda()
            x_2 += (I*epsilon)
            #
            #print("x_2.shape: ",x_2.shape)
            #print("#")
            #print("x_2: ",x_2)
            #print("#")
            #
            L = torch.cholesky(x_2)
            ortho_weights = torch.inverse(L).transpose(1,0) #* torch.sqrt(torch.tensor(x.shape[0]).float())
            x_ortho = torch.mm(x,ortho_weights)
            # print("x_ortho.shape: ",x_ortho.shape)
            # print("x:")
            # print(x)
            # print("#")
            # print("prev x_ortho: ")
            # print(x_ortho)
            # print("#")
            # # print("torch.mm(x_ortho.transpose(1,0),x_ortho): ")
            # print(torch.mm(x_ortho.transpose(1,0),x_ortho))
            # print("#")
            # print("torch.sum(torch.diag(torch.mm(x_ortho.transpose(1,0),x_ortho))): ",torch.sum(torch.diag(torch.mm(x_ortho.transpose(1,0),x_ortho))))
            # print("x_ortho.shape[1]: ",x_ortho.shape[1])
            act = int(np.round(torch.sum(torch.diag(torch.mm(x_ortho.transpose(1,0),x_ortho))).data.cpu().numpy()))
            exp = int(x_ortho.shape[1])
            #assert act == exp, "act: "+str(act)+", exp: "+str(exp)
            # if act != exp:
            #     print("get_ortho_I:")
            #     print("WARNING:"+"act: "+str(act)+", exp: "+str(exp))
            #     print("torch.diag(torch.mm(x_ortho.transpose(1,0),x_ortho)): ")
            #     print(torch.diag(torch.mm(x_ortho.transpose(1,0),x_ortho)))
            #     print("#")
        except RuntimeError as e:
            print("x:")
            print(x)
            raise e  
            #print("#")
            #print("WARNING: setting x as x_ortho. ")
            #print(e)
            #traceback.print_exc()
            #x_ortho = x
            #x_ortho = None
        return x_ortho

    def __get_ortho_norm(self,x_enc):
        x_enc_norm = x_enc / torch.sqrt(torch.diag(torch.mm(x_enc.transpose(1,0),x_enc)))
        return x_enc_norm

    def forward(self, x, is_decode=False):
        x_enc_i = self.i1(x)
        x_enc_i_ortho = self.get_ortho_I(x_enc_i)
        #x_enc_i_ortho = self.__get_ortho_norm(x_enc_i)
        return x_enc_i, x_enc_i_ortho
