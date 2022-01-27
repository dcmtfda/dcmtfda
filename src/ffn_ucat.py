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

class ffnucat(nn.Module): #VAE

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

    def __init__(self,input_dim,k,e_actf,num_clusters):
        super(ffnucat, self).__init__()
        #encoding layers
        num_iw_out = num_clusters
        #
        clust_layers_ucat = collections.OrderedDict()
        #
        if self.is_bn:
            clust_layers_ucat["ucat"] = nn.Linear(int(input_dim), int(k/2.0),bias=False)
            clust_layers_ucat["ucat-bn"] = nn.BatchNorm1d(num_features=int(k/2.0))
        else:
            clust_layers_ucat["ucat"] = nn.Linear(int(input_dim), int(k/2.0),bias=True)
        clust_layers_ucat["ucat-actf"] = self.get_actf(e_actf)
        #
        self.ucat =  nn.Sequential(clust_layers_ucat)
        #
        print("U:")
        print(self.ucat)
        print("#")

    def forward(self, x):
        U = self.ucat(x)
        return U