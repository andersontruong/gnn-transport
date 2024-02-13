#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 09:59:19 2022

@author: debasishjana
"""

# import numpy as np
import pickle
import networkx as nx
# import torch
# from utils import *
import random
# import torch.nn as nn
# from model_bet import GNN_Bet
# torch.manual_seed(20)
# import argparse

# from scipy import sparse
# import scipy
# import utils2
# from itertools import combinations, groupby
# import pandas as pd
import math
# import time
import numpy as np
from xx_AllFunctions import gnp_random_connected_graph, get_adjacency_feature_centrality,distance_sum, get_adjacency_centrality
# import tensorflow


"data generation parameters  (for examples)"

n_train_g = 2000
n_test_g = 200

nnodes=500

prob_edge = 1.2/(100-1)



# prob_edge = 0.05


"data generation parameters  (small graphs)"


max_node = 500  # initialization of maximum number of edges in all the graphs

list_train_graph_orig = []
list_train_graph = list()

list_test_graph_orig  = []
list_test_graph= list()

# list_node_weights_train=[]
# list_node_weights_test=[]

#create list for training graphs

new_graph = gnp_random_connected_graph(nnodes, prob_edge)
list_train_graph_orig.append(new_graph.to_directed())
print("Create training graphs")
for i in range(n_train_g):
        print(i)
        
        
        # print(len(new_graph.edges()))
        G0=new_graph.copy()
        while True:
            tempg=new_graph.copy()
            ndel=list(np.random.randint(0,len(new_graph.edges())-1,random.randint(0,20)))
            ndel.sort(reverse=True)
            # ndel=i
            try:
                for j in range(len(ndel)):
                    u,v=list(new_graph.edges())[ndel[j]][0:2]
                    
                    tempg.remove_edge(u, v)
                
                if nx.is_connected(tempg):
                    for j in range(len(ndel)):
                        u,v=list(new_graph.edges())[ndel[j]][0:2]
                        # G0.remove_edge(u, v)
                        G0.edges[u,v]['weight'] = 1000000000
                    break
                else:
                    print("try again")
                    continue
            except KeyError:
                continue
            except  nx.NetworkXError:
                continue
        # list_node_weights_train.append(np.random.random(max_node).reshape(1,-1))
        list_train_graph.append(G0.to_directed())
        
        
        
list_test_graph_orig.append(new_graph.to_directed())      
print("Create Testing graphs")
for i in range(n_test_g):
        print(i)
        
        # new_graph = gnp_random_connected_graph(nnodes, prob_edge)
        # print(len(new_graph.edges()))
        G0=new_graph.copy()
        while True:
            tempg=new_graph.copy()
            ndel=list(np.random.randint(0,len(new_graph.edges())-1,random.randint(0,20)))
            ndel.sort(reverse=True)
            # ndel=i
            try:
                for j in range(len(ndel)):
                    u,v=list(new_graph.edges())[ndel[j]][0:2]
                    
                    tempg.remove_edge(u, v)
                
                if nx.is_connected(tempg):
                    for j in range(len(ndel)):
                        u,v=list(new_graph.edges())[ndel[j]][0:2]
                        # G0.remove_edge(u, v)
                        G0.edges[u,v]['weight'] = 1000000000
                    break
                else:
                    print("try again")
                    continue
            except KeyError:
                continue
            except  nx.NetworkXError:
                continue
        # list_node_weights_test.append(np.random.random(max_node).reshape(1,-1))
        list_test_graph.append(G0.to_directed())
            
    
       
max_model_no = len(new_graph.nodes)*1



pfilename="orig1.edg"


list_edge_index_train_orig,list_edge_weight_train_orig, eff_mat_train_orig, list_fm_train_orig = get_adjacency_feature_centrality(list_train_graph_orig, max_model_no,pfilename=pfilename)#,total_sum_orig)
list_edge_index_test_orig,list_edge_weight_test_orig, eff_mat_test_orig, list_fm_test_orig = get_adjacency_feature_centrality(list_test_graph_orig, max_model_no,pfilename=pfilename)#,total_sum_orig)
  
list_edge_index_train,list_edge_weight_train, eff_mat_train= get_adjacency_centrality(list_train_graph, max_model_no)#,total_sum_orig)
list_edge_index_test,list_edge_weight_test, eff_mat_test = get_adjacency_centrality(list_test_graph, max_model_no)#,total_sum_orig)


eff_mat_final_train=1-(eff_mat_train-eff_mat_train_orig)/eff_mat_train_orig
eff_mat_final_test=1-(eff_mat_test-eff_mat_test_orig)/eff_mat_test_orig



eff_mat_final_complete=np.concatenate([eff_mat_final_train,eff_mat_final_test],axis=1)



eff_mat_final_complete_scaled = (eff_mat_final_complete - np.min(eff_mat_final_complete)) / (1 - np.min(eff_mat_final_complete))




eff_mat_final_train_scaled=eff_mat_final_complete_scaled[0,0:2000].reshape(1,-1)
eff_mat_final_test_scaled=eff_mat_final_complete_scaled[0,2000:].reshape(1,-1)


with open(r'D:\xx_clean_code_vul_fin\graph_data\graph_data_train.pickle', 'wb') as handle:
    pickle.dump([list_train_graph, list_edge_index_train,list_edge_weight_train, eff_mat_final_train_scaled,list_fm_train_orig,list_edge_index_train_orig,list_edge_weight_train_orig], handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(r'D:\xx_clean_code_vul_fin\graph_data\graph_data_test.pickle', 'wb') as handle:
    pickle.dump([list_test_graph, list_edge_index_test,list_edge_weight_test, eff_mat_final_test_scaled, list_fm_train_orig,list_edge_index_test_orig,list_edge_weight_test_orig], handle, protocol=pickle.HIGHEST_PROTOCOL)
