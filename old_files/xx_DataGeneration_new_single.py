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


# "data generation parameters  (for examples)"

n_train_g = 500
n_test_g = 50
# node_nos_low = 100
# nodee_nos_high = 500
nnodes=500
# rand_node_nos = random.randint(node_nos_low,nodee_nos_high)
# prob_edge = 2*1.2/(rand_node_nos-1)
prob_edge = 1.2/(100-1)



# prob_edge = 0.05


"data generation parameters  (small graphs)"

# n_train_g = 5000
# n_test_g = 50
# node_nos_low = 400
# nodee_nos_high = 600


# num_nodes = 50
# num_edges = 70

max_node = 500  # initialization of maximum number of edges in all the graphs





# pfilename="orig.edg"

new_graph = gnp_random_connected_graph(nnodes, prob_edge)
max_model_no = len(new_graph.nodes)*1
list_train_graph_orig = [new_graph.to_directed()]
list_test_graph_orig  = [new_graph.to_directed()]

list_edge_index_train_orig,list_edge_weight_train_orig, eff_mat_train_orig, list_fm_train_orig = get_adjacency_feature_centrality(list_train_graph_orig, max_model_no)#,total_sum_orig)
# list_edge_index_test_orig,list_edge_weight_test_orig, eff_mat_test_orig, list_fm_test_orig = get_adjacency_feature_centrality(list_test_graph_orig, max_model_no)#,total_sum_orig)
  
list_edge_index_test_orig=list_edge_index_train_orig.copy()
list_edge_weight_test_orig=list_edge_weight_train_orig.copy()
eff_mat_test_orig=eff_mat_train_orig.copy()
list_fm_test_orig=list_fm_train_orig.copy()





with open(r'D:/single_graph.pickle', 'wb') as handle:
    pickle.dump([new_graph,list_edge_index_train_orig,list_edge_weight_train_orig, eff_mat_train_orig, list_fm_train_orig, list_edge_index_test_orig,list_edge_weight_test_orig, eff_mat_test_orig, list_fm_test_orig], handle, protocol=pickle.HIGHEST_PROTOCOL)







with open(r'D:/single_graph.pickle',"rb") as f:
     # list_adj_deg_train_shuffled, list_adj_wt_train_shuffled,list_num_edge_train, bc_mat_train_shuffled, list_fm_train_shuffled, list_n_seq_train, list_n_unshuffle_train,list_adj_deg_test_shuffled, list_adj_wt_test_shuffled,list_num_edge_test, bc_mat_test_shuffled, list_fm_test_shuffled, list_n_seq_test, list_n_unshuffle_test=pickle.load(f)
     new_graph,list_edge_index_train_orig,list_edge_weight_train_orig, eff_mat_train_orig, list_fm_train_orig, list_edge_index_test_orig,list_edge_weight_test_orig, eff_mat_test_orig, list_fm_test_orig=pickle.load(f)
 



n_train_g = 5
n_test_g = 2
max_model_no = len(new_graph.nodes)*1
list_train_graph = list()


list_test_graph= list()

# list_node_weights_train=[]
# list_node_weights_test=[]

#create list for training graphs
print("Create training graphs")
for i in range(n_train_g):
        print(i)
        
        
        # print(len(new_graph.edges()))
        G0=new_graph.copy()
        while True:
            tempg=new_graph.copy()
            ndel=list(np.random.randint(0,len(new_graph.edges())-1,random.randint(0,10)))
            ndel.sort(reverse=True)
            # ndel=i
            try:
                for j in range(len(ndel)):
                    u,v=list(new_graph.edges())[ndel[j]][0:2]
                    
                    tempg.remove_edge(u, v)
                
                if nx.is_connected(tempg):
                    for j in range(len(ndel)):
                        u,v=list(new_graph.edges())[ndel[j]][0:2]
                    
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
        # list_train_graph_orig.append(new_graph.to_directed())
        
        
        
print("Create Testing graphs")
for i in range(n_test_g):
        print(i)
        
        # new_graph = gnp_random_connected_graph(nnodes, prob_edge)
        # print(len(new_graph.edges()))
        G0=new_graph.copy()
        while True:
            tempg=new_graph.copy()
            ndel=list(np.random.randint(0,len(new_graph.edges())-1,random.randint(0,10)))
            ndel.sort(reverse=True)
            # ndel=i
            try:
                for j in range(len(ndel)):
                    u,v=list(new_graph.edges())[ndel[j]][0:2]
                    
                    tempg.remove_edge(u, v)
                
                if nx.is_connected(tempg):
                    for j in range(len(ndel)):
                        u,v=list(new_graph.edges())[ndel[j]][0:2]
                    
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
        # list_test_graph_orig.append(new_graph.to_directed())    
    
        
# print("Create testing graphs")
# for i in range(n_test_g):
#         G0=new_graph.copy()
#         while True:
#             tempg=new_graph.copy()
#             ndel=random.randint(0,len(new_graph.edges())-1)
#             # ndel=i
#             u,v=list(new_graph.edges())[ndel][0:2]
            
#             tempg.remove_edge(u, v)
            
#             if nx.is_connected(tempg):
#                 G0.edges[u,v]['weight'] = 1000000000
#                 break
#             else:
#                 print("try again")
#                 continue
            
#         list_test_graph.append(G0.to_directed())    
        # list_test_graph_orig.append(new_graph)
    
    
    
    
#     rand_node_nos = random.randint(node_nos_low,nodee_nos_high)
#     # prob_edge = 2*1.2/(rand_node_nos-1)
#     prob_edge = 1.2/(rand_node_nos-1)
    
#     new_graph = gnp_random_connected_graph(rand_node_nos, prob_edge)
    
#     if len(new_graph.edges)>max_edge:
#         max_edge=len(new_graph.edges)
    
#     list_train_graph.append(new_graph)
  
# #create list for test graphs
# print("Create testing graphs")
# for i in range(n_test_g):
    
#     rand_node_nos = random.randint(node_nos_low,nodee_nos_high)
#     # prob_edge = 2*1.2/(rand_node_nos-1)
#     prob_edge = 1.2/(rand_node_nos-1)
    
#     new_graph = gnp_random_connected_graph(rand_node_nos, prob_edge)  
    
#     if len(new_graph.edges)>max_edge:
#         max_edge=len(new_graph.edges)
    
#     list_test_graph.append(new_graph)


# """ Here we can put the model number which should be greater than the max edge"""

# # max_model_no = max_edge ## we have to change the script accordingly

# "can be also specified by the user"

# max_model_no = math.ceil(max_edge/100)*100
max_model_no = len(new_graph.nodes)*1

# max_model_no = 141





# total_sum_orig=distance_sum(new_graph)


# effs=[]

# for i in range(len(list_train_graph)):
#     total_sum_new=distance_sum(list_train_graph[i])
#     eff=total_sum_orig/total_sum_new


#     effs.append(eff)



# "checking the node and edge nos"

# list_num_node_train = []
# list_num_node_test = []

# list_num_edge_train = []
# list_num_edge_test = []


# for i in range(len(list_train_graph)):
#     list_num_node_train.append(len(list_train_graph[i].nodes))
#     list_num_edge_train.append(len(list_train_graph[i].edges))
    

# for i in range(len(list_test_graph)):
#     list_num_node_test.append(len(list_test_graph[i].nodes))
#     list_num_edge_test.append(len(list_test_graph[i].edges))

# total_sum_orig=distance_sum(new_graph.to_directed())

# edge_index_orig = list(new_graph.to_directed().edges)
# edge_weight_orig = [new_graph.to_directed()[u][v]['weight'] for u, v in new_graph.to_directed().edges]
# edge_index = torch.tensor(list(new_graph.edges)).t().contiguous()
# edge_weight = torch.tensor([new_graph[u][v]['weight'] for u, v in new_graph.edges])



list_edge_index_train,list_edge_weight_train, eff_mat_train= get_adjacency_centrality(list_train_graph, max_model_no)#,total_sum_orig)
list_edge_index_test,list_edge_weight_test, eff_mat_test = get_adjacency_centrality(list_test_graph, max_model_no)#,total_sum_orig)






#EXECUTE FROM HERE




eff_mat_final_train=1-(eff_mat_train-eff_mat_train_orig)/eff_mat_train_orig
eff_mat_final_test=1-(eff_mat_test-eff_mat_test_orig)/eff_mat_test_orig



# # eff_mat_final_complete=np.concatenate([eff_mat_final_train,eff_mat_final_test],axis=1)



eff_mat_final_train_scaled = (eff_mat_final_train - 0.9822141302873455) / (1 - 0.9822141302873455)
eff_mat_final_test_scaled = (eff_mat_final_test - 0.9822141302873455) / (1 - 0.9822141302873455)

# eff_mat_final_train_scaled=eff_mat_final_train_scaled.reshape(-1,1)
# eff_mat_final_test_scaled=eff_mat_final_test_scaled.reshape(-1,1)



with open(r'graphs/train/graph_data_500_to_500_nodes_NEW_SINGLE_12_package_10.pickle', 'wb') as handle:
    pickle.dump([list_train_graph, list_edge_index_train,list_edge_weight_train, eff_mat_final_train_scaled,list_fm_train_orig,list_edge_index_train_orig,list_edge_weight_train_orig], handle, protocol=pickle.HIGHEST_PROTOCOL)


with open(r'graphs/test/graph_data_500_to_500_nodes_NEW_SINGLE_12_package_10_test.pickle', 'wb') as handle:
    pickle.dump([list_test_graph, list_edge_index_test,list_edge_weight_test, eff_mat_final_test_scaled, list_fm_test_orig,list_edge_index_test_orig,list_edge_weight_test_orig], handle, protocol=pickle.HIGHEST_PROTOCOL)


# 




# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(eff_mat_final_complete.reshape(-1,1))

# eff_mat_final_complete_scaled=scaler.transform(eff_mat_final_complete.reshape(-1,1))


# eff_mat_final_train_scaled=eff_mat_final_complete_scaled[0:500,0].reshape(1,-1)
# eff_mat_final_test_scaled=eff_mat_final_complete_scaled[500:,0].reshape(1,-1)

# eff_mat_final_train_scaled=eff_mat_final_complete_scaled[0,0:500].reshape(1,-1)
# eff_mat_final_test_scaled=eff_mat_final_complete_scaled[0,500:].reshape(1,-1)























# eff_mat_final_train=1-(eff_mat_train-eff_mat_train_orig)/eff_mat_train_orig
# eff_mat_final_test=1-(eff_mat_test-eff_mat_test_orig)/eff_mat_test_orig



# eff_mat_final_complete=np.concatenate([eff_mat_final_train,eff_mat_final_test],axis=1)

# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler()
# scaler.fit(eff_mat_final_complete.reshape(-1,1))

# eff_mat_final_complete_scaled=scaler.transform(eff_mat_final_complete.reshape(-1,1))


# eff_mat_final_train_scaled=eff_mat_final_complete_scaled[0:500,0].reshape(1,-1)
# eff_mat_final_test_scaled=eff_mat_final_complete_scaled[500:,0].reshape(1,-1)

# # list_fm_train=[]
# # list_fm_test=[]

# # for i in range(len(list_train_graph)):
# #     list_fm_train.append(list_fm_train_orig)
    
# # for i in range(len(list_test_graph)):
# #     list_fm_test.append(list_fm_test_orig)


# # a=list_fm_train_orig[0].to_dense().cpu().detach().numpy()
# # b=list_adj_train[0].to_dense().cpu().detach().numpy()

# with open(r'D:/graph_data_500_to_500_nodes_package_01.pickle', 'wb') as handle:
#     pickle.dump([list_train_graph, list_edge_index_train,list_edge_weight_train, eff_mat_final_train_scaled,list_fm_train_orig,list_edge_index_train_orig,list_edge_weight_train_orig], handle, protocol=pickle.HIGHEST_PROTOCOL)


# with open(r'D:/graph_data_500_to_500_nodes_package_01_test.pickle', 'wb') as handle:
#     pickle.dump([list_test_graph, list_edge_index_test,list_edge_weight_test, eff_mat_final_test_scaled, list_fm_test_orig,list_edge_index_test_orig,list_edge_weight_test_orig], handle, protocol=pickle.HIGHEST_PROTOCOL)
