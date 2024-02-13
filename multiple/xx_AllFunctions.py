#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 17:27:10 2022

@author: debasishjana
"""


import numpy as np
import pickle
import networkx as nx
import torch
# from utils import *
import random
import torch.nn as nn
# from model_bet import GNN_Bet
torch.manual_seed(20)
import argparse
import pickle
from scipy import sparse
import scipy
# import utils2
from itertools import combinations, groupby
import pandas as pd
import math
import time


import torch
import torch.nn as nn
import time
from scipy.stats import kendalltau
import random

from scipy.sparse import csr_matrix,hstack,vstack,identity
from node2vec.edges import AverageEmbedder
from node2vec import Node2Vec
from scipy import sparse

from gensim.models import Word2Vec

from pecanpy import pecanpy







"""FUNCTIONS REQUIRED FOR TRAINING DATA GENERATION"""


"FUNCTION FOR RANDOM CONNECTED GRAPH GENERATION"

def gnp_random_connected_graph(n, p):
    """
    Generates a random undirected graph, similarly to an Erdős-Rényi 
    graph, but enforcing that the resulting graph is conneted
    """
    edges = combinations(range(n), 2)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    if p <= 0:
        return G
    if p >= 1:
        return nx.complete_graph(n, create_using=G)
    for _, node_edges in groupby(edges, key=lambda x: x[0]):
        node_edges = list(node_edges)
        random_edge = random.choice(node_edges)
        G.add_edge(*random_edge)
        for e in node_edges:
            if random.random() < p:
                G.add_edge(*e)
    for (u, v) in G.edges():
        G.edges[u,v]['weight'] = random.random()*100
    return G



"FUNCTION FOR EDGE ADJACENCY MATRIX GENERATION"


def get_adj_matrix(graph):
    
    adj=nx.adjacency_matrix(graph, nodelist=None, dtype=None, weight='weight')
    
    
    # edges_array=np.array(graph.edges())
    
    # # edge_adj_sparse=csr_matrix((len(edges_array), len(edges_array)), dtype=np.int8)
    
    # row = []
    # col = []
    # data = []
    
    
    
    # for i in range(len(edges_array)-1):
    #     # i=0
    #     start_node=edges_array[i,0]
    #     end_node=edges_array[i,1]
    #     start_inds=list(np.where((edges_array[i+1:,0]==start_node) | (edges_array[i+1:,0]==end_node))[0]+(i+1))
    #     end_inds=list(np.where((edges_array[i+1:,1]==start_node) | (edges_array[i+1:,1]==end_node))[0]+(i+1))
        
    
    #     col_list=list(start_inds+end_inds)
    #     row_list=[i]*len(col_list)
    #     val_list=[1]*len(col_list)
        
    #     row.extend(row_list)
    #     col.extend(col_list)
    #     data.extend(val_list)
    
    
    # row_final = row.copy()
    # col_final = col.copy()
    # data_final = data.copy()
    
    
    # row_final.extend(col)
    # col_final.extend(row)
    # data_final.extend(data)
    
    # edge_adj_array = csr_matrix((data_final, (row_final, col_final)), shape = (len(edges_array), len(edges_array))).toarray()
    # edge_adj = csr_matrix((data_final, (row_final, col_final)), shape = (len(edges_array), len(edges_array)))
    
    # ebc=nx.edge_betweenness_centrality(graph)
    return adj


def distance_sum(g):
    lengths = dict(nx.floyd_warshall(g, weight='weight'))
    
    total_sum=0
    
    for i in range(len(g.nodes)):
        node_lengths=lengths[i]
        node_sum=sum(np.array(list(node_lengths.values())))
        # print(node_sum)
        total_sum+=node_sum
    return total_sum

def get_adjacency_feature_centrality(list_graph, num_nodes,pfilename="test.edg"):
    
    """Returns adjacency matrices, their node sequences and centrality matrix."""
    
    num_graph = len(list_graph)
    # list_adj = list()
    # list_adj_t = list()
    cent_mat = np.zeros((1,num_graph),dtype=float)
    list_nodelist = list()
    list_feature_mat = list()
    list_edge_index=list()
    list_edge_weight=list()
    list_distance_sum=list()

    for ind,g in enumerate(list_graph):
        print("Generating Graph: "+str(ind))
        print("Max_Node: "+str(num_nodes))
        node_seq = list(g.nodes())
        
        starttime2 = time.time()
        # ebc=nx.betweenness_centrality(g, weight = 'weight')
        total_sum_new=distance_sum(g)
        # eff=total_sum_orig/total_sum_new
        
        endtime2 = time.time()
        duration = endtime2-starttime2
        
        print("EFF:"+str(duration))
        cent_mat[0, ind] = total_sum_new
          
        # adj_mat = np.zeros((num_nodes,num_nodes))
        # adj_mat_excerpt = get_adj_matrix(g).todense()
        # adj_mat[0:adj_mat_excerpt.shape[0], 0:adj_mat_excerpt.shape[1]] = adj_mat_excerpt
        # adj_mat = sparse.csr_matrix(adj_mat)
        

        
        ndim = 256
        feature_mat = np.zeros((num_nodes,ndim))
        feature_mat_excerpt = get_node_embedding(g, dimensions = ndim, walk_length = 50, num_walks = 50, workers = 12,pfilename=pfilename).todense()
        feature_mat[0:feature_mat_excerpt.shape[0]] = feature_mat_excerpt
        feature_mat = sparse.csr_matrix(feature_mat)

        # adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)  
 
        feature_mat = sparse_mx_to_torch_sparse_tensor(feature_mat)
        
        # list_adj.append(adj_mat)
        list_edge_index.append(torch.tensor(list(g.edges)).t().contiguous())
        list_edge_weight.append(torch.tensor([g[u][v]['weight'] for u, v in g.edges]))

        
        list_nodelist.append(list(range(num_nodes)))
        list_feature_mat.append(feature_mat)
      
         
    # return list_adj,list_adj_t,list_nodelist,cent_mat,list_feature_mat
    return list_edge_index,list_edge_weight , cent_mat, list_feature_mat


def get_adjacency_centrality(list_graph, num_nodes):
    
    """Returns adjacency matrices, their node sequences and centrality matrix."""
    
    num_graph = len(list_graph)
    # list_adj = list()
    # list_adj_t = list()
    cent_mat = np.zeros((1,num_graph),dtype=float)
    list_nodelist = list()
    # list_feature_mat = list()
    list_edge_index=list()
    list_edge_weight=list()

    for ind,g in enumerate(list_graph):
        print("Generating Graph: "+str(ind))
        print("Max_Edge: "+str(num_nodes))
        node_seq = list(g.nodes())
        
        starttime2 = time.time()
        # ebc=nx.betweenness_centrality(g, weight = 'weight')
        total_sum_new=distance_sum(g)
        # eff=total_sum_orig/total_sum_new
        endtime2 = time.time()
        duration = endtime2-starttime2
        
        print("EFF:"+str(duration))
        cent_mat[0, ind] = total_sum_new
          
        # adj_mat = np.zeros((num_nodes,num_nodes))
        # adj_mat_excerpt = get_adj_matrix(g).todense()
        # adj_mat[0:adj_mat_excerpt.shape[0], 0:adj_mat_excerpt.shape[1]] = adj_mat_excerpt
        # adj_mat = sparse.csr_matrix(adj_mat)
        
        # adj_mat = sparse_mx_to_torch_sparse_tensor(adj_mat)  
        # list_adj.append(adj_mat)
        
        list_edge_index.append(torch.tensor(list(g.edges)).t().contiguous())
        list_edge_weight.append(torch.tensor([g[u][v]['weight'] for u, v in g.edges]))

        list_nodelist.append(list(range(num_nodes)))

    return list_edge_index,list_edge_weight, cent_mat#, list_feature_mat

def get_node_embedding(graph,dimensions=128,walk_length=30, num_walks=200, workers=4,pfilename="test.edg"):
    print("Edges: "+str(len(graph.edges())))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nodelist = list(graph.nodes)
    
    
    "already seen the q=5, lets try now q=2"
    
    ####Pecanpy
    nx.write_weighted_edgelist(graph, pfilename,delimiter='\t')
    
    # g=pecanpy.PreComp(p=1,q=5,workers=8,verbose=True)
    # g=pecanpy.SparseOTF(p=1,q=1,workers=8,verbose=True)
    
    g = pecanpy.SparseOTF(p=1, q=2, workers=8, verbose=True, extend=True, gamma=0)
    print("Extend=True")
    g.read_edg(pfilename, weighted=True ,directed=True)
    
    starttime2 = time.time()
    walks = g.simulate_walks(num_walks=10, walk_length=100)# use random walks to train embeddings
    model = Word2Vec(walks, vector_size=dimensions, window=5, min_count=0, sg=1, workers=8, epochs=10)
    endtime2 = time.time()
    duration = endtime2-starttime2
    
    print("Pecanpy:"+str(duration))
    
    
    ####Node2Vec
    # starttime2 = time.time()
    # node2vecc = Node2Vec(graph, dimensions=dimensions, walk_length=80, num_walks=10, weight_key='weight', workers=8).to(device)
    # model = node2vecc.fit(window=5, min_count=0, batch_words=4)
    # endtime2 = time.time()
    # duration = endtime2-starttime2
    # print("Node2Vec:"+str(duration))

    
    # edges_embs = AverageEmbedder(keyed_vectors=model.wv)
    feature_matrix = (model.wv.vectors)
    # feature_matrix=np.zeros((len(nodelist),dimensions))


    # for i,ei in enumerate(edgelist):
    #     sample_edge_embedding = edges_embs[(str(ei[0]),str(ei[1]))]
    #     feature_matrix[i,:]=sample_edge_embedding
    
    feature_matrix_sparse = csr_matrix(feature_matrix)
    
    return feature_matrix_sparse


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    
    sparse_mx = sparse_mx.tocoo().astype(float)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    # FOR UNWEIGHTED, data only contains ones!!!!
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)








""" FUNCTIONS REQUIRED FOR THE TRAINING"""



"weight based modifications on the adjacency matrix"

def adjacency_weight_modified(list_graph, list_adj):
    
    list_adj_temp = list_adj.copy()
    
    for i in range(len(list_graph)):
        print(i)
        
        graph = list_graph[i]
        
        eadj = np.array(list_adj[i].to_dense())
        rc = np.array([np.where(eadj==1)[0],np.where(eadj==1)[1]])
        
        edges = np.zeros((2,len(graph.edges)))
        edgeweights = np.zeros(len(graph.edges))
        
        counter=0
        for node1, node2, data in graph.edges(data = True):
            edges[0, counter] = node1
            edges[1, counter] = node2
            edgeweights[counter] = data['weight']
            counter+=1
        
        for j in range(rc.shape[1]):
            row = rc[0, j]
            col = rc[1, j]
            eweight = (edgeweights[row] + edgeweights[col])/2
            eadj[row, col] = 1/eweight
            eadj[col, row] = 1/eweight
            
        eadj = sparse.csr_matrix(eadj)
        eadj = sparse_mx_to_torch_sparse_tensor(eadj)
        list_adj_temp[i] = eadj
        
    return list_adj_temp




"node degree based modification of the adjacency matrix"




# list_graph=list_realsample_graph
# list_adj= list_adj_realsample
def adjacency_degree_modified(list_graph, list_adj):
    
    list_adj_temp = list_adj.copy()
    
    for i in range(len(list_graph)):
        print(i)
        
        graph = list_graph[i]
        eadj = np.array(list_adj[i].to_dense())
        rc = np.array([np.where(eadj==1)[0], np.where(eadj==1)[1]])
        edges = np.array(graph.edges)
        degrees = np.array(graph.degree())[:,1]
        degrees=degrees.astype(int)
        
        for j in range(rc.shape[1]):
            row = rc[0, j]
            col = rc[1, j]      
            node = np.intersect1d(edges[row, :], edges[col, :])
            assert len(node) == 1, "Error"
            node = node[0]
            degree = degrees[node]
            eadj[row, col] = 1/degree
            eadj[col, row] = 1/degree
        # eadj=eadj/np.max(eadj)
        eadj = sparse.csr_matrix(eadj)
        eadj = sparse_mx_to_torch_sparse_tensor(eadj)
        list_adj_temp[i] = eadj
        
    return list_adj_temp




"Shuffling starts"


def shuffling_everything(max_model_no, n_graph_nos, list_adj, list_adj_t, bc_mat, list_fm):
    
    list_adj_temp = list_adj.copy()
    list_adj_t_temp = list_adj_t.copy()
    bc_mat_temp = bc_mat.copy()
    list_fm_temp = list_fm.copy()
    
    max_model_size = max_model_no
    list_n_shuffle = [list(range(0, max_model_size))]*n_graph_nos
    list_n_unshuffle = [list(range(0, max_model_size))]*n_graph_nos

    for i in range(n_graph_nos):
        print(i)
        # i=0
        # graph = list_graph[i]
        
        nodeseq = list(np.arange(0, max_model_size))
        random.shuffle(nodeseq)
        
        ebc = bc_mat[:, i]
        ebc_new = ebc[nodeseq]
        bc_mat_temp[:, i] = ebc_new
        
        eadj = np.array(list_adj[i].to_dense())
        rows, cols = np.nonzero(eadj)       
        eadjn = np.zeros_like(eadj) 
        eadjn = eadj[nodeseq, :]
        eadjn = eadjn[:, nodeseq]          
        eadjn = sparse.csr_matrix(eadjn)
        eadjn = sparse_mx_to_torch_sparse_tensor(eadjn)
        list_adj_temp[i] = eadjn
        
        eadj = np.array(list_adj_t[i].to_dense())
        rows, cols = np.nonzero(eadj)    
        eadjn = np.zeros_like(eadj)    
        eadjn = eadj[nodeseq,:]
        eadjn = eadjn[:,nodeseq]   
        eadjn = sparse.csr_matrix(eadjn)
        eadjn = sparse_mx_to_torch_sparse_tensor(eadjn)
        list_adj_t_temp[i] = eadjn
        
        fm = np.array(list_fm[i].to_dense())        
        fmn = fm[nodeseq, :]
        fmn = sparse.csr_matrix(fmn)
        fmn = sparse_mx_to_torch_sparse_tensor(fmn)
        list_fm_temp[i] = fmn

        list_n_shuffle[i] = nodeseq
              
        # Create an inverse of the shuffled index array (to reverse the shuffling operation, or to "unshuffle")
        unshuf_order = np.zeros_like(nodeseq)
        unshuf_order[nodeseq] = np.arange(max_model_size)
        list_n_unshuffle[i] = unshuf_order
        
    return list_adj_temp, list_adj_t_temp, bc_mat_temp, list_fm_temp, list_n_shuffle, list_n_unshuffle



"this loss function yet to understand properly"


def new_loss_cal(y_out,true_val,num_nodes,device,model_size):

    y_out = y_out.reshape((num_nodes))
    true_val = true_val.reshape((num_nodes))
    #print(num_nodes)
    _,order_y_true = torch.sort(-true_val[:num_nodes])

    sample_num = num_nodes*20

    ind_1 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)
    ind_2 = torch.randint(0,num_nodes,(sample_num,)).long().to(device)
    

    rank_measure=torch.sign(-1*(ind_1-ind_2)).float()
        
    input_arr1 = y_out[:num_nodes][order_y_true[ind_1]].to(device)
    input_arr2 = y_out[:num_nodes][order_y_true[ind_2]].to(device)
        

    loss_rank = torch.nn.MarginRankingLoss(margin=1.0).forward(input_arr1,input_arr2,rank_measure)
 
    return loss_rank




def top_accuracy(y_out, true_val):
    model_size = y_out.shape[0]
    node_num = model_size
    y_out = y_out.reshape((model_size))
    true_val = true_val.reshape((model_size))

    predict_arr = y_out.cpu().detach().numpy()
    true_arr = true_val.cpu().detach().numpy()
    
    
    # 

    #check top-k accuracy
    k = 25
    top_k_predict = predict_arr.argsort()[::-1][:k]
    # print(predict_arr[top_k_predict])
    top_k_true = true_arr.argsort()[::-1][:k]
    # print(true_arr[top_k_true])
    acc = 0
    for item in top_k_predict:
        if item in top_k_true:
            acc += 1
    
    pred_ones=np.zeros_like(predict_arr)        
    pred_ones[top_k_predict]=1
    predict_arr_filter=np.multiply(predict_arr,pred_ones)
    
    true_ones=np.zeros_like(true_arr)        
    true_ones[top_k_true]=1
    true_arr_filter=np.multiply(true_arr,true_ones)
    
    kt_k,_ = kendalltau(predict_arr_filter,true_arr_filter)
    sp_k,_ = scipy.stats.spearmanr(predict_arr_filter,true_arr_filter)
    prs_k,_ = scipy.stats.pearsonr(predict_arr_filter,true_arr_filter)
    
    kt_all,_ = kendalltau(predict_arr,true_arr)
    sp_all,_ = scipy.stats.spearmanr(predict_arr,true_arr)
    prs_all,_ = scipy.stats.pearsonr(predict_arr,true_arr)
    
    result = [(acc/k)*100, kt_k, sp_k, prs_k, kt_all, sp_all, prs_all]
    
    return result,k

            
    # return (acc/k)*100,kt














