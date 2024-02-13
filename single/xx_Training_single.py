
import numpy as np
import pickle

import torch

import random
import torch.nn as nn
# from model_bet import Graph_Diff_Reg
import torch.optim as optim
import glob

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time

from sklearn.metrics import r2_score

from xx_AllFunctions import adjacency_weight_modified, adjacency_degree_modified, shuffling_everything, new_loss_cal, top_accuracy
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import networkx as nx




# with open('graph_data_100_to_100_nodes_package_01.pickle', 'rb') as handle:
#     list_train_graph, list_test_graph, list_edge_index_train,list_edge_weight_train, eff_mat_train, list_fm_train_orig, list_edge_index_test,list_edge_weight_test, eff_mat_test, list_fm_test_orig,list_edge_index_train_orig,list_edge_weight_train_orig,list_edge_index_test_orig,list_edge_weight_test_orig = pickle.load(handle)




def edge_list_to_adjacency_matrix(edge_list, edge_weights, num_nodes):
    # Initialize an empty adjacency matrix with zeros
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Iterate through the edge list and fill in the adjacency matrix
    for edge, weight in zip(edge_list, edge_weights):
        source, target = edge
        adj_matrix[source][target] = weight

    return adj_matrix


    

def train(filefolder):
    model.train() 
    
    loss_train = 0
    num_files_train = len(filefolder)
    
    starttime3 = time.time()
    y_trues=[]
    y_preds=[]
    
    random.shuffle(filefolder)
    n=0
    
    for i in range(num_files_train):
        print("Batch: "+str(i))

        filepath=filefolder[i]
        with open(filepath,"rb") as f:
            # list_adj_deg_train_shuffled, list_adj_wt_train_shuffled,list_num_edge_train, bc_mat_train_shuffled, list_fm_train_shuffled, list_n_seq_train, list_n_unshuffle_train,list_adj_deg_test_shuffled, list_adj_wt_test_shuffled,list_num_edge_test, bc_mat_test_shuffled, list_fm_test_shuffled, list_n_seq_test, list_n_unshuffle_test=pickle.load(f)
            list_train_graph, list_edge_index_train,list_edge_weight_train, eff_mat_train, list_fm_train_orig,list_edge_index_train_orig,list_edge_weight_train_orig=pickle.load(f)
        
        # for j in range(len(list_train_graph)):
        #     print(len(list_train_graph[j].edges()))
        
        
        for j in range(len(list_train_graph)):
            
            # g=list_train_graph
            # adj2=torch.tensor(nx.adjacency_matrix(list_train_graph[j]).toarray()).numpy()
            
            # for 
            
            
            edge_index=list_edge_index_train[j]
            edge_weigth=list_edge_weight_train[j]
            
            adj=torch.tensor(edge_list_to_adjacency_matrix(edge_index.numpy().transpose(), edge_weigth.numpy(), 500))
            adj=adj.to(device)
            
            edge_index=edge_index.to(device)
            edge_weigth=edge_weigth.to(device)
            fm=list_fm_train_orig[0]
            # fm=torch.tensor(np.eye(500,dtype=np.float32))
            
            batch_vector=np.zeros((fm.shape[0]),dtype=np.int64)
            batch_vector=torch.tensor(batch_vector)
            batch_vector=batch_vector.to(device)
            
            fm=fm.to(device).to_dense()
            # adj_orig=torch.tensor(nx.adjacency_matrix(list_train_graph_orig[j]).toarray())
            edge_index_orig=list_edge_index_train_orig[0]
            edge_weigth_orig=list_edge_weight_train_orig[0]
            
            adj_orig=torch.tensor(edge_list_to_adjacency_matrix(edge_index_orig.numpy().transpose(), edge_weigth_orig.numpy(), 500))
            adj_orig=adj_orig.to(device)
            
            adjn=adj.cpu().numpy()
            adjon=adj_orig.cpu().numpy()
            adjr=adjon-adjn
            adjr[adjr!=0]=1
            adjr=torch.tensor(adjr.astype(np.float32)).to(device)
            
            # edge_index_n=edge_index.cpu().numpy()
            # edge_index_orig_n=edge_index_orig.cpu().numpy()
            edge_weight_n=edge_weigth.cpu().numpy()
            edge_weight_n[edge_weight_n<10000]=1
            edge_weight_n[edge_weight_n!=1]=0
            edge_mask=torch.tensor(edge_weight_n.astype(np.int64)).to(device)
            
            edge_index_orig=edge_index_orig.to(device)
            
            edge_weigth_orig=edge_weigth_orig.to(device)
            
            y_true=torch.tensor(eff_mat_train[0,j].reshape([1,1]).astype(np.float32))
            y_true=y_true.to(device)
            
            # adj = list_adj_deg_train_shuffled[j]
            # num_nodes = list_num_edge_train[j]
            # adj_t = list_adj_wt_train_shuffled[j]
            # fm = list_fm_train_shuffled[j]
            # adj = adj.to(device)
            # adj_t = adj_t.to(device)
            # fm=fm.to(device)
            
            optimizer.zero_grad()
                
            # y_out = model(adj,adj_t,fm)
            y_out = model(edge_index,edge_weigth.float(),edge_index_orig,edge_weigth_orig.float(),fm.float(),batch_vector,adjr.float(),edge_mask)#,adj,adj_orig)
                        
            # true_arr = torch.from_numpy(bc_mat_train_shuffled[:,j]).float()
            # true_val = true_arr.to(device)
            
            # true_val_final = true_val[list_n_unshuffle_train[j]]
            # y_out_final = y_out[list_n_unshuffle_train[j]]
            
            # true_val_ultimate = true_val_final[:num_nodes]
            # y_out_ultimate = y_out_final[:num_nodes]
            
            mse = loss_fn(y_out, y_true)
            
            mse.backward()
            
            
            
            optimizer.step()
            
            y_trues.append(float(y_true))
            y_preds.append(float(y_out))
            
            
            # loss_rank = new_loss_cal(y_out_ultimate, true_val_ultimate, num_nodes, device, model_size)
            loss_train = loss_train + float(mse)
            n+=1
            
    mse_final=loss_train/n
    duration3 = time.time()-starttime3
    print("training time this epoch:"+str(duration3))
    print("training loss:"+str(loss_train))
    print("training rmse:"+str(np.sqrt(float(mse_final))))
    
    score=r2_score(y_trues, y_preds)
    print("Training_R_Squared:"+str(score))
    
    return loss_train,mse_final,score






def test_no_save(test_list):
    
    model.eval()
    y_trues=[]
    y_preds=[]
    
    all_result = []
    
    loss_train=0
    n=0
    
    filepath=test_list[0]
    
    with open(filepath,"rb") as f:
        # list_adj_deg_train_shuffled, list_adj_wt_train_shuffled,list_num_edge_train, bc_mat_train_shuffled, list_fm_train_shuffled, list_n_seq_train, list_n_unshuffle_train,list_adj_deg_test_shuffled, list_adj_wt_test_shuffled,list_num_edge_test, bc_mat_test_shuffled, list_fm_test_shuffled, list_n_seq_test, list_n_unshuffle_test=pickle.load(f)
        list_train_graph, list_edge_index_train,list_edge_weight_train, eff_mat_train, list_fm_train_orig,list_edge_index_train_orig,list_edge_weight_train_orig=pickle.load(f)
        
    f.close()
    num_samples_test = len(list_train_graph)
    for j in range(num_samples_test):
      
        
        edge_index=list_edge_index_train[j]
        edge_weigth=list_edge_weight_train[j]
        adj=torch.tensor(edge_list_to_adjacency_matrix(edge_index.numpy().transpose(), edge_weigth.numpy(), 500))
        adj=adj.to(device)
        
        edge_index=edge_index.to(device)
        
        edge_weigth=edge_weigth.to(device)
        fm=list_fm_train_orig[0]
        # fm=torch.tensor(np.eye(500,dtype=np.float32))
       
        batch_vector=np.zeros((fm.shape[0]),dtype=np.int64)
        batch_vector=torch.tensor(batch_vector)
        batch_vector=batch_vector.to(device)
       
        fm=fm.to(device).to_dense()
        edge_index_orig=list_edge_index_train_orig[0]
        edge_weigth_orig=list_edge_weight_train_orig[0]
        adj_orig=torch.tensor(edge_list_to_adjacency_matrix(edge_index_orig.numpy().transpose(), edge_weigth_orig.numpy(), 500))
        adj_orig=adj_orig.to(device)
        
        
        adjn=adj.cpu().numpy()
        adjon=adj_orig.cpu().numpy()
        adjr=adjon-adjn
        adjr[adjr!=0]=1
        adjr=torch.tensor(adjr.astype(np.float32)).to(device)
        
        edge_weight_n=edge_weigth.cpu().numpy()
        edge_weight_n[edge_weight_n<10000]=1
        edge_weight_n[edge_weight_n!=1]=0
        edge_mask=torch.tensor(edge_weight_n.astype(np.int64)).to(device)
        
        
        edge_index_orig=edge_index_orig.to(device)
        
        edge_weigth_orig=edge_weigth_orig.to(device)
        
        y_true=torch.tensor(eff_mat_train[0,j].reshape([1,1]).astype(np.float32))
        y_true=y_true.to(device)
       
        
        y_out = model(edge_index,
                      edge_weigth.float(),
                      edge_index_orig,
                      edge_weigth_orig.float(),
                      fm.float(),
                      batch_vector,
                      adjr.float(),
                      edge_mask)#,adj,adj_orig)
         
        y_trues.append(float(y_true))
        y_preds.append(float(y_out))
        
        mse = loss_fn(y_out, y_true)
        
        
        loss_train = loss_train + float(mse)
        n+=1
        
    mse_final=loss_train/n
        
    torch.cuda.empty_cache()
  
    
    
    mse_final=loss_train/n
    
    
    
    print("testing loss:"+str(loss_train))
    print("testing rmse:"+str(np.sqrt(float(mse_final))))
    
    score=r2_score(y_trues, y_preds)
    print("Testing_R_Squared:"+str(score))
    
    return loss_train,mse_final,score







#Model parameters  (for examples)
# hidden = 64
# num_epoch = 20

hidden = 256
num_epoch = 200


"dropout, hidden parameter number, and learning rate is subjected to hyperparameter tuning"


n_features= 256#list_fm_train_shuffled[0].shape[1]
dropout_val = 0.4
learning_rate = 0.00005
# learning_rate = 0.0001
# model_size = 100# bc_mat_train_shuffled.shape[0]


# device = torch.device("cpu")
# model = GNN_Bet(ninput = model_size, nhid = hidden, n_features = n_features,dropout = dropout_val)

from model_bet import Graph_Diff_Reg
# from torchsummary import summary

model =Graph_Diff_Reg(input_dim=n_features,hidden_dim=hidden,output_dim=1,dropout=dropout_val)



print('The following device is used: '+str(device))
model.to(device)

loss_fn = nn.MSELoss()  # mean square error
# optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=learning_rate/10)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.00005/10)

training_loss_epochs = []
training_mse_epochs = []

testing_loss_epochs= []
testing_mse_epochs= []

training_score_epochs = []
testing_score_epochs = []

# all_result_test_mean_epochs = []
# all_result_test_std_epochs = []
# all_result_train_mean_epochs = []
# all_result_train_std_epochs = []



filefolder=glob.glob(r"D:\xx_clean_code_vul_fin\single\graph_data_train_single\*")
test_list=glob.glob(r"D:\xx_clean_code_vul_fin\single\graph_data_test_single\*")

print("Training")
print(f"Total Number of epoches: {num_epoch}")
for e in range(num_epoch):
    print(f"Epoch number: {e+1}/{num_epoch}")
    print("LR: "+str(learning_rate))
  
    
    train_loss,train_mse,train_score = train(filefolder)
    training_loss_epochs.append(train_loss)
    training_mse_epochs.append(train_mse)
    training_score_epochs.append(train_score)

    #to check test loss while training
    with torch.no_grad():
        # print("test data score")
        # [all_result_test_mean, all_result_test_std, all_result,youts,truevals,meandegs,stddegs] = test(test_list)
        # with open("erdos_10000_results.pickle", 'wb') as handle:
        #     pickle.dump([all_result_test_mean, all_result_test_std,all_result,youts,truevals,meandegs,stddegs], handle, protocol=pickle.HIGHEST_PROTOCOL)

        
        print("test data score")
        test_loss,test_mse,test_score = test_no_save(test_list)
        testing_loss_epochs.append(test_loss)
        testing_mse_epochs.append(test_mse)
        testing_score_epochs.append(test_score)


# from scipy.io import savemat


# mdic = {"Mean_mod_test": all_result_test_mean_epochs ,"label": "experiment"}


# savemat("Mean_mod_test.mat", mdic)







# torch.save(model,r"C:\Users\srila\Desktop\xx_clean_code_v2\GNN_gnmtest_new_model_trained.pt")






# import math

# n = 1500
# k = [0,1,2,3,4,5,6,7,8,9,10]
# num_ways=0

# for ki in k:
#     num_ways += math.comb(n, ki)
# print("Number of ways to delete 10 edges from a graph with 1500 edges:", num_ways)







###############################################################################################

# ###Loading and testing model



# # -*- coding: utf-8 -*-
# """
# Created on Fri Jul  1 17:37:57 2022

# @author: debasishjana
# """


# import numpy as np
# import pickle
# # import networkx as nx
# import torch
# # from utils import *
# import random
# # import torch.nn as nn
# from model_bet import GNN_Bet
# # torch.manual_seed(20)
# # import argparse
# # import pickle
# # from scipy import sparse
# # import scipy
# # import utils2
# # from itertools import combinations, groupby
# # import pandas as pd
# # import math
# # import time
# import glob

# import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
# import networkx as nx


# # import torch
# # import torch.nn as nn
# import time
# # from scipy.stats import kendalltau
# # import random


# from xx_AllFunctions import adjacency_weight_modified, adjacency_degree_modified, shuffling_everything, new_loss_cal, top_accuracy
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")






# def test_save(test_matrizes,test_grpahs):
    
#     model.eval()
#     # loss_val = 0
#     # list_kt = list()
#     # list_ktf=list()
#     # num_samples_test = len(test_list)
#     # accus=[]
    
#     all_result = []
    
#     # youts=[]
#     # truevals=[]
#     meandegs=[]
#     stddegs=[]
#     properties=[]
    
#     # filepath=test_list[0]
    
#     # with open(filepath, 'rb') as f:
#     #     list_adj_deg_train_shuffled, list_adj_wt_train_shuffled,list_num_edge_train, bc_mat_train_shuffled, list_fm_train_shuffled, list_n_seq_train, list_n_unshuffle_train,list_adj_deg_test_shuffled, list_adj_wt_test_shuffled,list_num_edge_test, bc_mat_test_shuffled, list_fm_test_shuffled, list_n_seq_test, list_n_unshuffle_test=pickle.load(f)
#     # f.close()
    
    
 
#     with open(test_grpahs, 'rb') as f:
#        list_test_graph, _, _, _, _, _, _, _ = pickle.load(f)
#     f.close()
    

    
    
#     with open(test_matrizes, 'rb') as f:
#         list_adj_deg_train_shuffled, list_adj_wt_train_shuffled,list_num_edge_train, bc_mat_train_shuffled, list_fm_train_shuffled, list_n_seq_train, list_n_unshuffle_train,list_adj_deg_test_shuffled, list_adj_wt_test_shuffled,list_num_edge_test, bc_mat_test_shuffled, list_fm_test_shuffled, list_n_seq_test, list_n_unshuffle_test=pickle.load(f)
#     f.close()
    
    
    
    
#     num_samples_test = 500#len(list_adj_deg_train_shuffled)
#     for j in range(num_samples_test):
#         print(j)
#         # print(j)
#         # print(i)
     
       
#         G=list_test_graph[j]
        
#         Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
#         G = G.subgraph(Gcc[0])
        
#         avg_deg=np.mean(np.array(list(G.degree()))[:,1])
#         std_deg=np.std(np.array(list(G.degree()))[:,1])
        
#         meandegs.append(avg_deg)
#         stddegs.append(std_deg)
        
#         props=graph_properties(G)
#         properties.append(props)
        
#         adj = list_adj_deg_train_shuffled[j]
#         adj_t =list_adj_wt_train_shuffled[j]
#         adj = adj.to(device)
#         adj_t = adj_t.to(device)
#         num_nodes = list_num_edge_train[j]
#         fm =list_fm_train_shuffled[j]
#         fm = fm.to(device)
#         y_out = model(adj,adj_t,fm)
    
        
#         true_arr = torch.from_numpy(bc_mat_train_shuffled[:,j]).float()
#         true_val = true_arr.to(device)
        
        
#         true_val_final = true_val[list_n_unshuffle_train[j]]
#         y_out_final = y_out[list_n_unshuffle_train[j]]
        
#         true_val_ultimate = true_val_final[:num_nodes]
#         y_out_ultimate = y_out_final[:num_nodes]
        
#         # youts.append(y_out_ultimate)
#         # truevals.append(true_val_ultimate)
    
#         # kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
#         # list_kt.append(kt)
        
#         #g_tmp = list_graph_test[j]
#         #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")
        
        
#         # accu,ktf = top_accuracy(y_out_ultimate, true_val_ultimate)
#         # accus.append(accu)
#         # list_ktf.append(ktf)
        
#         [one_result, kk] = top_accuracy(y_out_ultimate, true_val_ultimate)
#         all_result.append(one_result)
        
#         del adj,adj_t,fm,y_out,true_val
        
#         torch.cuda.empty_cache()
#         # print(one_result)
        
        
        
#     # print(f"   Average KT score on test graphs is: {np.mean(np.array(list_ktf))} and std: {np.std(np.array(list_ktf))}")
#     # print(np.mean(accus))
#     # print(np.mean(list_ktf))
#     # return list_ktf, true_val_ultimate, y_out_ultimate
    
    
#     mean_all_result = np.mean(np.array(all_result), axis=0)
#     std_all_result = np.std(np.array(all_result), axis=0)
    
#     print("top k perc, k is:"+str(kk))
#     print("accu_k, kt_k, sp_k, prs_k, kt_all, sp_all, prs_all")
#     print(mean_all_result)
    
#     return mean_all_result, std_all_result,all_result,meandegs,stddegs,properties





# def graph_properties(G1):
#     """returns the clustering coefficient and the average shortest path length"""
    
#     node_nos = len(G1.nodes())
#     edge_nos = len(G1.edges())
    
#     cluster_array = nx.clustering(G1)
#     clustering_coeff = np.sum(np.array(list(cluster_array.values())))/node_nos
    
#     avg_shortest_path = nx.average_shortest_path_length(G1, weight='weight')
    
#     graph_props = [node_nos, edge_nos, clustering_coeff, avg_shortest_path]
    
#     return graph_props





# model = torch.load(r"C:\Users\srila\Desktop\xx_clean_code_v2\GNN_sm_new_model_trained.pt")

# test_matrizes=r"C:/Users/srila/Desktop/xx_clean_code_v2/graphs/sm_new_mod/sm_new_batch02.pickle"
# test_grpahs=r"C:/Users/srila/Desktop/xx_clean_code_v2/sm_new_graph_data_1000_to_5000_nodes_package_02.pickle"

# mean_all_result, std_all_result,all_result,meandegs,stddegs,properties=test_save(test_matrizes,test_grpahs)


# with open(r"C:/Users/srila/Desktop/xx_clean_code_v2/results_sythetic/final_train_result_sm_new_02.pickle", 'wb') as handle:
#     pickle.dump([mean_all_result, std_all_result,all_result,meandegs,stddegs,properties], handle, protocol=pickle.HIGHEST_PROTOCOL)













# with open(r"C:\Users\srila\Desktop\xx_clean_code_v2\results_sythetic\final_train_result_erdos.pickle","rb") as f:
#     mean_all_result, std_all_result,all_result,meandegs,stddegs,properties=pickle.load(f)



# ######## Testing on cities

# model = torch.load(r"C:/Users/srila/Desktop/xx_clean_code_v2/GNN_erdos_model_trained.pt")
# model.eval()

# # [all_result_test_mean, all_result_test_std, all_result,youts,truevals,meandegs,stddegs] = test(test_list)
# # with open("erdos_10000_results.pickle", 'wb') as handle:
# #     pickle.dump([all_result_test_mean, all_result_test_std,all_result,youts,truevals,meandegs,stddegs], handle, protocol=pickle.HIGHEST_PROTOCOL)





# citytestlist=glob.glob(r"D:\city_matrizes_reduced\*")



# def test_on_city(test_list):
    
#     model.eval()
#     # loss_val = 0
#     # list_kt = list()
#     # list_ktf=list()
#     num_samples_test = len(test_list)
#     # accus=[]
    
#     all_result = []
#     filenames=[]
    
#     # youts=[]
#     # truevals=[]
#     # meandegs=[]
#     # stddegs=[]  
    
#     for j in range(num_samples_test):
        
#         print(j)
#         # print(i)
#         filepath=test_list[j]
#         with open(filepath, 'rb') as handle:
#             list_adj_deg_train_shuffled, list_adj_wt_train_shuffled, list_num_edge_train, bc_mat_train_shuffled, list_fm_train_shuffled, list_n_unshuffle_train = pickle.load(handle)
#         filenames.append(filepath)    
#         # degs=1/np.array(list_adj_deg_train_shuffled[0].to_dense())
#         # degs[degs==np.inf]=0
#         # degs=degs.astype(int)
#         # degs=degs.reshape([-1])
#         # degs=degs[degs!=0]
        
#         # meandegs.append(np.mean(degs))
#         # stddegs.append(np.std(degs))
        
#         adj = list_adj_deg_train_shuffled[0]
#         adj_t =list_adj_wt_train_shuffled[0]
#         adj = adj.to(device)
#         adj_t = adj_t.to(device)
#         num_nodes = list_num_edge_train[0]
#         fm =list_fm_train_shuffled[0]
#         fm = fm.to(device)
#         y_out = model(adj,adj_t,fm)
    
        
#         true_arr = torch.from_numpy(bc_mat_train_shuffled).float()
#         true_val = true_arr.to(device)
        
        
#         true_val_final = true_val[list_n_unshuffle_train[0]]
#         y_out_final = y_out[list_n_unshuffle_train[0]]
        
#         true_val_ultimate = true_val_final[:num_nodes]
#         y_out_ultimate = y_out_final[:num_nodes]
        
#         # youts.append(y_out_ultimate)
#         # truevals.append(true_val_ultimate)
    
#         # kt = ranking_correlation(y_out,true_val,num_nodes,model_size)
#         # list_kt.append(kt)
        
#         #g_tmp = list_graph_test[j]
#         #print(f"Graph stats:{g_tmp.number_of_nodes()}/{g_tmp.number_of_edges()},  KT:{kt}")
        
        
#         # accu,ktf = top_accuracy(y_out_ultimate, true_val_ultimate)
#         # accus.append(accu)
#         # list_ktf.append(ktf)
        
#         [one_result, kk] = top_accuracy(y_out_ultimate, true_val_ultimate)
#         all_result.append(one_result)
        
#         # print(one_result)
        
        
        
#     # print(f"   Average KT score on test graphs is: {np.mean(np.array(list_ktf))} and std: {np.std(np.array(list_ktf))}")
#     # print(np.mean(accus))
#     # print(np.mean(list_ktf))
#     # return list_ktf, true_val_ultimate, y_out_ultimate
    
    
#     mean_all_result = np.mean(np.array(all_result), axis=0)
#     std_all_result = np.std(np.array(all_result), axis=0)
    
#     print("top k perc, k is:"+str(kk))
#     print("accu_k, kt_k, sp_k, prs_k, kt_all, sp_all, prs_all")
#     print(mean_all_result)
    
#     return mean_all_result, std_all_result,all_result,filenames


# [mean_all_result, std_all_result,all_result,filenames] = test_on_city(citytestlist)





# all_result_array2=np.array(all_result)


# Z = [x for _,x in sorted(zip(list(all_result_array2[:,4]),citytestlist))]
# print(Z)





# # i=0
# # adj = list_adj_deg_train_shuffled[i]
# # num_nodes = list_num_edge_train[i]
# # adj_t = list_adj_wt_train_shuffled[i]


# # fm = list_fm_train_shuffled[i]
# # adj = adj.to(device)
# # adj_t = adj_t.to(device)
# # fm=fm.to(device)

# # optimizer.zero_grad()
    
# # y_out = model(adj,adj_t,fm)
# # true_arr = torch.from_numpy(bc_mat_train_shuffled[:,i]).float()
# # true_val = true_arr.to(device)

# # true_val_final = true_val[list_n_unshuffle_train[i]]
# # y_out_final = y_out[list_n_unshuffle_train[i]]

# # true_val_ultimate = true_val_final[:num_nodes]
# # y_out_ultimate = y_out_final[:num_nodes]










