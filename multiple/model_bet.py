# import torch.nn as nn
# import torch.nn.functional as F
# from layer import GNN_Layer
# from layer import GNN_Layer_Init
# from layer import MLP
# import torch 


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool,SAGPooling#,DiffPool

class Graph_Diff_Reg(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim,dropout):
        super(Graph_Diff_Reg, self).__init__()
        self.dropout = dropout

        self.gcn1 = GCNConv(input_dim, hidden_dim)
        # self.gcn12 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        # self.pool= SAGPooling(hidden_dim)
        # self.gcn22 = GCNConv(hidden_dim, hidden_dim)
        # self.gcn3 = GCNConv(input_dim, hidden_dim)
        # self.gcn4 = GCNConv(input_dim, hidden_dim)
        
        # self.diff_pool = DiffPool(hidden_dim, hidden_dim)
        # self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(int(hidden_dim), hidden_dim),
            # nn.ReLU(),
            # nn.Tanh(),
            # nn.Sigmoid(),
            nn.Dropout(p=self.dropout),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.Dropout(p=self.dropout),
            nn.Linear(int(hidden_dim/2), int(hidden_dim/4)),
            nn.Dropout(p=self.dropout),
            # # nn.Linear(int(hidden_dim/4), int(hidden_dim/8)),
            # # nn.ReLU(),
            # # nn.Tanh(),
            # nn.Dropout(p=self.dropout),
            nn.Linear(int(hidden_dim/4),1)
        )


    def forward(self,edge_index1,edge_weight1,edge_index2,edge_weight2,fm0, fm1,batch_tensor):#,adjr,edge_mask):#,adj,adj_orig):
        # x, edge_index, batch = data.x, data.edge_index, data.batch

        # First GCN layer
        x11 = self.gcn1(fm0, edge_index1,edge_weight1)
        # print(sum())
        # x11 = F.relu(x11)

        # First GCN layer
        # with torch.no_grad():
        x12 = self.gcn1(fm1, edge_index2,edge_weight2)
        # x12 = F.relu(x12)
        
        
        ###Secong GCN layer
        x21 = self.gcn2(x11, edge_index1,edge_weight1)
        # x21 = F.relu(x21)

        # First GCN layer
        # with torch.no_grad():
        x22 = self.gcn2(x12 , edge_index2,edge_weight2)
        # x22 = F.relu(x22)
        
        
       #### Third GCN layer
        # x31 = self.gcn3(x21, edge_index1,edge_weight1)
        # x31 = F.relu(x31)

        # # Third GCN layer
        # x32 = self.gcn3(x22 , edge_index2,edge_weight2)
        # x32 = F.relu(x32)
        
        # x41 = self.gcn4(x21, edge_index1,edge_weight1)
        # x41 = F.relu(x31)
        
        #  # Third GCN layer
        # x42 = self.gcn4(x22 , edge_index2,edge_weight2)
        # x42 = F.relu(x32)
        
        
        # x1=x11+x21+x31
        # x2=x12+x22+x32
        
        x1=x12-x11
        x2=x22-x21
        # x3=x32-x31
        # x4=x42-x41
        
        x=x1*x2#*x3#*x4
        
        
        # s = self.diff_pool(x, edge_index1, edge_weight1)

        # # Updated node representations after DiffPool
        # x = s.x
        
        
        
        
        # x=x1-x2
        
        # print(x)
        # print(x.shape)
        # Graph pooling
        
        # from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
        # x = torch.cat([gmp(x, batch_tensor), 
        #             gap(x, batch_tensor)], dim=1)
        
        x = global_mean_pool(x,batch_tensor)
        
        # MLP layers
        x = self.mlp(x)

        return x











# class GNN_Bet(nn.Module):
#     def __init__(self, ninput, nhid, n_features,dropout):
#         super(GNN_Bet, self).__init__()
        
        
#         self.gc1 = GNN_Layer_Init(ninput,n_features,nhid)
#         self.gc2 = GNN_Layer(nhid,nhid)
#         self.gc3 = GNN_Layer(nhid,nhid)
#         self.gc4 = GNN_Layer(nhid,nhid)
#         self.gc5 = GNN_Layer(nhid,nhid)


#         self.dropout = dropout

#         self.score_layer1 = MLP(nhid,self.dropout)
#         self.score_layer2 = MLP(nhid,self.dropout)
#         self.score_layer3 = MLP(nhid,self.dropout)
#         self.score_layer4 = MLP(nhid,self.dropout)
#         self.score_layer5 = MLP(nhid,self.dropout)






#     def forward(self,adj1,adj2,fm):
        
        
        
#         #Layers for aggregation operation
#         x_1 = F.normalize(F.leaky_relu(self.gc1(adj1,fm)),p=2,dim=1)
#         x2_1 = F.normalize(F.leaky_relu(self.gc1(adj2,fm)),p=2,dim=1)
        
#         xf_1=x_1-x2_1


#         x_2 = F.normalize(F.leaky_relu(self.gc2(x_1, adj1)),p=2,dim=1)
#         x2_2 = F.normalize(F.leaky_relu(self.gc2(x2_1, adj2)),p=2,dim=1)
        
#         xf_2=x_2-x2_2


#         x_3 = F.normalize(F.leaky_relu(self.gc3(x_2,adj1)),p=2,dim=1)
#         x2_3 = F.normalize(F.leaky_relu(self.gc3(x2_2,adj2)),p=2,dim=1)
        
#         xf_3=x_3-x2_3
        
#         x_4 = F.normalize(F.leaky_relu(self.gc4(x_3,adj1)),p=2, dim=1)
#         x2_4 = F.normalize(F.leaky_relu(self.gc4(x2_3,adj2)),p=2,dim=1)
        
#         xf_4=x_4-x2_4

#         x_5 = F.leaky_relu(self.gc5(x_4,adj1))
#         x2_5 = F.leaky_relu(self.gc5(x2_4,adj2))
        
#         xf_5=x_5-x2_5
        
        

        
        
#         score1_1 = self.score_layer1(x_1,self.dropout)
#         score1_2 = self.score_layer2(x_2,self.dropout)
#         score1_3 = self.score_layer3(x_3,self.dropout)
#         score1_4 = self.score_layer4(x_4,self.dropout)
#         score1_5 = self.score_layer5(x_5,self.dropout)


#         score2_1 = self.score_layer1(x2_1,self.dropout)
#         score2_2 = self.score_layer2(x2_2,self.dropout)
#         score2_3 = self.score_layer3(x2_3,self.dropout)
#         score2_4 = self.score_layer4(x2_4,self.dropout)
#         score2_5 = self.score_layer5(x2_5,self.dropout)
        
        
        
        

      
#         score1 = score1_1 + score1_2 + score1_3 + score1_4 + score1_5
#         score2 = score2_1 + score2_2 + score2_3 + score2_4 + score2_5

#         x = torch.mul(score1,score2)

#         return x
