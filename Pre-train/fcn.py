 # %%
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.conv import HGTConv
from dgllife.model.readout.weighted_sum_and_max import WeightedSumAndMax
class DTI(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_feat_size, layer_num=3, 
                 num_heads=2, num_ntypes=2, num_etypes=4, dropout=0.05, use_norm=True):
        super(DTI, self).__init__()

        # 初始化 HGTConv 层
        self.convs = nn.ModuleList()
        for _ in range(layer_num):
            conv = HGTConv(in_size=node_feat_size, 
                           head_size=hidden_feat_size, 
                           num_heads=num_heads, 
                           num_ntypes=num_ntypes, 
                           num_etypes=num_etypes, 
                           dropout=dropout, 
                           use_norm=use_norm)
            self.convs.append(conv)
        self.fc = FC(hidden_feat_size*2, hidden_feat_size*2, 3, 0.05, 1)
        self.fc_pre = FC(hidden_feat_size*2, hidden_feat_size*2, 2, 0.05, 128)

    def forward(self, bg_0,bg_1):
        
        with bg_0.local_scope():  # 创建一个局部作用域，‌确保对图的操作不会影响原始图。‌
            for conv in self.convs:
                node_feats = conv(bg_0,bg_0.ndata['h'],bg_0.ndata['_TYPE'], bg_0.edata['_TYPE'])
        bg_0.ndata['h']=node_feats
        graph_feats_0 = dgl.readout_nodes(bg_0,'h')
        logits = self.fc(graph_feats_0)

        
        with bg_1.local_scope():  # 创建一个局部作用域，‌确保对图的操作不会影响原始图。‌
            for conv in self.convs:
                node_feats = conv(bg_1,bg_1.ndata['h'],bg_1.ndata['_TYPE'], bg_1.edata['_TYPE'])
        bg_1.ndata['h']=node_feats
        graph_feats_1 = dgl.readout_nodes(bg_1,'h')
        
        output_0 = self.fc_pre(graph_feats_0)
        output_1 = self.fc_pre(graph_feats_1)
        
        return F.normalize(output_0, dim=1),F.normalize(output_1, dim=1),graph_feats_0 ,graph_feats_1,logits 
    
    
class FC(nn.Module):
    def __init__(self, d_graph_layer, d_FC_layer, n_FC_layer, dropout, n_tasks):
        super(FC, self).__init__()
        self.d_graph_layer = d_graph_layer
        self.d_FC_layer = d_FC_layer
        self.n_FC_layer = n_FC_layer
        self.dropout = dropout
        self.predict = nn.ModuleList()
        for j in range(self.n_FC_layer):
            if j == 0:
                self.predict.append(nn.Linear(self.d_graph_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))
            if j == self.n_FC_layer - 1:
                self.predict.append(nn.Linear(self.d_FC_layer, n_tasks))
            else:
                self.predict.append(nn.Linear(self.d_FC_layer, self.d_FC_layer))
                self.predict.append(nn.Dropout(self.dropout))
                self.predict.append(nn.LeakyReLU())
                self.predict.append(nn.BatchNorm1d(d_FC_layer))

    def forward(self, h):
        for layer in self.predict:
            h = layer(h)

        return h
