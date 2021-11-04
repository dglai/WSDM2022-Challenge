#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Xuhong Wang
# @Email  : wxuhong@amazon.com or wang_xuhong@sjtu.edu.cn
# Feel free to send me an email if you have any question. 
# You can also CC Quan Gan (quagan@amazon.com).
import torch.nn.functional as F
import torch
import dgl
import dgl.nn.pytorch as dglnn
from torch import nn
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('Base')
    parser.add_argument('--dataset', type=str, choices=["A", "B"], default='A', help='Dataset name')
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument("--emb_dim", type=int, default=10, help="number of hidden gnn units")
    parser.add_argument("--n_layers", type=int, default=2, help="number of hidden gnn layers")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight for L2 loss")

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    return args

class HeteroConv(nn.Module):
    def __init__(self, etypes, n_layers, in_feats, hid_feats, activation, dropout=0.2):
        super(HeteroConv, self).__init__()
        self.etypes = etypes
        self.n_layers = n_layers
        self.in_feats = in_feats
        self.hid_feats = hid_feats
        self.act = activation
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.hconv_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(n_layers+1):
            self.norms.append(nn.BatchNorm1d(hid_feats)) 

        # input layer
        self.hconv_layers.append(self.build_hconv(in_feats,hid_feats,activation=self.act))
        # hidden layers
        for i in range(n_layers - 1):
            self.hconv_layers.append(self.build_hconv(hid_feats,hid_feats,activation=self.act))   
        # output layer
        self.hconv_layers.append(self.build_hconv(hid_feats,hid_feats)) # activation None        

        self.fc1 = nn.Linear(hid_feats*2+10, hid_feats)
        self.fc2 = nn.Linear(hid_feats, 1)
    
    def build_hconv(self,in_feats,out_feats,activation=None):
        GNN_dict = {}
        for event_type in self.etypes:
            GNN_dict[event_type] = dglnn.SAGEConv(in_feats=in_feats,out_feats=out_feats,aggregator_type='mean',activation=activation) 
        return dglnn.HeteroGraphConv(GNN_dict, aggregate='sum')

    def forward(self, g, feat_key='feat'):
        h = g.ndata[feat_key]
        if not isinstance(h,dict):
            h = {'Node':g.ndata[feat_key]}
        for i, layer in enumerate(self.hconv_layers):
            h = layer(g, h)
            for key in h.keys():
                h[key] = self.norms[i](h[key])
        return h

    def emb_concat(self, g, etype):
        def cat(edges):
            return {'emb_cat': torch.cat([edges.src['emb'], edges.dst['emb']],1)}
        with g.local_scope():
            g.apply_edges(cat, etype=etype)
            emb_cat = g.edges[etype].data['emb_cat']
        return emb_cat

    def time_encoding(self, x, bits=10): 
        '''
        This function is designed to encode a unix timestamp to a 10-dim vector. 
        And it is only one of the many options to encode timestamps.
        Users can also define other time encoding methods such as Neural Network based ones.
        '''
        inp = x.repeat(10,1).transpose(0,1)
        div = torch.cat([torch.ones((x.shape[0],1))*10**(bits-1-i) for i in range(bits)],1)
        return (((inp/div).int()%10)*0.1).float()
        
    def time_predict(self, node_emb_cat, time_emb):
        h = torch.cat([node_emb_cat, time_emb], 1)
        h = self.dropout(h)
        h = self.fc1(h)
        h = self.act(h)
        h = self.fc2(h)
        return h
    
def train(args, g):
    if args.dataset == 'B':
        dim_nfeat = args.emb_dim*2 
        for ntype in g.ntypes:
            g.nodes[ntype].data['feat'] = torch.randn((g.number_of_nodes(ntype), dim_nfeat)) 
    else:
        dim_nfeat = g.ndata['feat'].shape[1]

    model = HeteroConv(g.etypes, args.n_layers, dim_nfeat, args.emb_dim, F.relu)

    optimizer = torch.optim.AdamW(model.parameters(),
                                 lr=args.lr,
                                 weight_decay=args.weight_decay)
    loss_fcn = nn.BCEWithLogitsLoss()  
    loss_values = []

    for i in range(args.epochs):
        model.train()
        node_emb = model(g)
        loss = 0
        for ntype in g.ntypes:
            g.nodes[ntype].data['emb'] = node_emb[ntype]
        for i, etype in enumerate(g.etypes):
            if etype.split('_')[-1] == 'reversed':
                # Etype that end with 'reversed' is the reverse edges we added for GNN message passing.
                # So we do not need to compute loss in training.
                continue
            emb_cat = model.emb_concat(g, etype)
            ts = g.edges[etype].data['ts']
            idx = torch.randperm(ts.shape[0])
            ts_shuffle = ts[idx]
            neg_label = torch.zeros_like(ts)
            neg_label[ts_shuffle>=ts] = 1

            time_emb = model.time_encoding(ts) 
            time_emb_shuffle = model.time_encoding(ts_shuffle) 

            pos_exist_prob = model.time_predict(emb_cat, time_emb).squeeze()
            neg_exist_prob = model.time_predict(emb_cat, time_emb_shuffle).squeeze()

            probs = torch.cat([pos_exist_prob,neg_exist_prob],0)
            label = torch.cat([torch.ones_like(ts),neg_label],0)
            loss += loss_fcn(probs, label)/len(g.etypes)
                
        loss_values.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print('Loss:', loss_values[-1])
        torch.cuda.empty_cache()
        # test every epoch
        test(args, g, model)
    return g, model
    
    
def preprocess(args, directed_g):
    # this function is used to add reverse edges for model computing
    if args.dataset == 'A':
        g = dgl.add_reverse_edges(directed_g, copy_edata=True)
    if args.dataset == 'B':
        graph_dict = {}
        for (src_type, event_type, dst_type) in directed_g.canonical_etypes:
            graph_dict[(src_type, event_type, dst_type)] = directed_g.edges(etype = (src_type, event_type, dst_type))
            src_nodes_reversed = directed_g.edges(etype = (src_type, event_type, dst_type))[1]
            dst_nodes_reversed = directed_g.edges(etype = (src_type, event_type, dst_type))[0]
            graph_dict[(dst_type, event_type+'_reversed', src_type)] = (src_nodes_reversed,dst_nodes_reversed)
        g = dgl.heterograph(graph_dict)
        for etype in g.etypes:
            g.edges[etype].data['ts'] = directed_g.edges[etype.split('_')[0]].data['ts']
            if 'feat' in directed_g.edges[etype.split('_')[0]].data.keys():
                g.edges[etype].data['feat'] = directed_g.edges[etype.split('_')[0]].data['feat']
    return g

@torch.no_grad()
def test(args, g, model):
    model.eval()
    data_path = 'toy_testset/'
    test_csv = pd.read_csv(data_path + f'Dataset_{args.dataset}_test_toy.csv')
    label = test_csv.exist.values
    src = test_csv.src.values
    dst = test_csv.dst.values
    start_at = torch.tensor(test_csv.start_at.values)
    end_at = torch.tensor(test_csv.end_at.values)
    if args.dataset == 'A':
        emb_cats =  torch.cat([g.ndata['emb'][src],g.ndata['emb'][dst]], 1)
    if args.dataset == 'B':
        emb_cats =  torch.cat([g.ndata['emb']['User'][src],g.ndata['emb']['Item'][dst]], 1)

    start_time_emb = model.time_encoding(start_at)
    end_time_emb = model.time_encoding(end_at)
    start_prob = model.time_predict(emb_cats, start_time_emb).squeeze()
    end_prob = model.time_predict(emb_cats, end_time_emb).squeeze()
    exist_prob = end_prob - start_prob
    
    AUC = roc_auc_score(label,exist_prob)
    print(f'AUC is {round(AUC,5)}')

if __name__ == "__main__":
    args = get_args()
    if args.dataset == 'B':
        g= dgl.load_graphs(f'DGLgraphs/Dataset_{args.dataset}.bin')[0][0]
    elif args.dataset == 'A':
        g, etype_feat = dgl.load_graphs(f'DGLgraphs/Dataset_{args.dataset}.bin')
        g = g[0]
    g = preprocess(args, g)
    g, model = train(args,g)
    test(args, g, model)