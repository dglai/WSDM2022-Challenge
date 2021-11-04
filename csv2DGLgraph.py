import pandas as pd
import os
import sys
import argparse
import torch
import dgl
import numpy as np

def csv2graph(args):
    if args.dataset == 'A':
        src_type = 'Node'
        dst_type = 'Node'
    elif args.dataset == 'B':    
        src_type = 'User'
        dst_type = 'Item'
    else:
        print(' Input parameters error')

    edge_csv = pd.read_csv(f'train_csvs/edges_train_{args.dataset}.csv',header=None)
    
    heterogenous_group = edge_csv.groupby(2)
    graph_dict = {}
    ts_dict = {}

    for event_type, records in heterogenous_group:
        event_type = str(event_type)
        graph_dict[(src_type, event_type, dst_type)] = (records[0].to_numpy(),records[1].to_numpy())
        ts_dict[(src_type, event_type, dst_type)] = (torch.FloatTensor(records[3].to_numpy()))
    g = dgl.heterograph(graph_dict)
    
    g.edata['ts'] = ts_dict


    if args.dataset == 'A':
        # Assign Node feature in to graph
        node_feat_csv = pd.read_csv('train_csvs/node_features.csv',header=None)
        node_feat = node_feat_csv.values[:,1:]
        node_idx = node_feat_csv.values[:,0]
        g.nodes[src_type].data['feat'] = torch.zeros((g.number_of_nodes(src_type), 8))
        g.nodes[src_type].data['feat'][node_idx] = torch.FloatTensor(node_feat)
        
        # Assign Edge Type Feature as the graph`s label, which can be saved along with dgl.heterograph
        etype_feat_csv = pd.read_csv('train_csvs/edge_type_features.csv',header=None)
        etype_feat_tensor = torch.FloatTensor(etype_feat_csv.values[:,1:])
        etype_feat = {}
        for i,etype in enumerate(g.etypes):
            etype_feat[etype] = etype_feat_tensor[i]

        dgl.save_graphs(f"./DGLgraphs/Dataset_{args.dataset}.bin", g, etype_feat)

    if args.dataset == 'B':
        etype_feat = None
        # Assign Edge Feature
        for event_type, records in heterogenous_group:
            event_type = str(event_type)
            etype = (src_type, event_type, dst_type)
            if len(str(records[4].iloc[0])) > 3:
                g.edges[etype].data['feat'] = (extract_edge_feature(records[4]))
        dgl.save_graphs(f"./DGLgraphs/Dataset_{args.dataset}.bin", g)   

    

def extract_edge_feature(records):

    # if you face with OOM issue, using looping method istead of mapping method

    # parallelism mapping method
    feat = torch.FloatTensor(np.array(list(map(lambda x: x.strip().split(','), records))).astype('float'))

    # Recurrent looping method

    # feat_l = []
    # for record in records:
    #     feat_l.append(record.strip().split(','))
    # feat = torch.FloatTensor(np.array(feat_l).astype('float32'))  
      
    return feat

def get_args():
    ### Argument and global variables
    parser = argparse.ArgumentParser('csv2DGLgraph')
    parser.add_argument('--dataset', type=str, choices=["A", "B"], default = 'A', help='Dataset name')
    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    
    return args

if __name__ == "__main__":
    args = get_args()
    if not os.path.exists('train_csvs/'):
        os.system('wget -P train_csvs https://data.dgl.ai/dataset/WSDMCup2022/edges_train_A.csv.gz')
        os.system('wget -P train_csvs https://data.dgl.ai/dataset/WSDMCup2022/node_features.csv.gz')
        os.system('wget -P train_csvs https://data.dgl.ai/dataset/WSDMCup2022/edge_type_features.csv.gz')
        os.system('wget -P train_csvs https://data.dgl.ai/dataset/WSDMCup2022/edges_train_B.csv.gz')
        os.system('gzip -d train_csvs/*.gz')
    csv2graph(args)
    
