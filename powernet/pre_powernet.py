import torch
from dgl import DGLGraph
import numpy as np
from core.data.constants import GRAPH, LABELS, N_CLASSES, N_RELS, N_ENTITIES
from core.models.constants import GNN_NODE_LABELS_KEY, GNN_NODE_ATTS_KEY, GNN_EDGE_FEAT_KEY
from core.models.constants import GNN_EDGE_LABELS_KEY, GNN_EDGE_NORM
import pandas as pd
import os

#dn ->node_id


# def make_node_dict(matrix):
#     f = open('./node_dict.txt', 'a+')
#     for i in range(len(matrix)):
#         node_dict[matrix[i][0]]=i+1
#         f.write(str(matrix[i][0])+','+str(i+1)+'\n')
#     return

def load_node_dir(node_dict_file='F:\\PyCharm 2018.3.5\\PyCharmProjects\\edGNN\\edGNN\\powernet\\nodedict.txt'):
    lines=open(node_dict_file).readlines()
    node_dict = dict()
    for line in lines:
        line=line.strip('\r\n').split(',')
        node_dict[line[0]]=int(line[1])

    return node_dict


def read_edges(edge_file):
    node_dict=load_node_dir()
    pre_edges = []
    edges=open(edge_file).readlines()
    for edge in edges:
        edge=edge.strip('\r\n').split(',')
        pre_edges.append([node_dict[e] for e in edge ])
    return pre_edges


# def load_labels():
#     for root, dirs, files in os.walk('F:\\PyCharm 2018.3.5\\PyCharmProjects\\edGNN\\edGNN\\powernet\\alarm'):
#         for result_time in files:
#             lines=open('F:\\PyCharm 2018.3.5\\PyCharmProjects\\edGNN\\edGNN\\powernet\\alarm\\'+result_time).readlines()
#             #F:\\PyCharm 2018.3.5\\PyCharmProjects\\edGNN\\edGNN\\powernet\\alarm\\'+result_time+'.csv
#             labels=[]
#             counter=0
#             for line in lines:
#                 temp=int(line)
#                 if temp>0 :
#                     temp=1
#                     counter+=1
#                 labels.append(temp)
#             print(result_time+' :'+str(counter))
#     #return labels

def load_labels(result_time):
    lines = open('F:\\PyCharm 2018.3.5\\PyCharmProjects\\edGNN\\edGNN\\powernet\\alarm\\' + result_time+'.csv').readlines()
    labels = []
    for line in lines:
        temp = int(line)
        if temp > 0:
            temp = 1
        labels.append(temp)
    return labels



def make_graph(result_time):
    g=DGLGraph()

#node
    node_atts = open('F:\\PyCharm 2018.3.5\\PyCharmProjects\\edGNN\edGNN\\powernet\\features\\'+result_time+'.csv').readlines()
    node_atts = np.delete(node_atts, [0], axis=0)

    matrix = []

    for line in node_atts:
        line = line.strip('\r\n').split(',')
        matrix.append(line)

    matrix=np.delete(matrix,[0],axis=1)
    matrix_folat=[]
    for line in matrix:
        line=[float(x) for x in line]
        matrix_folat.append(line)

    matrix_folat=np.array(matrix_folat)
    g.add_nodes(len(matrix_folat))

    g.ndata[GNN_NODE_ATTS_KEY] = torch.from_numpy(matrix_folat)



    #edges
    for edge in read_edges('F:\\PyCharm 2018.3.5\\PyCharmProjects\\edGNN\\edGNN\\powernet\\edges.txt'):
        g.add_edge(edge[0], edge[1])

    g.edata[GNN_EDGE_LABELS_KEY]=torch.ones((g.number_of_edges(),))
    edge_src, edge_dst = g.edges()
    edge_dst = list(edge_dst.data.numpy())
    edge_type = list(g.edata[GNN_EDGE_LABELS_KEY])
    _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True,
                                        return_counts=True)
    degrees = count[inverse_index]
    edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)
    g.edata[GNN_EDGE_NORM] = torch.FloatTensor(edge_norm)

    g.edata[GNN_EDGE_FEAT_KEY] = torch.ones((g.number_of_edges(),))
    return g




def make_mask(n):

    random_idx1=np.random.randint(0,n,(1,n))
    random_idx=random_idx1[0]
    val_msak=np.zeros((n,),dtype=np.int)
    test_mask=np.zeros((n,),dtype=np.int)
    train_mask=np.zeros((n,),dtype=np.int)

    val_msak[random_idx[:n//5]]=1
    test_mask[random_idx[n//5:(n//5)*2]]=1
    train_mask[random_idx[(n//5)*2:n]]=1

    #return train_mask,test_mask,val_msak
    return random_idx[(n//5)*2:n],random_idx[n//5:(n//5)*2],random_idx[:n//5]

