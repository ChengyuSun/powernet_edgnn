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

# def deletefile():
#     file_list = ['1578674010', '1579187700', '1578785580', '1578990510', '1579209420', '1579190310',
#                  '1579405110', '1578655140', '1578644010', '1578648510']
#     for root, dirs, files in os.walk("./alarm", topdown=False):
#         for name in files:
#             time=name.strip('.csv')
#             if time not in file_list:
#                 os.remove(os.path.join(root,name))
#
#     for root, dirs, files in os.walk("./features", topdown=False):
#         for name in files:
#             time=name.strip('.csv')
#             if time not in file_list:
#                 os.remove(os.path.join(root,name))

def load_node_dir(node_dict_file='./nodedict.txt'):
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
#             lines=open('./alarm/'+result_time+'.csv
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

def load_labels():
    file_list = ['1578674010', '1579187700', '1578785580', '1578990510', '1579209420', '1579190310',
                 '1579405110', '1578655140', '1578644010','1578648510']
    res=[]
    for result_time in file_list:
        lines = open('./alarm/' + result_time+'.csv').readlines()
        labels = []
        for line in lines:
            temp = int(line)
            if temp > 0:
                temp = 1
            labels.append(temp)
        res.extend(labels)
    return res
    # r=[]
    # for i in range(674):
    #     if i%2 ==0:
    #         r.append(1)
    #     else: r.append(0)
    # return r

def read_attr(result_time):
    node_atts = open(
        './features/' + result_time + '.csv').readlines()
    node_atts = np.delete(node_atts, [0], axis=0)

    matrix = []

    for line in node_atts:
        line = line.strip('\r\n').split(',')
        matrix.append(line)

    matrix = np.delete(matrix, [0], axis=1)
    matrix_folat = []
    for line in matrix:
        line = [float(x) for x in line]
        matrix_folat.append(line)

    return matrix_folat


def make_graph():

    g=DGLGraph()

    file_list=['1578674010', '1579187700', '1578785580', '1578990510', '1579209420', '1579190310',
               '1579405110', '1578655140', '1578644010','1578648510']
#node
    matrix_folat=[]
    for file in file_list:
        matrix_folat.extend(read_attr(file))

    g.add_nodes(len(matrix_folat))
    print('len(matrix_folat)'+str(len(matrix_folat)))
    matrix_folat=np.array(matrix_folat)
    g.ndata[GNN_NODE_ATTS_KEY] = torch.from_numpy(matrix_folat)

    #edges
    for i in range(10):
        for edge in read_edges('./edges.txt'):
            g.add_edge(edge[0]+i*10, edge[1]+i*10)
    # g.edata[GNN_EDGE_LABELS_KEY]=torch.ones((g.number_of_edges(),))
    # edge_src, edge_dst = g.edges()
    # edge_dst = list(edge_dst.data.numpy())
    # edge_type = list(g.edata[GNN_EDGE_LABELS_KEY])
    # _, inverse_index, count = np.unique((edge_dst, edge_type), axis=1, return_inverse=True,
    #                                     return_counts=True)
    # degrees = count[inverse_index]
    # edge_norm = np.ones(len(edge_dst), dtype=np.float32) / degrees.astype(np.float32)
    # temp = torch.FloatTensor(edge_norm)
    # g.edata[GNN_EDGE_NORM]=temp.view(g.number_of_edges(),1)
    # g.edata[GNN_EDGE_FEAT_KEY] = torch.ones((g.number_of_edges(),))
    return g

def make_mask(n):

    random_idx1=np.random.randint(0,n,(1,n))
    random_idx=random_idx1[0]
    # val_msak=np.zeros((n,),dtype=np.int)
    # test_mask=np.zeros((n,),dtype=np.int)
    # train_mask=np.zeros((n,),dtype=np.int)

    # val_msak[random_idx[:n//5]]=1
    # test_mask[random_idx[n//5:(n//5)*2]]=1
    # train_mask[random_idx[(n//5)*2:n]]=1

    #return train_mask,test_mask,val_msak
    return random_idx[(n//5)*2:n],random_idx[n//5:(n//5)*2],random_idx[:n//5]
