from entropy.CountMotif_nr import countMotifs
from entropy.Entropy import graphEntropy
from entropy.edge_entropy import edgeEntropy
from entropy.countedge import countEdge
import entropy.io as io
import numpy as np


def writeEdgeEntropy(graphfile):
    if graphfile.endswith(".xlsx"):
        graphfile=io.translata_xlsx_to_csv(graphfile)
        print('转变格式成功')
    A, nodN = io.read_adjMatrix_csv(graphfile)
    #return edgeEntropy(graphEntropy(countMotifs(A,nodN),nodN),countEdge(A,nodN))
    return countEdge(A,nodN)

print(writeEdgeEntropy('./data/graph10.xlsx'))

def writeEdgeAttribute(graph_ids,adj):
    edge_entropys=[]
    # build graphs with nodes
    edge_index=0
    node_index_begin=0
    for g_id in set(graph_ids):
        print('正在处理图：'+str(g_id))
        node_ids = np.argwhere(graph_ids == g_id).squeeze()
        node_ids.sort()

        temp_nodN=len(node_ids)
        temp_A=np.zeros([temp_nodN,temp_nodN],int)

        edge_index_begin=edge_index
        print('node_ids'+str(node_ids))



        while (edge_index<len(adj))and(adj[edge_index][0]-1 in node_ids):
            temp_A[adj[edge_index][0]-1-node_index_begin][adj[edge_index][1]-1-node_index_begin]=1
            edge_index+=1

        print('temp_A\n'+str(temp_A))

        entropy_matrix = edgeEntropy(graphEntropy(countMotifs(temp_A, temp_nodN),temp_nodN),countEdge(temp_A, temp_nodN))

        print('entropy_matrix\n '+str(entropy_matrix))

        print(str(edge_index_begin)+'  加入属性的起止边：'+str(edge_index-1))
        for j in range(edge_index_begin,edge_index):
            edge_entropys.append(entropy_matrix[adj[j][0]-1-node_index_begin][adj[j][1]-1-node_index_begin])

        node_index_begin+=temp_nodN
    print('edge_entropys长度为:'+str(len(edge_entropys)))
    return edge_entropys