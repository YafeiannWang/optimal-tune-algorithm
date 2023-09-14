import pdb
import random
import os
import numpy as np
from igraph import *
from scipy.special import expit as sigmoid 


def generate_graph(N,edges):
    G = Graph(directed=True)
    G.add_vertices(N)
    while edges>0:
        a = np.random.randint(0,N)
        b = np.random.randint(0,N)
        G.add_edges([(a,b)])
        edges = edges-1
        if not G.is_dag() or not G.is_simple():
            G.delete_edges([(a,b)])
            edges = edges+1
    return G

#def simulate_parameter(B, w_ranges=((-2.0, -0.5), (0.5, 2.0))):
#    """Simulate SEM parameters for a DAG.
#
#    Args:
#        B (np.ndarray): [d, d] binary adj matrix of DAG
#        w_ranges (tuple): disjoint weight ranges
#
#    Returns:
#        W (np.ndarray): [d, d] weighted adj matrix of DAG
#    """
#    W = np.zeros(B.shape)
#    S = np.random.randint(len(w_ranges), size=B.shape)  # which range
#    for i, (low, high) in enumerate(w_ranges):
#        U = np.random.uniform(low=low, high=high, size=B.shape)
#        W += B * (S == i) * U
#    return W

#Tanh generator
def generate_nonlinear_Gau(S,N,edge,top_sort,relation_matrix,i):
    samples = []
    for t in range(S):
        x_vector = np.zeros(N)
        for j in range(relation_matrix.shape[0]):
            x_2 = np.tanh(x_vector)#+np.cos(x_vector)+np.sin(x_vector)
            pdb.set_trace()
            x_vector[top_sort[j]] = np.dot(relation_matrix[:,top_sort[j]],x_2.T)+np.random.normal(0,1)
        samples.append(x_vector)
    #save_path = './'+str(i)+'/generate_nonlinear_Gau_node'+str(N)+'_edge'+str(edge)+'.csv'
    #np.savetxt(save_path,samples,delimiter=',')
    

#Abs generator
def generate_abs_Gau(S,N,edge,top_sort,relation_matrix,i):
    #generate linear data with Guassian
        samples = []
        for t in range(S):
            x_vector = np.zeros(N)
            for j in range(relation_matrix.shape[0]):
                random_num = np.random.randint(0,2,size = N)
                b_para = random_num*np.random.uniform(-2,-0.5,size = N)+(1-random_num)*np.random.uniform(0.5,2,size = N)
                x_2 = b_para*np.abs(x_vector)
                x_vector[top_sort[j]]=np.dot(relation_matrix[:,top_sort[j]],x_2.T)+np.random.normal(0,1)
            samples.append(x_vector)
        save_path = './'+str(i)+'/generate_abs_Gau_node'+str(N)+'_edge'+str(edge)+'.csv'
        np.savetxt(save_path,samples,delimiter=',')

#Abs-Tanh mix generator
def generate_mixtrue_squad(S,N,edge,top_sort,relation_matrix,i):
        samples = []
        for t in range(S):
            x_vector = np.zeros(N)
            for j in range(relation_matrix.shape[0]):
                a_para = np.random.normal(0,1)
                if j%2 == 0:
                    x_2 = np.abs(x_vector)
                    x_vector[top_sort[j]]=np.dot(relation_matrix[:,top_sort[j]],x_2.T)+np.random.normal(0,1)
                else:
                    x_2 = np.tanh(x_vector)#+np.cos(x_vector)+np.sin(x_vector)
                    x_vector[top_sort[j]] = np.dot(relation_matrix[:,top_sort[j]],x_2.T)+np.random.exponential(1)
            samples.append(x_vector)
        save_path = './'+str(i)+'/generate_mixture_abstanh_node'+str(N)+'_edge'+str(edge)+'.csv'
        np.savetxt(save_path,samples,delimiter=',')

#MLP-Tanh mix generator
def generate_mixture_linear(S,N,edge,top_sort,relation_matrix,i):
    samples = []
    for t in range(S):
        x_vector = np.zeros(N)
        for j in range(relation_matrix.shape[0]):
            if j%7 == 0:
                x_2 = np.tanh(x_vector)+np.cos(x_vector)+np.sin(x_vector)
                x_vector[top_sort[j]] = np.dot(relation_matrix[:,top_sort[j]],x_2.T)+np.random.exponential(1)
            else:
                x_vector[top_sort[j]] = np.dot(relation_matrix[:,top_sort[j]], x_vector.T)+np.random.normal(0,1)
        samples.append(x_vector)
    save_path = './'+str(i)+'/generate_mixture_linear_node'+str(N)+'_edge'+str(edge)+'.csv'
    np.savetxt(save_path,samples,delimiter=',')

if __name__ == "__main__":

    N = [10] #define the number of features
    edges = [40] #define the number of edges on dag
    s_number = [400]
    for i in range(10):
        os.mkdir('./'+str(i))
        for node_number,edge_number,S in zip(N,edges,s_number):
            G = generate_graph(node_number,edge_number)

            top_sort = G.topological_sorting()
            
            relation_matrix = np.array(G.get_adjacency().data)        
            save_path = './'+str(i)+'/relation_matrix_node'+str(node_number)+'_edge'+str(edge_number)+'.csv'
            np.savetxt(save_path,relation_matrix,delimiter=',')

            generate_abs_Gau(S,node_number,edge_number,top_sort,relation_matrix,i)
