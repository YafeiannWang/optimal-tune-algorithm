import numpy as np
import pdb
import time
import csv
from igraph import *
import pandas as pd
import tensorflow.compat.v1 as tf
from cvxopt.solvers import qp
from cvxopt import matrix
import networkx as nx
#from cdt.metrics import precision_recall,SHD

#function for calculating HSIC between variables X and Y
def hsic(x,y):
    x_y = np.dot(x,y)
    n = x_y.shape[0]
    h = np.trace(x_y)/n**2+np.mean(x)*np.mean(y)-2*np.mean(x_y)/n
    return h*n**2/(n-1)**2

#function for calculating second-order HSIC of samples
def hsic_second_order_fun(samples,sample_type,folder):
    feature_size = samples.shape[1]
    result = [[int(feature_size*(feature_size-1)/2)]]
    second_order_samples = np.zeros((samples.shape[0]+2,int(feature_size*(feature_size-1)/2)))
    index = 0
    for i in range(feature_size):
        for j in range(i+1,feature_size):
            second_order_samples[0:2,index] = [i,j]
            second_order_samples[2:,index] = samples[:,i] + samples[:,j]
            index = index+1
    for i in range(feature_size):
        for j in range(second_order_samples.shape[1]):
            if not second_order_samples[0:2,j].__contains__(float(i)):
                x = samples[:,i]
                y = second_order_samples[2:,j]
                #Gaussian Kernel
                #Kx=np.expand_dims(x,0)-np.expand_dims(x,1)
                #Kx=np.exp(-Kx**2)
                #Ky=np.expand_dims(y,0)-np.expand_dims(y,1)
                #Ky=np.exp(-Ky**2)
                
                #Tanh Kernel
                Kx=np.expand_dims(x,0)*np.expand_dims(x,1)
                Kx=np.tanh(Kx)
                Ky=np.expand_dims(y,0)*np.expand_dims(y,1)
                Ky=np.tanh(Ky)
                xy_hsic = hsic(Kx,Ky)
                result.append([i]+second_order_samples[0:2,j].tolist()+[xy_hsic])
    file_name = folder+'hsic_second_order_'+'_'.join(sample_type.split('_')[1:])

    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in result:
            writer.writerow(row)
    return result

#function for calculation first-order HSIC of samples
def hsic_first_order_fun(samples,sample_type,folder):
    feature_size = samples.shape[1]
    result = [feature_size]
    for i in range(feature_size):
        for j in range(i+1,feature_size):
            x = samples[:,i]
            y = samples[:,j]
            #Gaussian kernel
            #Kx=np.expand_dims(x,0)-np.expand_dims(x,1)
            #Kx=np.exp(-Kx**2)
            #Ky=np.expand_dims(y,0)-np.expand_dims(y,1)
            #Ky=np.exp(-Ky**2)

            #Tanh Kernel
            Kx=np.expand_dims(x,0)*np.expand_dims(x,1)
            Kx=np.tanh(Kx)
            Ky=np.expand_dims(y,0)*np.expand_dims(y,1)
            Ky=np.tanh(Ky)
            xy_hsic = hsic(Kx,Ky)
            result.append(xy_hsic)
    file_name = folder+'hsic_first_order_'+'_'.join(sample_type.split('_')[1:])
    np.savetxt(file_name,result,delimiter=',')
    return result

#generating gumbel samples
def sample_gumbel(shape, eps=1e-20): 
    U = tf.random_uniform(shape, minval=0, maxval=1, dtype=tf.float64)
    return -tf.log(-tf.log(U + eps) + eps)


#generating gumbel_softmax sample
def gumbel_softmax_sample(logits, temperature): 
    y = logits/temperature + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax(y)


#define the gradient of gumbel_softmax
def gumbel_softmax(logits, temperature, hard=True):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keepdims=True)), y.dtype)
        y = tf.stop_gradient(y_hard-y) + y
    return y

#optimal phase of OT algorithm
def optimal_component(first_order_samples):
    sess = tf.InteractiveSession()
    tf.compat.v1.disable_eager_execution()

    Node = int(first_order_samples[0])
    first_order_samples = first_order_samples[1:].reshape(1,-1)
    N = int(Node*(Node-1)/2)#total edges to be selected
    K = Node #optimal selected edges
    tau = tf.placeholder(tf.float64, shape=[]) 

    #set variables for data flow in
    X = tf.placeholder(tf.float64, shape=[None, N])

    #set initial gumble distribution
    initial_gumbel = tf.Variable(tf.truncated_normal((N,N), dtype=tf.float64))

    sample_selection = gumbel_softmax(logits=initial_gumbel, temperature=tau, hard = True)
    sample_selection = tf.reduce_sum(sample_selection, axis=0)

    #calculate gradient parameters in sample selection
    w_raw = tf.Variable(tf.truncated_normal((N,), dtype = tf.float64))
    w_temp = tf.exp(w_raw)*sample_selection

    w = tf.expand_dims(w_temp/tf.reduce_sum(w_temp), axis=-1)

    #caculate loss function
    structure_component = tf.matmul(X, w)
    sparsity_component = tf.nn.l2_loss(w)
    loss = -structure_component + 0.01*sparsity_component

    #calculate optimal object
    opt = tf.train.AdamOptimizer(0.001)
    train = opt.minimize(loss)
    sess.run(tf.global_variables_initializer())

    for I in range(10001):
        #if I % 1000 == 0:
           #print(structure_component.eval(feed_dict={X: samples, tau: 0.1/np.log(np.e+I)}))
           #print(sparsity_component.eval(feed_dict={tau: 0.1/np.log(np.e+I)}))
        train.run(feed_dict={X: first_order_samples, tau: 0.1/np.log(np.e+I)})

    selected_ids = np.unique(np.argmax(initial_gumbel.eval(),axis=1))
    result = np.zeros(N)
    result[selected_ids]=1
    for i in range(Node):
        result = np.insert(result,i*Node,[0]*(i+1),axis=0)
    n = len(selected_ids)
    result = np.reshape(result,(Node,Node))
    return result

#tuning phase of OT algorithm
def correction_component(result,node_number,first_order_samples,second_order_samples):
    second_order_size = second_order_samples[0]
    second_order_samples = second_order_samples[1:]

    #reshape first order hsic into N*N matrix
    first_order_samples = first_order_samples[1:]
    for i in range(node_number):
        first_order_samples = np.insert(first_order_samples,i*node_number,[0]*(i+1),axis=0)
    first_order_samples = np.reshape(first_order_samples,(node_number,node_number))
    first_order_samples = first_order_samples + first_order_samples.T

    #delete edges violated the rule on second order hsic and mark them
    for triple in second_order_samples:
        edge1 = (int(float(triple[1])),int(triple[0]))
        edge2 = (int(float(triple[2])),int(triple[0]))
        max_edge,min_edge = [edge1,edge2] if first_order_samples[edge1] > first_order_samples[edge2] else [edge2,edge1]
        value = float(triple[3])

        if first_order_samples[min_edge]>value and result[max_edge]+result[min_edge] == 2:
            result[min_edge] = -1
            result[max_edge] = -1
        if first_order_samples[max_edge]>value and first_order_samples[min_edge]<value \
                and result[max_edge] == 1 \
                and first_order_samples[max_edge]!=max(first_order_samples[max_edge[1],:]):
                    #pdb.set_trace()
                    result[max_edge] = -1

    #add edge consist with the rule on second order hsic, including some edges deleted
    for triple in second_order_samples:
        edge1 = (int(triple[1]),triple[0])
        edge2 = (int(triple[2]),triple[0])
        max_edge,min_edge = [edge1,edge2] if first_order_samples[edge1] > first_order_samples[edge2] else [edge2,edge1]
        value = triple[3]

        if first_order_samples[max_edge]<value \
           and (result[edge1] + result[edge2] <= 0):
               result[edge1] = 1
               result[edge2] = 1

    result[result == -1] = 0

    #mark edges with di-oriention
    repeat_edges = []
    for i in range(result.shape[0]):
        for j in range(i+1,result.shape[1]):
            if result[i,j] == result[j,i] and result[i,j] == 1:
                repeat_edges.append((i,j))
    
    #select di-oriention edges with the rule based on the second order hsic
    edge_triples = np.array(second_order_samples)[:,:3].astype(float).astype(int)
    second_order_samples = np.array(second_order_samples)
    second_order_samples[:,:3] = edge_triples
    for repeat_edge in repeat_edges:
        edge_triples = second_order_samples[second_order_samples[:,0]==repeat_edge[0]]
        triple_result = edge_triples[edge_triples[:,1] == repeat_edge[1]]
        triple_result = np.r_[triple_result,edge_triples[edge_triples[:,2] == repeat_edge[1]]]
        edge_triples = second_order_samples[second_order_samples[:,0]==repeat_edge[1]]
        triple_result = np.r_[triple_result,edge_triples[edge_triples[:,1] == repeat_edge[0]]]
        triple_result = np.r_[triple_result,edge_triples[edge_triples[:,2] == repeat_edge[0]]]
        selected_edge = triple_result[np.where(triple_result == max(triple_result[:,3]))[0]][0]
        if selected_edge[0] == repeat_edge[0]:
            result[(int(repeat_edge[0]),int(repeat_edge[1]))] = 0
        else:
            result[(int(repeat_edge[1]),int(repeat_edge[0]))] = 0
    return result
 

if __name__ == '__main__':

    N = [10,20,25,30,40]#,50]#,60] #define the number of features
    edges = [40,100,150,250,400]#,1000]#,1700] #define the number of edges on dag
    path_type = ['generate_mixture_linear_node','generate_nonlinear_Gau_node','generate_linear_Gau_node']#'generate_mixture_squad_node',
    #path_type = ['generate_mlp_node','generate_exp_node']
    #selected_type = path_type[0]
    time_record = []
    auc_data,shd_data = [],[]
    for i in range(1,10):
        folder = './'+str(i)+'/'
        for selected_type in path_type:
            for node_number,edge_number in zip(N,edges):
                sample_path = selected_type+str(node_number)+'_edge'+str(edge_number)+'.csv'
                #sample_path = 'test_normalized_mixture_squar_data.csv'
                samples = np.loadtxt(folder+sample_path,dtype='float',delimiter=',') #need to add further
                nodes = samples.shape[1]
                hsic_first_order = np.array(hsic_first_order_fun(samples,sample_path,folder))

                time_start = time.time()
                hsic_second_order = hsic_second_order_fun(samples,sample_path,folder)
                result = optimal_component(hsic_first_order)
                
                result = correction_component(result,node_number,hsic_first_order,hsic_second_order)
                time_end = time.time()
                matrix_save_path = folder+'_'.join(selected_type.split('_')[1:3])+'_predict_matrix_OT_node'+str(node_number)+'_edge'+str(edge_number)+'.csv'
                np.savetxt(matrix_save_path,result,delimiter=',')
                matrix_path = folder+'relation_matrix_node'+str(node_number)+'_edge'+str(edge_number)+'.csv'
                true_structure = nx.read_edgelist(path = matrix_path, create_using = nx.DiGraph,nodetype=int)
                result = nx.from_numpy_array(result,create_using=nx.DiGraph)
                #aupr_not,curve_not = precision_recall(true_structure,result)
                #SHD_not = SHD(true_structure,result,double_for_anticausal=False)
                #auc_data.append(aupr_not)
                #shd_data.append(SHD_not)
                time_record.append(time_end-time_start)
                #print(aupr_not," ",SHD_not)
        np.savetxt(folder+selected_type+'_OT_time_record.csv',time_record,delimiter=',')
    #auc_data = pd.DataFrame(auc_data,columns = ['IT'])
    #save_path = '_'.join(selected_type.split('_')[1:3])+'_IT_auc.csv'
    #auc_data.to_csv(save_path)
    #save_path ='_'.join(selected_type.split('_')[1:3])+'_IT_shd.csv'
    #shd_data = pd.DataFrame(shd_data,columns = ['IT'])
    #shd_data.to_csv(save_path)

