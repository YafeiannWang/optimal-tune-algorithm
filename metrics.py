import numpy as np
import pdb
import time
import csv
from igraph import *
import pandas as pd
import networkx as nx
from cdt.metrics import precision_recall,SHD,SID

if __name__ == '__main__':

    file_path = 'predict_matrix'
    type_path =['sig_Gau']#['mixture_squad']##['linear_Gau','mixture_linear']
    method = 'OT'
    folder_list = ['1/','2/','3/','4/','5/','6/','7/','8/']#,'9/']#,'4/']

    N = [10]#,20,25,30,40]#,50] #define the number of features
    edges = [40]#,100,150,250,400]#,1000] #define the number of edges on dag
    #path_type = ['generate_mixture_squad_node','generate_mixture_linear_node','generate_nonlinear_Gau_node','generate_linear_nonGau_node','generate_linear_Gau_node']
    #selected_type = path_type[]
    results = []

    for select_type in type_path:
        mean_sid = []
        std_sid = []
        mean_aupr = []
        std_aupr = []
        for node_number,edge_number in zip(N,edges):
            sum_sid = []
            sum_aupr = []
            for folder in folder_list:
                matrix_save_path = './'+folder+select_type+'_predict_matrix_'+method+'_node'+str(node_number)+'_edge'+str(edge_number)+'.csv'
                est_structure = np.loadtxt(matrix_save_path,delimiter=',')
                #est_structure = nx.from_numpy_array(est_structure,create_using=nx.DiGraph)
                matrix_path = './'+folder+'relation_matrix_node'+str(node_number)+'_edge'+str(edge_number)+'.csv'
                true_structure = np.loadtxt(matrix_path, delimiter=',')
                sid = SID(true_structure,est_structure)
                aupr_re,aupr_curve = precision_recall(true_structure,est_structure)
                sum_sid.append(sid)
                sum_aupr.append(aupr_re)
            mean_sid.append(np.mean(sum_sid))
            std_sid.append(np.std(sum_sid))
            mean_aupr.append(np.mean(sum_aupr))
            std_aupr.append(np.std(sum_aupr))

        save_path = '_'.join([select_type,method,'mean','sid'])
        mean_sid = pd.DataFrame(mean_sid,columns = [method])
        mean_sid.to_csv(save_path+'.csv')
        save_path = '_'.join([select_type,method,'std','sid'])
        std_sid = pd.DataFrame(std_sid,columns = [method])
        std_sid.to_csv(save_path+'.csv')
        save_path = '_'.join([select_type,method,'mean','aupr'])
        mean_aupr = pd.DataFrame(mean_aupr,columns = [method])
        mean_aupr.to_csv(save_path+'.csv')
        save_path = '_'.join([select_type,method,'std','aupr'])
        std_aupr = pd.DataFrame(std_aupr,columns = [method])
        std_aupr.to_csv(save_path+'.csv')
