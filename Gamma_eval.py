#!/usr/bin/env python
# coding: utf-8

# In[5]:


import Base_functions
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
#import math
import numpy as np
import random as random
import numpy.linalg as np_math
import networkx as nx
import sklearn
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import seaborn as sns
import pandas as pd 
import statsmodels.api as sm
from scipy import stats
from scipy.special import expit, logit
from scipy.special import comb
import sys


# In[ ]:


itera = int(sys.argv[1]) # iteration for parameter set
p_in_i = float(sys.argv[2]) # percent increase
p_out_i = float(sys.argv[3]) # expected edges per node
num_groups = int(sys.argv[4])
total_nodes = int(sys.argv[5])
cat_cont = int(sys.argv[6]) # 1 catagorical, # 2 logistic continuous # 3 MLE



# In[ ]:

G, X, B_true = Base_functions.Data_generator(num_groups = num_groups,
                              p_in_i = p_in_i, p_out_i = p_out_i, total_nodes = total_nodes, cat_cont = cat_cont)
                              
iterations = 50
gammas = np.arange(0.0, 1.05, 0.05)
#gammas = [0]

#plt.figure(figsize=(15, 12))
#plt.subplots_adjust(hspace=0.5)
#plt.suptitle("SBM Graphs \n Varying Gamma", fontsize=18, y=0.95)

select_gamma = np.zeros((len(gammas)*iterations, 2))
select_gamma[:, 0] = np.tile(gammas, iterations)

select_gamma_2 = np.zeros((len(gammas)*iterations, 2))
select_gamma_2[:, 0] = np.tile(gammas, iterations)

for iterate in range(iterations):
    iter_gamma = []
    iter_gamma_2 = []
    for n,gamma in enumerate(gammas):
        # sometimes rounding is weird
        gamma = round(gamma, 4)
        
        # running algorithm
        G, res, X_colors, node_dis, B, Q, positions = Base_functions.Vertex_Positions(G, step_size = 0.1, thresh = 0.000001, X = X, 
                                                  gamma = gamma, B_true = B_true, cat_cont = cat_cont)
    
        A = np.tril(nx.to_scipy_sparse_array(G).todense(), k = -1)
        # tracking selection metric
        #print(node_dis[range(5), :])
        #group_0 = node_dis[range(group_size), :]
        #mean_point = [group_0[:, 0].mean(), group_0[:, 1].mean()]
        #g0 = np.square((group_0 - mean_point)).sum(axis = 1).mean()

        #group_1 = node_dis[range(group_size, 2*group_size), :]
        #mean_point = [group_1[:, 0].mean(), group_1[:, 1].mean()]
        #g1 = np.square((group_1 - mean_point)).sum(axis = 1).mean()
        #print(g0, g1)
        i, j = np.tril_indices(node_dis.shape[0], k=-1)
        #select_data = np.transpose(np.vstack((i, j, node_dis[i, j])))


        #g1 = select_data[(select_data[:, 0] < group_size) & (select_data[:, 1] < group_size)][:, 2].mean()
        #g1sd = select_data[(select_data[:, 0] < group_size) & (select_data[:, 1] < group_size)][:, 2].std()

        #g2 = select_data[(select_data[:, 0] >= group_size) & (select_data[:, 1] >= group_size)][:, 2].mean()
        #g2sd = select_data[(select_data[:, 0] >= group_size) & (select_data[:, 1] >= group_size)][:, 2].std()

        #indicies = np.argwhere(np.tril(nx.to_scipy_sparse_array(G).todense()) == 1)
        #i_1 = indicies[:,0].tolist()
        #j_1 = indicies[:,1].tolist()
        #e_all = node_dis[i_1, j_1].std()
        #m = node_dis[i_1, j_1].mean()
        #sd = node_dis[i_1, j_1].std()
        ##print(gamma, g1, g2)
        
        #B_inv = np.reciprocal(B,
        #                  where = B !=0)
        
        #B_inv_std = (B_inv[i, j] - B_inv[i, j].min()) / (B_inv[i, j].max() - B_inv[i, j].min())
        #B_inv[i, j] = B_inv_std * (1 - 0) + 0
        
        #weighted_dis = np.multiply(node_dis, 
        #                               B_inv)
        
        #if(B[i, j].std() != 0):
        #    m1 = np.multiply((B[i, j] - B[i, j].mean())/B[i, j].std(),
        #                     (node_dis[i, j] - node_dis[i, j].mean())/node_dis[i, j].std()).sum()
        #else:
        #    m1 = np.multiply(B[i, j], 
        #                              (node_dis[i, j] - node_dis[i, j].mean())/node_dis[i, j].std()).sum()
        
        m1 = np.multiply((B[i, j] - B[i, j].mean()),
                             (node_dis[i, j] - node_dis[i, j].mean())).sum()
        
        # A and node_dis is a lower triangular matrix
        #m2 = np.multiply(A, node_dis).sum()/A.sum() + np.multiply(B, node_dis).sum()/(total_nodes*(total_nodes- 1)*B.mean())
        
        # Energy is Q
        #m3 = Q
        
        # Average Edge length
        #m4 = np.multiply(A, node_dis).sum()/A.sum() 
        
        #
        
        #m6 = np.multiply((1 - B[i, j].mean())*A, 
        #            (node_dis - node_dis[i, j].mean())/node_dis[i, j].std()).mean()
        
        #m7 = np.multiply((A + B), node_dis).sum()
        
        #m8 = np.multiply(B[i,j] - B[i,j].mean(), node_dis[i,j] - node_dis[i,j].mean()).sum()
        #m9 = node_dis[i,j].mean()
        #m10 = np.multiply(A, (node_dis - node_dis[i, j].mean())/node_dis[i, j].std()).sum()
        m10 = np.multiply(A, (node_dis - node_dis[i, j].mean())).sum()

        #print((node_dis - node_dis[i, j].mean())/node_dis[i, j].std())
        iter_gamma.append(m1)
        iter_gamma_2.append(m10)
        
        #print(gamma, node_dis)
        
        #if(iterate == iterations - 1):
            ## print(iterate)
            ## add a new subplot iteratively
            #ax = plt.subplot(4, 3, n + 1)
            ##ax = plt.subplot(1, 1, n + 1)
            #ax.set_title(gamma, fontsize=16)
            #ax.autoscale(enable = False, axis = 'both')
            #ax.set(xlim=(-1.8, 1.8), ylim=(-1.8, 1.8))
            ## filter df and plot ticker on the new subplot axis
            #if(cat_cont == 1):
            #    nx.draw(G, pos = res, with_labels = False, node_color = X_colors, ax = None)
            #elif(cat_cont == 2): 
            #    nx.draw(G, pos = res, with_labels = False, node_color = X_colors, ax = None, cmap=plt.cm.Reds)
       
    #print(n)
    #print(iterate)
    #print(gammas, iter_gamma)
    #print(range(iterations*n, iterations*n + iterations))
    #print(iter_gamma)
    iter_gamma_std = (iter_gamma - np.mean(iter_gamma))/np.std(iter_gamma)
    iter_gamma_2_std = (iter_gamma_2 - np.mean(iter_gamma_2))/np.std(iter_gamma_2)
    
    #print(iter_gamma_std)
    select_gamma[range(len(gammas)*iterate, len(gammas)*iterate + len(gammas)), 1] = iter_gamma_std
    select_gamma_2[range(len(gammas)*iterate, len(gammas)*iterate + len(gammas)), 1] = iter_gamma_2_std

    #nx.draw(G, node_color = X_colors, ax = None)
    #print(B)
    #print(B_inv)

#plt.savefig("T13_SBM_Graphs_Varying_Gamma_onerun.pdf")


# In[ ]:


select_gamma_final = np.zeros((len(gammas)*iterations, 3))
select_gamma_final[:, 0] = select_gamma[:, 0]
select_gamma_final[:, 1] = select_gamma[:, 1]
select_gamma_final[:, 2] = select_gamma[:, 1] + select_gamma_2[:, 1]

data = select_gamma_final


# In[ ]:


gamma = Base_functions.Locate_gamma(data, gammas, 2)


# In[ ]:

# Finding p_in and p_out. can be seen in Base functions as well
#if(cat_cont == 1):
#    # Calculates weighted probability such that there is an p_in_i increase but avg degree is 5
#    group_size = total_nodes/num_groups
#
#    n_wg = num_groups*comb(group_size, 2)
#    n_bg = comb(total_nodes, 2) - n_wg
#
#    p_out = (p_out_i/total_nodes)*(n_wg + n_bg)/(n_wg*(1 + p_in_i) + n_bg)
#    p_in = p_out + p_in_i*p_out      
#elif(cat_cont == 2):
#    p_out = p_out_i
#    p_in = p_in_i 
#elif(cat_cont == 3):
#    # edge probabilities and group size for continuous data using MLE
#    tao = p_in_i + 0.01 # 0.01 is considered no homophily
#    theta = (p_out_i * np.sqrt(1 + 2*tao**2)) / (total_nodes - 1) # averge degree is p_out_i
#    p_out = theta
#    p_in = tao


final = [ [0]*6 for i in range(2)]
final[0] = ["total_nodes", "num_groups", "p_in", "p_out", "gamma", "cat_cont"]
final[1] = [int(total_nodes), int(num_groups), round(p_in_i, 2), round(p_out_i, 2), round(gamma,4), int(cat_cont)]

file_name = "IT_" + str(itera) + "_TN_" + str(total_nodes) + "_NG_" + str(num_groups) + "_PI_" + str(round(p_in_i, 4)) + "_PO_" + str(round(p_out_i , 4)) + "_CC_" + str(cat_cont) + ".csv"

if(cat_cont == 1):
	pd.DataFrame(final).to_csv("/home/ot25/Research/JP/Covariate_Smoothing/Results/Categorical/" + file_name, index = False, header = None)
elif(cat_cont == 2 or cat_cont == 3):
	pd.DataFrame(final).to_csv("/home/ot25/Research/JP/Covariate_Smoothing/Results/Continuous/" + file_name, index = False, header = None)
elif(cat_cont == 4 or cat_cont == 5 or cat_cont == 6):
	pd.DataFrame(final).to_csv("/home/ot25/Research/JP/Covariate_Smoothing/Results/Real_Data/" + file_name, index = False, header = None)

