#!/usr/bin/env python
# coding: utf-8

# In[5]:


import Base_functions
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
        i, j = np.tril_indices(node_dis.shape[0], k=-1)
        
        m1 = np.multiply((B[i, j] - B[i, j].mean()),
                             (node_dis[i, j] - node_dis[i, j].mean())).sum()
        

        m10 = np.multiply(A, (node_dis - node_dis[i, j].mean())).sum()

        iter_gamma.append(m1)
        iter_gamma_2.append(m10)
        
    iter_gamma_std = (iter_gamma - np.mean(iter_gamma))/np.std(iter_gamma)
    iter_gamma_2_std = (iter_gamma_2 - np.mean(iter_gamma_2))/np.std(iter_gamma_2)
    
    select_gamma[range(len(gammas)*iterate, len(gammas)*iterate + len(gammas)), 1] = iter_gamma_std
    select_gamma_2[range(len(gammas)*iterate, len(gammas)*iterate + len(gammas)), 1] = iter_gamma_2_std



# In[ ]:


select_gamma_final = np.zeros((len(gammas)*iterations, 3))
select_gamma_final[:, 0] = select_gamma[:, 0]
select_gamma_final[:, 1] = select_gamma[:, 1]
select_gamma_final[:, 2] = select_gamma[:, 1] + select_gamma_2[:, 1]

data = select_gamma_final


# In[ ]:


gamma = Base_functions.Locate_gamma(data, gammas, 2)


# In[ ]:


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

