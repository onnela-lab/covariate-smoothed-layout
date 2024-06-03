#!/usr/bin/env python
# coding: utf-8

################################################################
# This script runs the algorithm and saves the recomended nodal coordinates.
################################################################

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
import sys
import os



# In[ ]:

data = pd.read_csv('plotting_data_AddH.csv')

for i in range(data.shape[0]):

    print(i/data.shape[0])
    p_in_i = round(float(data.loc[i, 'p_increase']), 4)
    p_out_i = round(float(data.loc[i, 'p_out']), 4)
    num_groups = int(data.loc[i, 'num_groups'])
    total_nodes = int(data.loc[i, 'total_nodes'])
    cat_cont = int(data.loc[i, 'cc'])
    gamma = round(float(data.loc[i, 'gamma']), 4)


    # In[3]:


    # In[5]:


    G, X, B_true = Base_functions.Data_generator(num_groups = num_groups, 
                                  p_in_i = p_in_i, p_out_i = p_out_i, total_nodes = total_nodes, cat_cont = cat_cont)


    # In[6]:


    G, res, X_colors, node_dis, B, Q, positions =  Base_functions.Vertex_Positions(G = G, step_size = 0.1, thresh = 0.000001, X = X, 
                                                      gamma = gamma, B_true = B_true, cat_cont = cat_cont)


    # In[9]:

    print(positions)
    

    np.savetxt(str("Nodal_Covariates" + 
                   "_TN_" + str(total_nodes) +
                   "_NG_" + str(num_groups) +
                   "_PI_" + str(p_in_i) + 
                   "_PO_" + str(p_out_i) + 
                   "_CC_" + str(cat_cont) + 
                   "_Gamma_" + str(gamma) +
                   ".csv"), np.insert(X, 0, G.nodes(), axis=1), delimiter=',', comments='') # Inserts Node names in first column

    np.savetxt(str("Nodal_Positions" + 
                   "_TN_" + str(total_nodes) +
                   "_NG_" + str(num_groups) +
                   "_PI_" + str(p_in_i) + 
                   "_PO_" + str(p_out_i) + 
                   "_CC_" + str(cat_cont) + 
                   "_Gamma_" + str(gamma) +
                   ".csv"), np.insert(positions, 0, G.nodes(), axis=1), delimiter=',', comments='')

    nx.write_edgelist(G, str("Edge_List" + 
                   "_TN_" + str(total_nodes) +
                   "_NG_" + str(num_groups) +
                   "_PI_" + str(p_in_i) + 
                   "_PO_" + str(p_out_i) + 
                   "_CC_" + str(cat_cont) + 
                   "_Gamma_" + str(gamma) +
                   ".csv"), delimiter=",", data=False)
                   
                   
 


    # In[ ]:





    # In[ ]:





    # In[2]:



