#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# This script investigates the robustness of the layout to missing data. 

#!/usr/bin/env python
# coding: utf-8
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
from scipy.spatial import procrustes
import sys
from scipy.special import comb
import Base_functions
import os


itera = int(sys.argv[1]) # iteration for parameter set
p_in_i = float(sys.argv[2]) # percent increase
p_out_i = float(sys.argv[3]) # expected edges per node
num_groups = int(sys.argv[4])
total_nodes = int(sys.argv[5])
cat_cont = int(sys.argv[6]) # 1 catagorical, # 2 continuous

# Generate Data
G, X, B_true = Base_functions.Data_generator(num_groups = num_groups,
                              p_in_i = p_in_i, p_out_i = p_out_i, total_nodes = total_nodes, cat_cont = cat_cont)

# How many samples of missingness to take
# 20 for procrutes and 5 for visuals
grid = 5
#grid = 20

missing = np.linspace(0, 0.9, grid) 
diff = [] 
FR_diff = [] 
all_positions = np.zeros((grid, total_nodes, 2)) 
FR_all_positions = np.zeros((grid, total_nodes, 2))

all_dist = np.zeros((grid, 1, )) 
FR_all_dist = np.zeros((grid, 1, ))

j, k = np.tril_indices(total_nodes, k=-1)

# selecting for gamma
gamma = Base_functions.Gamma_Selector(G, X, B_true, cat_cont)

# Base Case
G_1, res, X_colors, node_dis, B, Q, positions = Base_functions.Vertex_Positions(G, step_size = 0.1, thresh = 0.000001, X = X, 
                                                  gamma = gamma, B_true = B_true, cat_cont = cat_cont)

FR_pos = nx.fruchterman_reingold_layout(G_1, iterations=20)
FR_pos_raw = np.array([FR_pos[node] for node in G_1.nodes()])

X_std = (FR_pos_raw[:, 0] - FR_pos_raw[:, 0].min()) / (FR_pos_raw[:, 0].max() - FR_pos_raw[:, 0].min())
FR_pos_raw[:, 0] = X_std * (0.9 - -0.9) + -0.9

Y_std = (FR_pos_raw[:, 1] - FR_pos_raw[:, 1].min()) / (FR_pos_raw[:, 1].max() - FR_pos_raw[:, 1].min())
FR_pos_raw[:, 1] = Y_std * (0.9 - -0.9) + -0.9

FR_node_dis = np.tril(np.sqrt(np.square((FR_pos_raw[:, None, :] - FR_pos_raw[None, :, :])).sum(axis=-1)), -1)

base_d = positions
FR_base_d = FR_pos_raw


# Saving info for visual plots
os.chdir("/home/ot25/Research/JP/Covariate_Smoothing/Results/Missing_Plot_Data/Raw/Visuals")
# Other Cases
for i in range(grid): 
    print(i) 
    # create a copy of the graph 
    G_1 = G.copy()

    # calculate number of edges to remove
    num_edges = G_1.number_of_edges()
    num_edges_to_remove = int(missing[i] * num_edges)

    # select random subset of edges to remove
    edges_to_remove = random.sample(list(G_1.edges()), num_edges_to_remove)

    # remove edges from copy of graph
    G_1.remove_edges_from(edges_to_remove)

    # Averaging out nodal distances at each level of missingness
    x1 = []
    x2 = []
    
    for m in range(1): # Iterations are pulled together later
        #selecting for gamma
        gamma = Base_functions.Gamma_Selector(G_1, X, B_true, cat_cont)
        G_1, res, X_colors, node_dis, B, Q, positions = Base_functions.Vertex_Positions(G_1, step_size = 0.1, thresh = 0.000001, X = X, 
                                                  gamma = gamma, B_true = B_true, cat_cont = cat_cont)

        FR_pos = nx.fruchterman_reingold_layout(G_1, iterations=20)
        FR_pos_raw = np.array([FR_pos[node] for node in G_1.nodes()])

        X_std = (FR_pos_raw[:, 0] - FR_pos_raw[:, 0].min()) / (FR_pos_raw[:, 0].max() - FR_pos_raw[:, 0].min())
        FR_pos_raw[:, 0] = X_std * (0.9 - -0.9) + -0.9

        Y_std = (FR_pos_raw[:, 1] - FR_pos_raw[:, 1].min()) / (FR_pos_raw[:, 1].max() - FR_pos_raw[:, 1].min())
        FR_pos_raw[:, 1] = Y_std * (0.9 - -0.9) + -0.9

        FR_node_dis = np.tril(np.sqrt(np.square((FR_pos_raw[:, None, :] - FR_pos_raw[None, :, :])).sum(axis=-1)), -1)

        all_positions[i] = positions
        FR_all_positions[i] = FR_pos_raw

        mtx1, mtx2, d = procrustes(all_positions[i], base_d)
        mtx1, mtx2, FR_d = procrustes(FR_all_positions[i], FR_base_d)


        x1.append(d)
        x2.append(FR_d)

    all_dist[i] = x1
    FR_all_dist[i] = x2


    diff.append(sum(x1)/len(x1))
    FR_diff.append(sum(x2)/len(x2))
    
    
    
    
    np.savetxt(str("Nodal_Covariates" + 
                   "_IT_" + str(itera) +
                   "_TN_" + str(total_nodes) +
                   "_NG_" + str(num_groups) +
                   "_PI_" + str(p_in_i) + 
                   "_PO_" + str(p_out_i) + 
                   "_CC_" + str(cat_cont) + 
                   "_Gamma_" + str(gamma) +
                   "_Missing_" + str(np.round(missing[i], 4)) +
                   ".csv"), np.insert(X, 0, G_1.nodes(), axis=1), delimiter=',', comments='') # Inserts Node names in first column

    np.savetxt(str("US_Nodal_Positions" + 
                   "_IT_" + str(itera) +
                   "_TN_" + str(total_nodes) +
                   "_NG_" + str(num_groups) +
                   "_PI_" + str(p_in_i) + 
                   "_PO_" + str(p_out_i) + 
                   "_CC_" + str(cat_cont) + 
                   "_Gamma_" + str(gamma) +
                   "_Missing_" + str(np.round(missing[i], 4)) +
                   ".csv"), np.insert(positions, 0, G_1.nodes(), axis=1), delimiter=',', comments='')
    
    np.savetxt(str("FR_Nodal_Positions" + 
                   "_IT_" + str(itera) +
                   "_TN_" + str(total_nodes) +
                   "_NG_" + str(num_groups) +
                   "_PI_" + str(p_in_i) + 
                   "_PO_" + str(p_out_i) + 
                   "_CC_" + str(cat_cont) + 
                   "_Gamma_" + str(gamma) +
                   "_Missing_" + str(np.round(missing[i], 4)) +
                   ".csv"), np.insert(FR_pos_raw, 0, G_1.nodes(), axis=1), delimiter=',', comments='')

    nx.write_edgelist(G_1, str("Edge_List" + 
                   "_IT_" + str(itera) +
                   "_TN_" + str(total_nodes) +
                   "_NG_" + str(num_groups) +
                   "_PI_" + str(p_in_i) + 
                   "_PO_" + str(p_out_i) + 
                   "_CC_" + str(cat_cont) + 
                   "_Gamma_" + str(gamma) +
                   "_Missing_" + str(np.round(missing[i], 4)) +
                   ".csv"), delimiter=",", data=False)
    
    
    
result = np.column_stack((np.round(missing, 4), np.round(diff, 4), np.round(FR_diff, 4)))

os.chdir("/home/ot25/Research/JP/Covariate_Smoothing/Results/Missing_Plot_Data/Raw/Procrustes")


