#!/usr/bin/env python
# coding: utf-8

# In[2]:


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


# In[16]:


G, X, B_true = Base_functions.Data_generator(num_groups = 3,
                           p_in_i = 1.5, p_out_i = 5, total_nodes = 99, cat_cont = 1)


# In[17]:


G, res, X_colors, node_dis, B, Q, positions = Base_functions.Vertex_Positions(G, step_size = 0.1, thresh = 0.000001, X = X, 
                                               gamma = 0.5, B_true = B_true, cat_cont = 1)


# In[19]:


nx.draw(G, pos = res, with_labels = False, node_color = X_colors, ax = None, cmap=plt.cm.Reds)
plt.savefig('plot.png')


# In[ ]:




