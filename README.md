# covariate-smoothed-layout

This code constructs plotting visuals, using a smoothing algorithm, incorporating edge information located within covaraites. Varying levels of information within the covaraites are explored, as well as covariate data types and network structures. Finally, an added layer of robustness of nodal positions is investigated. 

# File Overview

## Package installation
requirements.txt contains a list of packages that need to be installed

## Core Functions
Base_functions.py: main class of functions to generate our data, contruct our energy function, optimize our energy function, and select our tuning parameter.

## Data Generation

Gamma_eval.py: Excecutes data generation for a set level of informative coavariates, and runs our layout algorithm for a variety of tuning paramters. The optimal tuning parameter is selected, and exported to a .csv.

Missingness_Plots.py: Excecutes data generation for a set level of informative coavariates, deletes edges completely at random, and gages the robustness of the nodal positions in the recomended layout from our algorithm. 

Plot_positions.py: Excecutes data generation for a set level of informative coavariates, runs our algorithm, and stoes the plotting coordinates to be transfered to R.

## Processing
gamma_table_creator.R: This script pulls the Gamma_eval.py selction data into a neat table for our paper.

## Visulaization
gamma_eval.R: This script creates the Gamma_eval.py plots where we see the visuals as a function of gamma.

plotting_code.R: This code visulaizes simulated and empirical data. 

procrustes_plots.R: This script creates our plots gaging the robustness of our nodal positions. 

## Usage
Below is a basic example of how to run our algorithm on a network with categorical nodal covariates with 3 groups, informative covariates, 2.5:1 odds off connection within verses between, (p_in_i = 1.5), average nodal degree of 5, 99 nodes, and gamma value of 0.5. The code can also be ran by running the following code in the terminal after downloading the needed files Base_functions.py, run_code.py: 'make -f Makefile.txt'

The calculated within and between connection probabilities are printed on the screen as well as written here respectively rounded to 4 digits: 0.0847, 0.0339.

1. Import the necessary packages/files.
   
    ```
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
    ```

3. Generate Data

   ```
   G, X, B_true = Base_functions.Data_generator(num_groups = 3,
                              p_in_i = 1.5, p_out_i = 5, total_nodes = 99, cat_cont = 1)
   ```
   
5. Run algorithm

   ```
   G, res, X_colors, node_dis, B, Q, positions = Base_functions.Vertex_Positions(G, step_size = 0.1, thresh = 0.000001, X = X, 
                                                  gamma = 0.5, B_true = B_true, cat_cont = 1)
   ```

7. Plot graph using the exported/configured positions.

   ```
   nx.draw(G, pos = res, with_labels = False, node_color = X_colors, ax = None, cmap=plt.cm.Reds)
   ```

