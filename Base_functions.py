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
from scipy.special import comb
from scipy.optimize import root_scalar
from scipy.integrate import quad
from sklearn.preprocessing import LabelEncoder
import sys
import os

    
def Q_eval(A, positions, alpha, k, B, gamma): # k is no longer needed here
    sqdist = torch.tril((positions[:, None, :] - positions[None, :, :]).square().sum(axis=-1))
    i, j = np.tril_indices(sqdist.size(0), k=-1)
    sqdist_nozero = sqdist[i, j]
    repulsive = 0
    attractive1 = 0
    
    # modifying diagnoal because of sqrt(0) derivative
    repulsive = (alpha/sqdist_nozero.sqrt()).mean()
    attractive = torch.mul(((1 - gamma)*A + gamma*B), sqdist).mean()
        
    Q = attractive + repulsive
    return Q

def B_function(A, X, cat_cont):
    # Determining lower diagonal indicies
    i, j = np.tril_indices(A.size(0), k=-1)
    
    # Constructing features and regression outcome
    ## outcome
    A_o = torch.flatten(A[i, j])
    
    ## feature
    X = torch.from_numpy(X)
    
    
    if(cat_cont != 3):
        
        if(cat_cont == 1):
            xsq_f = np.column_stack(((X[i] != X[j]).square())).reshape(-1, 1)
        elif(cat_cont == 2):
            xsq_f = np.column_stack((abs(X[i] - X[j]))).reshape(-1, 1)
        elif(cat_cont == 4):
            xsq_f = np.column_stack(((X[i, 0] == X[j, 0]), (X[i, 1] == X[j, 1]), (X[i, 2] == X[j, 2]), (X[i, 3] == X[j, 3])))
            #xsq_f = np.column_stack(((X[i, 2] != X[j, 2]), (X[i, 3] != X[j, 3])))
        elif(cat_cont == 5):
                #xsq_f = np.column_stack(((X[i, 0] == X[j, 0]), (X[i, 1] == X[j, 1])))
                xsq_f = np.column_stack((X[i, 3] == X[j, 3])).reshape(-1, 1)
        elif(cat_cont == 6):
            xsq_f = np.column_stack((X[i, 0] == X[j, 0])).reshape(-1, 1)
            
        # Logistic Regression
        if all(x == A_o[0] for x in A_o):
            B_matrix = np.zeros((A.size(0), A.size(0)))
        else:
            model = LogisticRegression().fit(xsq_f, A_o.numpy().ravel())
            proba = model.predict_proba(xsq_f)[:, 1]
            # Create a column of True values
            column = np.full((xsq_f.shape[0], 1), True)

            # Concatenate the column with the array
            X_t = np.hstack((column, xsq_f))


            model = LogisticRegression().fit(xsq_f, A_o.numpy().ravel())
            proba = model.predict_proba(xsq_f)[:, 1]

            model_sm = sm.Logit(A_o.numpy().ravel(), X_t)
            result = model_sm.fit(tol=1e-1)

            # Builing B matrix 
            B_matrix = np.zeros((A.size(0), A.size(0)))
            B_matrix[i, j] = proba
    else:
        def log_L(A, X, theta, tao):
            i, j = np.tril_indices(A.size(0), k=-1)
            sqdist_x = torch.square((X[:, None, :] - X[None, :, :]).sum(axis=-1))
  
            L = torch.sum(torch.mul(A[i,j], torch.log(theta*torch.exp(-sqdist_x[i,j]/2))) + \
                          torch.mul(1 - A[i,j], torch.log(1 - theta*torch.exp(-sqdist_x[i,j]/2)))) #+ \
                          #torch.log(torch.mul(1/tao, torch.mul(1/torch.sqrt(sqdist_x[i,j]), \
                          #                                     torch.exp(torch.mul(1/torch.square(tao), -sqdist_x[i,j]/2))))))
                          
            return -1*L
            
        theta = torch.tensor([0.5], requires_grad=True)
        #tao = torch.tensor([2.0], requires_grad=True)
        
        optimizer = torch.optim.SGD([theta], lr=0.000001)
        
        for i in range(1000):
            # Compute the value of the function and its gradient
            #loss = log_L(A, X, theta, torch.clamp(tao_prime, min=0, max=50))
            
            loss = log_L(A, X, theta, tao = 1) #tao doesnt matter here becase it isnt in likelihood
                
            loss.backward()

            # Update the variables
            optimizer.step()
            
            torch.clamp(theta, min=0, max=1)
            #torch.clamp(tao, min=0, max=10)

            # Zero the gradients
            optimizer.zero_grad()
        
        X = X.detach().numpy()
        theta = theta.item()
        
        def theta_ij(i, j, X, theta):
            return theta * np.exp(-(X[i] - X[j])**2 / 2)
    
        B_matrix = np.fromfunction(lambda i, j: theta_ij(i, j, X, theta), (total_nodes, total_nodes), dtype=int)
        B_matrix = np.squeeze(B_matrix)
            
    return B_matrix
     
def Vertex_Positions(G, step_size, thresh, X, gamma, B_true, cat_cont):
    # Initial graph
    for iteration in range(1):
        A = torch.tensor(nx.to_scipy_sparse_array(G).todense())
    
        # Initial parameters
        k = 2
        alpha = 2
    
        if iteration == 0:
            # Constructing B matrix
            B = B_function(A, X, cat_cont)
        else:
            B = B_true

        positions = torch.rand(A.shape[0], 2, requires_grad=True)
    
        # Optimizer
        optimizer = torch.optim.Adam([positions], lr=step_size)
    
        track = []
        x = 1000
        while x > thresh:
        #for k in range(500):
            # Compute Energy
            Q = Q_eval(A, positions, alpha, k, B, gamma)
            track.append(Q.detach())
        
            # Back Propogation
            optimizer.zero_grad()
            Q.backward()
            optimizer.step()
            if len(track) > 2:
                x = abs(track[-2:][0] - track[-2:][1])
        
    
        positions = positions.detach().numpy()
        # Scaling node positions
        
        X_std = (positions[:, 0] - positions[:, 0].min()) / (positions[:, 0].max() - positions[:, 0].min())
        positions[:, 0] = X_std * (0.9 - -0.9) + -0.9
        
        Y_std = (positions[:, 1] - positions[:, 1].min()) / (positions[:, 1].max() - positions[:, 1].min())
        positions[:, 1] = Y_std * (0.9 - -0.9) + -0.9
    
        res = dict(zip(G.nodes(), positions))
        node_dis = np.tril(np.sqrt(np.square((positions[:, None, :] - positions[None, :, :])).sum(axis=-1)), -1)
        
        # Node Colors
        X_colors = []
        if(cat_cont == 1):
            for x in X:
                if x == 0:
                    X_colors.append("red")
                elif x == 1:
                    X_colors.append("blue")
                elif x == 2 :
                    X_colors.append("grey")
                elif x == 3:
                    X_colors.append("black")
                else:
                    X_colors.append("green")
        elif(cat_cont == 2 or cat_cont == 3):
            carac = pd.DataFrame({ 'ID':range(G.number_of_nodes()), 'myvalue':X[:, 0].tolist()})
            carac= carac.set_index('ID')
            carac=carac.reindex(G.nodes())
            X_colors = carac['myvalue'].astype(float)
        elif(cat_cont == 4):
            carac = pd.DataFrame({ 'ID':G.nodes(), 'myvalue':X[:, 0].tolist()})
            carac= carac.set_index('ID')
            carac=carac.reindex(G.nodes())
            X_colors = carac['myvalue'].astype(float)
            
    return G, res, X_colors, node_dis, B, Q.detach().numpy(), positions



def Locate_gamma(data, gammas, metric):
    # data: array containing the metric information for each gammas
    # gammas: gammas to consider
    # metric to consider
        # 1: smoothing metric
        # 2: edge and smoothing metric
    # Checking for decreasing smoothing trend in the data
    x = np.array(gammas).reshape((-1, 1))
    y = []
    
    for gam in gammas:
        data_gam = data[data[:, 0] == gam, 1]
        y.append(np.median(data_gam))
    
    x2 = sm.add_constant(x)
    model = sm.OLS(y, x2).fit()
    
    # Finding gamma as minimum of convex fucntion of smoothing and edge increase
    y = []
    for gam in gammas:
        data_gam = data[data[:, 0] == gam, 2]
        y.append(np.median(data_gam))
        
    return(round(gammas[np.array(y).argmin()], 4))


# Graph and covariate construction

def Data_generator(num_groups, p_in_i, p_out_i, total_nodes, cat_cont):
    # Generates our Network and Covariate Data
    if(cat_cont == 1):
        # Calculates weighted probability such that there is an p_in_i increase but avg degree is 5
        group_size = total_nodes/num_groups

        n_wg = num_groups*comb(group_size, 2)
        n_bg = comb(total_nodes, 2) - n_wg

        p_out = (p_out_i/total_nodes)*(n_wg + n_bg)/(n_wg*(1 + p_in_i) + n_bg)
        p_in = p_out + p_in_i*p_out
        
        print(p_out, p_in)
        
        #random.seed(10)
        G = nx.planted_partition_graph(int(num_groups), int(group_size), p_in, p_out)
        random.seed(None)
        X = np.asarray([block["block"] for node, block in G.nodes(data = True)])
        X = X.reshape(-1, 1)

        # Adjanceny matrix
        A = np.tril(nx.to_scipy_sparse_array(G).todense(), k = -1)


        # True Block matrix
        B_true = np.zeros((total_nodes, total_nodes))
        for i in range(1, total_nodes):
            for j in range(0, i):
                if X[i] == X[j]:
                    B_true[i, j] = p_in
                else:
                    B_true[i, j] = p_out
                #print(i, j, B_true[i,j], X[i], X[j])
    elif(cat_cont == 2):
        B_1 = -p_in_i

        # Define the function to integrate
        def f(z, b_0, b_1 = B_1):
            integrad = (2 - 2*z)/(1 + np.exp(-(b_0 + b_1*z)))
            return integrad

        # Define the function to solve for
        def Q(b_0):
        # integral minus average degree
            return quad(f, a=0, b=1, args=(b_0))[0] - p_out_i/(total_nodes - 1)


        sol = root_scalar(Q, x0= (1.0), x1 = (0.0))


        B_0 = sol.root


        G = nx.empty_graph(total_nodes)

        #random.seed(10)
        X = np.sort(np.random.uniform(low=0, high=1, size=total_nodes))
        X = X.reshape(-1, 1)
        random.seed(10)
        B_true = np.zeros((total_nodes, total_nodes))
        for i in range(1, total_nodes):
            for j in range(0, i):
                p_ij = expit(B_0 + B_1*abs(X[i] - X[j]))
                B_true[i, j] = p_ij

                if np.random.uniform(low = 0, high = 1, size = 1) < p_ij:
                    G.add_edge(i, j)
        
    elif(cat_cont == 3):
        # edge probabilities and group size for continuous data using MLE
        tao = p_in_i + 0.01
        theta = (p_out_i * np.sqrt(1 + 2*tao**2)) / (total_nodes - 1) # averge degree is p_out_i
        
        G = nx.empty_graph(total_nodes)
        random.seed(10)
        X = np.sort(np.random.normal(loc=0, scale=tao, size=total_nodes))
        X = X.reshape(-1, 1)
        random.seed(10)
        
        B_true = np.zeros((total_nodes, total_nodes))
        for i in range(1, total_nodes):
            for j in range(0, i):
                #p_ij = expit(-1.8 + -0.05*abs(X[i] - X[j]))
                p_ij = theta * np.exp(-(X[i] - X[j])**2 / 2)
                B_true[i, j] = p_ij
                #print(p_ij)

                if np.random.uniform(low = 0, high = 1, size = 1) < p_ij:
                    G.add_edge(i, j)
    elif(cat_cont == 4 or cat_cont == 5 or cat_cont == 6): # real data
        
        # read in the XML file as a networkx graph
        G = nx.read_graphml("/home/ot25/Research/JP/Covariate_Smoothing/Data/community_04.xml")

        # force the graph to be undirected
        G = G.to_undirected()
        
        for u, v, attrs in G.edges(data=True):
            if 'weight' in attrs:
                del attrs['weight']

        nodes, sex, race, grade, school = zip(*[(node, attrs['sex'], attrs['race'], attrs['grade'], attrs['school']) for node, attrs in G.nodes(data=True)])

        nodes = np.array(nodes)
        sex = np.array(sex)
        race = np.array(race)
        grade = np.array(grade)
        school = np.array(school)

        pos_miss_cov = np.column_stack((sex, race, grade))

        # identify the nodes with 0 as missing
        grade_0_indices = np.where((pos_miss_cov == "0").any(axis=1))[0]


        # remove the corresponding nodes and attributes from the arrays and the graph
        for index in reversed(grade_0_indices):
            node = nodes[index]
            G.remove_node(node)
            nodes = np.delete(nodes, index)
            sex = np.delete(sex, index)
            race = np.delete(race, index)
            grade = np.delete(grade, index)
            school = np.delete(school, index)
            
        # Building data to be exported School has no missing data
        X = np.column_stack((sex, race, grade, school))
        le = LabelEncoder()
        for i in range(X.shape[1]):
            X[:, i] = le.fit_transform(X[:, i])
            
        X = X.astype(int)
            
        B_true = [] # There is no B_ture to use on real data
        
        
    return G, X, B_true
        

def Gamma_Selector(G, X, B_true, cat_cont):
    iterations = 50
    gammas = np.arange(0.0, 1.05, 0.05)

    # Gamma will be selected from these lists
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
            G, res, X_colors, node_dis, B, Q, positions = Vertex_Positions(G, step_size = 0.1, thresh = 0.000001, X = X, 
                                                      gamma = gamma, B_true = B_true, cat_cont = cat_cont)

            A = np.tril(nx.to_scipy_sparse_array(G).todense(), k = -1)

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



    select_gamma_final = np.zeros((len(gammas)*iterations, 3))
    select_gamma_final[:, 0] = select_gamma[:, 0]
    select_gamma_final[:, 1] = select_gamma[:, 1]
    select_gamma_final[:, 2] = select_gamma[:, 1] + select_gamma_2[:, 1]

    data = select_gamma_final


 

    gamma = Locate_gamma(data, gammas, 2)
    
    return(gamma)
