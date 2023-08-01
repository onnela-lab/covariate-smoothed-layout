# covariate-smoothed-layout

This code constructs plotting visuals, using a smoothing algorithm, incorporating edge information located within covaraites. Varying levels of information within the covaraites are explored, and an added layer of robustness of nodal positions is investigated. 

# File Overview

## Core Functions
Base_functions.py: main class of functions to generate our data, contruct our energy function, optimize our energy function, and select our tuning parameter.

## Data Generation

Gamma_eval.py: Excecutes data generation for a set level of informative coavariates, and runs our layout algorithm for a variety of tuning paramters. The optimal tuning parameter is selected, and exported to a .csv.

Missingness_Plots.py: Excecutes data generation for a set level of informative coavariates, deletes edges completely at random, and gages the robustness of the nodal positions in the recomended layout from our algorithm. 

Plot_positions.py: Excecutes data generation for a set level of informative coavariates, runs our algorithm, and stoes the plotting coordinates to be transfered to R.

## Processing
gamma_table_creator.R: This script pulls the Gamma_eval.py selction data into a neat table.

## Visulaization
gamma_eval.R: This script creates the Gamma_eval.py plots where we see the visuals as a function of gamma.

plotting_code.R: This code visulaizes are simulated and empirical data. 

procrustes_plots.R: This script creates our plots gaging the robustness of our nodal positions. 
