###############################################################
# This scripts cerates the homophilly strength plots.
# We assume 100 nodes and aveage degree of 5
# All probabilities are relative to the mean.
###############################################################


# Load required libraries
library(tidyr)
library(ggplot2)

# Creating Fucntions
expit <- function(p_1, n = 100){ 
  # n: number of nodes in the graph
  # p_1 is percent increase
  
  x <- sort(runif(10000, 18, 90))
  x_j <- mean(x)
  x_tilda <- abs(x - x_j)
  
  
  p_out <-  ((5*2)/n)/(2 + p_1) # This estimates the average degree is 5 (not exact)
  p_in <-  p_out + p_1*p_out 
  
  #print(c(p_0, p_1))
  
  B0 <- log(p_in / (1 - p_in))
  B1 <- (log(p_out / (1 - p_out)) - B0)/(90 - 17)
  
  return(1/(1 + exp(-(B0 + B1*x_tilda))))
}

MLE <- function(p_1, n = 100){
  tao = p_1 + 0.01
  theta = (5*sqrt(1 + 2*tao^2))/(n -1) # average degree is 5
  
  x <- sort(rnorm(10000, 0, tao))
  
  return(theta * exp(-(x)^2))
}

# Homphilly strengths to test
p_1 <- c(0, 0.1, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4)


# Constructing Data and Plots
Data_generator <- function(type){
  # type: uniform, normal
  
  if(type == "Normal"){
    # Doing Normal
    data <- sapply(p_1, function(p) MLE(p))
  }else{
    if(type == "Uniform"){
      data <- sapply(p_1, function(p) expit(p))
    }
  }
  colnames(data) <- p_1
  data = data.frame(data)
  data$x <- 1:dim(data)[1]
  
  # convert the data frame from wide to long format using tidyr
  df_long <- tidyr::gather(data, key = "variable", value = "value", -x)
  
  # plot the line graph using ggplot2
  ggplot(data = df_long, aes(x = x, y = value, color = variable)) + 
    geom_line() + 
    labs(x = "Continuous covariate", y = "Linkage probability assuming
       average partner", title = paste(type, " Homophilly Plot (100 Nodes)"), sep = "") +
    scale_color_discrete(labels = paste(p_1 * 100, "%", sep = ""), 
                         name = "Homophilly strength")
}


# Running Results
Data_generator(("Normal"))

Data_generator(("Uniform"))





