########################################################################
# This script pulls the gamma selction data into a neat table.
########################################################################
library(ggplot2)
library(gridExtra)
library(grid)
library(cowplot)
library(stringr)


Gamma_Summary <- function(cat_cont){
  #cat_cont: 1, catagorical/ 2, continuous
  
  
  if(cat_cont == 1){
    setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Categorical/Raw_Data")
    
    all_files <- list.files()
    all_files <- all_files[-which(all_files == "Old")]
    
    
    raw_data <- matrix(NA, 0, 5)
    colnames(raw_data) <- c("total_nodes", "num_groups", "p_in", "p_out", "gamma")
    
    for(i in 1:length(all_files)){
      temp <- read.csv(all_files[i])
      
      raw_data <- rbind(raw_data, temp)
    }
    
    
    total_nodes <-  sort(unique(raw_data[, "total_nodes"]))
    num_groups <- c(2, 5)
    p_increase <- sort(unique(raw_data[, "p_in"]))
    
    final_data <- matrix(NA, length(total_nodes)*length(num_groups), length(p_increase))

    for(TN in 1:length(total_nodes)){
      for(p in 1:length(p_increase)){
        i <- 2*TN - 1
        j <- 2*TN
        
        temp_i <- raw_data[which(raw_data[, "total_nodes"] == total_nodes[TN] & 
                                   raw_data[, "p_in"] == p_increase[p] & 
                                   raw_data[, "num_groups"] == num_groups[1]), ]
        
        if(dim(temp_i)[1] < 2){
          final_data[TN, p] <- NA
        }else{
          final_data[i, p] <- paste(round(mean(temp_i[, "gamma"]), 4), 
                                    " (", round(sd(temp_i[, "gamma"]), 4), ")", sep = "")
        }
        
        
        temp_j <- raw_data[which(raw_data[, "total_nodes"] == total_nodes[TN] & 
                                   raw_data[, "p_in"] == p_increase[p] & 
                                   raw_data[, "num_groups"] == num_groups[2]), ]
        
        if(dim(temp_j)[1] < 2){
          final_data[TN, p] <- NA
        }else{
          final_data[j, p] <- paste(round(mean(temp_j[, "gamma"]), 4), 
                                    " (", round(sd(temp_j[, "gamma"]), 4), ")", sep = "")
        }
      }
    }
    
    colnames(final_data) <- paste(p_increase*100, "%", sep = "")
    rownames(final_data) <- paste(rep(total_nodes, each = 2), "_", num_groups, sep = "")
    
    
    setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Categorical")
    
    write.csv(final_data, "categorical_gamma_results.csv")
    
  }else{
    if(cat_cont == 2){
      setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Continuous/Raw_Data")
      
      all_files <- list.files()
      all_files <- all_files[-which(all_files == "Old")]
      
      
      raw_data <- matrix(NA, 0, 5)
      colnames(raw_data) <- c("total_nodes", "num_groups", "p_in", "p_out", "gamma")
      
      for(i in 1:length(all_files)){
        temp <- read.csv(all_files[i])
        
        raw_data <- rbind(raw_data, temp)
      }
      
      p_increase <- sort(unique(raw_data[, "p_in"]))
      
      final_data <- matrix(NA, length(total_nodes), length(p_increase))
      
      
      
      
      for(TN in 1:length(total_nodes)){
        for(p in 1:length(p_increase)){
          
          temp_i <- raw_data[which(raw_data[, "total_nodes"] == total_nodes[TN] & 
                                     raw_data[, "p_in"] == p_increase[p]), ]
          
          
          if(dim(temp_i)[1] < 2){
            final_data[TN, p] <- NA
          }else{
            final_data[TN, p] <- paste(round(mean(temp_i[, "gamma"]), 4), 
                                       " (", round(sd(temp_i[, "gamma"]), 4), ")", sep = "")
          }
        }
      }
      
      colnames(final_data) <- paste(p_increase*100, "%", sep = "")
      rownames(final_data) <- paste(total_nodes, "c", sep = "_")
      
      
      setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Continuous")
      
      write.csv(final_data, "continuous_gamma_results.csv")
      
    }else{
      if(cat_cont == 3){
        setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Continuous/Raw_Data")
        
        all_files <- list.files()
        all_files <- all_files[-which(all_files == "Old")]
        
        
        raw_data <- matrix(NA, 0, 5)
        colnames(raw_data) <- c("total_nodes", "num_groups", "p_in", "p_out", "gamma")
        
        for(i in 1:length(all_files)){
          temp <- read.csv(all_files[i])
          
          raw_data <- rbind(raw_data, temp)
        }
        
        p_increase <- raw_data[, "p_in"] - 0.01
        raw_data <- cbind(raw_data, p_increase)
        
        
        total_nodes <-  sort(unique(raw_data[, "total_nodes"]))
        num_groups <- total_nodes
        p_increase <- sort(unique(raw_data[, "p_increase"]))
        
        final_data <- matrix(NA, length(total_nodes), length(p_increase))
        
        
        
        
        for(TN in 1:length(total_nodes)){
          for(p in 1:length(p_increase)){
            
            temp_i <- raw_data[which(raw_data[, "total_nodes"] == total_nodes[TN] & 
                                       raw_data[, "p_increase"] == p_increase[p]), ]
            
            
            if(dim(temp_i)[1] < 2){
              final_data[TN, p] <- NA
            }else{
              final_data[TN, p] <- paste(round(mean(temp_i[, "gamma"]), 4), 
                                         " (", round(sd(temp_i[, "gamma"]), 4), ")", sep = "")
            }
          }
        }
        
        colnames(final_data) <- paste(p_increase*100, "%", sep = "")
        rownames(final_data) <- paste(total_nodes, "c", sep = "_")
        
        
        setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Continuous")
        
        write.csv(final_data, "continuous_gamma_results.csv")
        
      }
    }
  }
  
  return(final_data)
}

Plot_Params <- function(total_nodes, p_increase, p_out, num_groups){
  # Reading in Categorical Data
  
  setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Categorical/Raw_Data")
  
  all_files <- list.files()
  all_files <- all_files[-which(all_files == "Old")]
  
  
  cat_raw_data <- matrix(NA, 0, 5)
  colnames(cat_raw_data) <- c("total_nodes", "num_groups", "p_in", "p_out", "gamma")
  
  for(i in 1:length(all_files)){
    temp <- read.csv(all_files[i])
    
    cat_raw_data <- rbind(cat_raw_data, temp)
  }
  
  
  # Reading in Continuous Data
  setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Continuous/Raw_Data")
  
  all_files <- list.files()
  all_files <- all_files[-which(all_files == "Old")]
  
  
  cont_raw_data <- matrix(NA, 0, 5)
  colnames(cont_raw_data) <- c("total_nodes", "num_groups", "p_in", "p_out", "gamma")
  
  for(i in 1:length(all_files)){
    temp <- read.csv(all_files[i])
    
    cont_raw_data <- rbind(cont_raw_data, temp)
  }
  
  
  
  # Initializing storage data
  plotting_data <- matrix(NA, length(num_groups)*length(total_nodes)*length(p_increase), 6)
  colnames(plotting_data) <- c("total_nodes", "cc", "p_increase", "p_out", "num_groups", "gamma")
  
  for(i in 0:I(length(total_nodes) - 1)){
    for(j in 0:I(length(p_increase)- 1)){
      for(k in 1:length(num_groups)){
        row <- i*length(p_increase)*length(num_groups) + j*length(num_groups) + k
        
        #print(c(row, total_nodes[i + 1], p_increase[j + 1], num_groups[k]))
        #print(row)
        
        # Filling in Data
        plotting_data[row, "total_nodes"] <- total_nodes[i + 1]
        plotting_data[row, "p_increase"] <- p_increase[j + 1]
        plotting_data[row, "p_out"] <- p_out
        
        if(num_groups[k] == "cc"){
          plotting_data[row, "cc"] <- 2
          plotting_data[row, "num_groups"] <- total_nodes[i + 1] # continuous data
          
          # Logistic Regression
          # temp_i <- cont_raw_data[which(cont_raw_data[, "total_nodes"] == total_nodes[i + 1] &
          #                                 cont_raw_data[, "p_in"] == round(p_out/total_nodes[i + 1] + p_out/total_nodes[i + 1]*p_increase[j + 1], 4) &
          #                                 cont_raw_data[, "p_out"] == p_out/total_nodes[i + 1] ), ]
          
          # MLE
          temp_i <- cont_raw_data[which(cont_raw_data[, "total_nodes"] == total_nodes[i + 1] &
                                          cont_raw_data[, "p_in"] == p_increase[j+1] &
                                          cont_raw_data[, "p_out"] == p_out), ] # average degree is p_out 
        }else{
          plotting_data[row, "cc"] <- 1
          plotting_data[row, "num_groups"] <- num_groups[k] # cat data
          
          n_wg <- as.numeric(num_groups[k])*choose(total_nodes[i+1]/as.numeric(num_groups[k]), 2)
          n_bg <- choose(total_nodes[i+1], 2) - n_wg
            
          p_out_ij <-(p_out/total_nodes[i + 1])*(n_wg + n_bg)/(n_wg*(1 + p_increase[j + 1]) + n_bg)
          
          temp_i <- cat_raw_data[which(cat_raw_data[, "total_nodes"] == total_nodes[i + 1] &
                                          cat_raw_data[, "p_in"] == p_increase[j+1] &
                                         cat_raw_data[, "p_out"] == p_out &
                                         cat_raw_data[, "num_groups"] == num_groups[k]), ]
        }
        
        # Taking mean gamma from simulations
        plotting_data[row, "gamma"] <- mean(temp_i[, "gamma"])
      }
    }
  }
  
  return(plotting_data)
}



cat_results <- Gamma_Summary(1)
cont_results <- Gamma_Summary(3)


full_data <- rbind(cat_results, cont_results)


final <- full_data[c(1, 2, 9, 3, 4, 10, 5, 6, 11, 7, 8, 12), ]
setwd("/Users/octavioustalbot/Desktop/PyTorch/Meeting")
write.csv(final, "Gamma_Table.csv")
#final <- full_data[c(5, 6, 11), ]



plot_data <- Plot_Params(total_nodes = 100,
                         p_increase = c(0, 0.5, 1.5),
                         p_out = 5,
                         num_groups = c(2, 5, "cc"))

setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Sim_Plotting_Data/Raw")
write.csv(plot_data, "plotting_data.csv")



# Line plots of gamma by percent increase separated by num_groups
line_plots <- function(final, group, min, max){
  
  final <- final[, -which(colnames(final) == "10%")]
  # group: index of total_nodes you want to plot
  
  total_nodes <- sort(unique(as.numeric(substr(rownames(final), 
                      1, 
                      nchar(rownames(final)) - 2))))
  
  # Extract numeric values from each entry in the matrix
  rows = I(3*(group - 1) + 1):I(group*3)
  values_matrix <- matrix(as.numeric(gsub("\\s.*", "", final)), 
                           nrow = nrow(final))[rows, ]
  
  std_matrix <- matrix(as.numeric(str_match(final, "\\((.*?)\\)")[, 2]),
                       nrow = nrow(final))[rows, ]
  
  numeric_matrix <- rbind(values_matrix, std_matrix)
  
  
  
  # Define the vector of percentages
  percentages <- colnames(final)
  
  # Remove the percentage sign and divide by 100
  decimals <- as.numeric(sub("%", "", percentages))/100
  
  
  # Convert matrix to data frame
  df <- data.frame(x = decimals, t(numeric_matrix))
  
  colnames(df) <- c("x", c("2 Groups", "5 Groups", "Continuous", "2_std", "5_std", "C_std"))  
  
  # Reshape data frame to long format
  df_long_values <- reshape2::melt(df[ ,c("x", "2 Groups", "5 Groups", "Continuous")], 
                                   id.vars = "x",
                                   value.name = "value",
                                   variable.name = "v1")
  df_long_std <- reshape2::melt(df[ ,c("x", "2_std", "5_std", "C_std")], 
                                id.vars = "x",
                                value.name = "std",
                                variable.name = "v2")
  
  df_long <- cbind(df_long_values, df_long_std[, c("v2", "std")])
  
  # Using standard error instead of standard deviation
  df_long[, "std"] <- df_long[, "std"]/sqrt(100)
  
  # Create line plot
  p <- ggplot(df_long, 
              aes(x = x, y = value, group = v1, 
                           color = v1, fill = factor(v1))) +
    geom_line() +
    labs(y = expression(gamma),
         x = "Homophilly",
         title = paste(letters[group], ")", sep = ""),
         color = "") +
    scale_x_continuous(breaks = decimals, labels = paste(decimals*100, "%", sep ="")) +
    ylim(c(min, max))  +
    scale_color_manual(values = c("#1f77b4", "#d62728", "#7f7f7f")) +
    geom_errorbar(aes(ymin = ifelse(value - 1.96*std < 0, 0, value - 1.96*std),
                      ymax = ifelse(value + 1.96*std > 1, 1, value + 1.96*std),
                      group = v1),
                  width = 0.1,
                  position = position_dodge(width = 0.2)) +
    theme_bw() +
    theme(axis.title = element_text(size = 7),
          plot.title = element_text(size = 7, hjust = 0),
          axis.text = element_text(size = 7, angle = 90, hjust = 1),
          legend.position = c(0.80, 0.90), 
          legend.box = "inside",
          legend.key.size = unit(0.20, "cm"),
          legend.margin = margin(0),
          legend.text = element_text(size = 7),
          legend.background = element_blank(),
          legend.box.background = element_blank())
    
  
  #plot(p)
}


final_line_plot <- function(final, min, max){
  plots <- list()
  
  for(i in 1:4){
    plots[[i]] <- line_plots(final = final, group = i, min = min, max = max)
  }
  
  #common_legend <- get_legend(plots[[1]] + theme(legend.position = "right"))
  #plot_title <- textGrob("% Increase vs Gamma", gp = gpar(fontsize = 15, fontface = "bold"))
  
  # combine the ggplots into a single panel
  setwd("/Users/octavioustalbot/Desktop/PyTorch/plots")
  
  ggsave("Gamma_Trend.pdf", 
         grid.arrange(plots[[1]] + theme(legend.position = "none", 
                                         axis.line.x = element_blank(),
                                         axis.text.x = element_blank(),
                                         axis.ticks.x = element_blank(),
                                         plot.title = element_text(hjust = 0)) + xlab(" "), 
                      plots[[2]] + theme(axis.line.x = element_blank(),
                                         axis.text.x = element_blank(),
                                         axis.ticks.x = element_blank(),
                                         axis.line.y = element_blank(),
                                         axis.text.y = element_blank(),
                                         axis.ticks.y = element_blank(),
                                         plot.title = element_text(hjust = 0)) + xlab(" ") + ylab(" "), 
                      plots[[3]] + theme(legend.position = "none",
                                         plot.title = element_text(hjust = 0)), 
                      plots[[4]] + theme(legend.position = "none", 
                                         axis.line.y = element_blank(),
                                         axis.text.y = element_blank(),
                                         axis.ticks.y = element_blank(),
                                         plot.title = element_text(hjust = 0))  + ylab(" "), 
                      ncol = 2,
                      #right = common_legend,
                      #top = plot_title,
                      widths = c(1, 1), 
                      heights = c(1, 1.2)), 
         width = 5.6, 
         height = 4.2, 
         units = "in")
}


final_line_plot(final, 0.20, 0.9)





ADD_plot_data <- function(){
  # Add Data (Finding Gamma)
  setwd("~/Desktop/PyTorch/R/Data/Add_Data/Raw")
  
  final <- matrix(NA, 2, 6)
  colnames(final) <- c("total_nodes", 
                       "cc",
                       "p_increase",
                       "p_out",
                       "num_groups",
                       "gamma")
    
  for(cc in c(4, 5)){
    all_files <- list.files()
    all_files <- grep(paste("CC_", cc, sep = ""), 
                      all_files,
                      value = TRUE)
    
    full <- NULL
    
    for(i in 1:length(all_files)){
      temp_data <- read.csv(all_files[i])
      
      full <- rbind(full,temp_data)
    }
    
    final[cc - 3, "total_nodes"] <- 290
    final[cc - 3, "cc"] <- cc
    final[cc - 3, c("p_increase",
               "p_out",
               "num_groups")] <- 0
    final[cc - 3, "gamma"] <- mean(full[, "gamma"])
  }
  
  setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Add_Data/Plotting_Data/Raw")
  write.csv(final, "plotting_data_AddH.csv", row.names = FALSE)
  
}

  
  
  