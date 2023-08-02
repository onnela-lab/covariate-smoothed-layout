########################################################################################
# This script organizes the procretus results
########################################################################################
library(stringr)
library(ggplot2)
library(reshape2)
library(dplyr)
library(igraph)
library(ggraph)
library(gridExtra)


Procrustes_plots <- function(TN, NG){
  # This function creates the Procrustes plots

  # TN: total nodes
  # NG: Number of gropus
  
  setwd("~/Desktop/PyTorch/R/Data/Missing_Plot_Data/Raw/Procrustes")
  
  all_files <- list.files()
  all_files <- all_files[-which(all_files == "Gamma_0.85")]
  
  full_data <- NULL
  for(i in 1:length(all_files)){
    print(i/length(all_files))
    
    raw_data <- read.csv(all_files[i])
    
    temp_data <- melt(raw_data, 
                      id.vars = "Missingness",
                      value.name = "Procrustes")
    
    added_info <- strsplit(gsub(".csv", "" ,all_files[i]), "_")[[1]]
    
    temp_data$col <- i
    temp_data$IT <- added_info[which(added_info == "IT") + 1]
    temp_data$TN <- added_info[which(added_info == "TN") + 1]
    temp_data$NG <- added_info[which(added_info == "NG") + 1]
    temp_data$PI <- added_info[which(added_info == "PI") + 1]
    temp_data$PO <- added_info[which(added_info == "PO") + 1]
    temp_data$CC <- added_info[which(added_info == "CC") + 1]
    
    full_data <- rbind(full_data, temp_data)
  }
  
  
  full_data$PI <- factor(paste0(as.numeric(full_data$PI)*100, "%"),
                            levels = paste0(sort(unique(as.numeric(full_data$PI)*100)), "%"))
  
  # Calculate the mean values for each x-value
  # Gamma = 0.85
  
  full_data_avg <- full_data[which(full_data$TN == TN &
                                     full_data$NG == NG &
                                     full_data$IT %in% 1:100 &
                                     !full_data$PI %in% c("25%", "75%", "100%")
                                   ),] %>% 
    group_by(NG, Missingness, variable, PI) %>% 
    mutate(mean_P = mean(Procrustes))
  
  # Custom labeling function
  custom_labels <- function(x) {
    labels <- paste0(x*100, "%")
    return(labels)
  }
  
  my_plot <- ggplot(full_data_avg, 
                    aes(y = Procrustes, x = Missingness, group = IT)) + 
    geom_line(color = "lightgrey") +
    facet_grid(PI ~ variable) +
    #scale_color_gradient(low = "blue", high = "red") +
    guides(color = 'none') + 
    geom_line(data = full_data_avg, aes(Missingness, mean_P), color = "black") +
    theme_classic() + 
    theme(panel.background = element_rect(fill = "white"),
          plot.title = element_text(hjust = 0.5, size = 8),
          axis.title = element_text(size = 8),
          axis.text = element_text(size = 8),
          axis.text.x = element_text(angle = 90, hjust = 1),
          legend.text = element_text(size = 8),
          strip.text = element_blank()
    ) +
    scale_x_continuous(labels = custom_labels) +
    geom_text(aes(label = paste0(letters[as.numeric(PI) + as.numeric(variable) - 1], ")"), 
                                       x = 0.8, 
                                       y = 0.6,
                                       group = variable),
              hjust = 0, vjust = 1, color = "black", size = 2)  
  
  setwd("/Users/octavioustalbot/Desktop/PyTorch/plots")
  ggsave(filename = "Procrustes.pdf", 
         my_plot, 
         width = 5.2, 
         height = 3.5, 
         units = "in")
}
Procrustes_plots(TN = 100, NG = 100)




Missing_visuals <- function(TN, NG, PI){
  # Creates the missing robustness plots
  
  if(PI == 1){
    PI = "1.0"
  }else{
    if(PI == 0){
      PI = "0.0"
    }
  }
  setwd("~/Desktop/PyTorch/R/Data/Missing_Plot_Data/Raw/Visuals")
  all_files <- list.files()
  all_files <- all_files[-which(all_files == "Gamma_0.85")]
  
  iso_files <- unique(sub("Edge_List_|Nodal_Covariates_|Nodal_Positions_", "", all_files))
  iso_files <- unique(sub("US_|FR_", "", iso_files))
  iso_files <- unique(sub("_Missing.*", "", iso_files))
  iso_files <- unique(gsub("_Gamma_[0-9.]+", "", iso_files))
  
  target_file <- grep(paste("TN_", TN, "_NG_", NG, "_PI_", PI, "_",  sep = ""), 
                      iso_files,
                      value = TRUE)
  
  target_files <- grep(target_file, all_files, value = TRUE)
  
  missing_rates <- sort(unique(sub(".*Missing_([0-9.]+)\\.csv*", "\\1", target_files)))
  
  US_all_plots <- list()
  FR_all_plots <- list()
  for(i in 1:length(missing_rates)){
    graph_files <- grep(paste0(missing_rates[i], "\\."), target_files, value = TRUE)
    
    # Reading in Data
    v1 <- read.csv(grep("Nodal_Cova", graph_files, value = TRUE), row.names = NULL, header = FALSE)
    #v1$Nodes <- 1:dim(v1)[1] 
    
    e1 <- read.csv(grep("Edge_Lis", graph_files, value = TRUE), header = FALSE)
    e1[, 1] <- as.character(e1[, 1])
    e1[, 2] <- as.character(e1[, 2])
    e1 <- as.matrix(e1)
    
    FR_p1 <- read.csv(grep("FR_Nodal_Posit", graph_files, value = TRUE), header = FALSE)
    FR_p1 <- as.matrix(FR_p1[, -1])
    
    US_p1 <- read.csv(grep("US_Nodal_Posit", graph_files, value = TRUE), header = FALSE)
    US_p1 <- as.matrix(US_p1[, -1])
    
    gamma <- str_extract(iso_files[i], "(?<=Gamma_)\\d+\\.?\\d*")
    
    miss <- paste(as.numeric(missing_rates[i])*100, 
                  "%",
                  sep = "")
    
    cc <- str_extract(iso_files[i], "(?<=CC_)\\d+\\.?\\d*")
    
    
    # Constructing Network
    G <- make_empty_graph(n = dim(v1)[1],
                          directed = FALSE)
    V(G)$name <- as.character(v1[, 1]) # changing vertex names to that from the original graph
    G <- add_edges(G, t(e1))
    
    # Plotting Network
    palette <- colorRampPalette(c("#1f77b4", "#d62728"))
    if(cc == 1){
      US_all_plots[[i]] <- ggraph(G, layout = US_p1) +
        geom_edge_link(alpha = 0.2) +
        geom_node_point(size = 0.7, color = palette(length(v1[, 2]))[rank(v1[, 2])]) +
        theme(panel.background = element_rect(color = "black", fill = "white"),
              plot.title = element_text(hjust = 0, size = 8),
              axis.title = element_text(size = 8),
              axis.text = element_text(size = 8),
              axis.text.x = element_blank(),
              axis.text.y = element_blank(),
              legend.text = element_text(size = 8)) +
        labs(title = paste(letters[i], ")", sep = "")) +
        ylab("") +
        xlab("") 
        
      
      FR_all_plots[[i]] <- ggraph(G, layout = FR_p1) +
        geom_edge_link(alpha = 0.2) +
        geom_node_point(size = 0.7, color = palette(length(v1[, 2]))[rank(v1[, 2])]) +
        theme(panel.background = element_rect(color = "black", fill = "white"),
              plot.title = element_text(hjust = 0, size = 8),
              axis.title = element_text(size = 8),
              axis.text = element_text(size = 8),
              axis.text.x = element_blank(),
              axis.text.y = element_blank(),
              legend.text = element_text(size = 8)) +
        #labs(title = miss) +
        ylab("") +
        xlab("") 
      
    }else{
      if(cc == 2 | cc == 3){
        US_all_plots[[i]] <- ggraph(G, layout = US_p1) +
          geom_edge_link(alpha = 0.2) +
          geom_node_point(size = 0.7, color = palette(length(v1[, 2]))[rank(v1[, 2])]) +
          theme(panel.background = element_rect(color = "black", fill = "white"),
                plot.title = element_text(hjust = 0, size = 8),
                axis.title = element_text(size = 8),
                axis.text = element_text(size = 8),
                axis.text.x = element_blank(),
                axis.text.y = element_blank(),
                legend.text = element_text(size = 8)) +
          labs(title = paste(letters[i], ")", sep = "")) +
          ylab("") +
          xlab("") 
        
        FR_all_plots[[i]] <- ggraph(G, layout = FR_p1) +
          geom_edge_link(alpha = 0.2) +
          geom_node_point(size = 0.7, color = palette(length(v1[, 2]))[rank(v1[, 2])]) +
          theme(panel.background = element_rect(color = "black", fill = "white"),
                plot.title = element_text(hjust = 0, size = 8),
                axis.title = element_text(size = 8),
                axis.text = element_text(size = 8),
                axis.text.x = element_blank(),
                axis.text.y = element_blank(),
                legend.text = element_text(size = 8)) +
        #  labs(title = miss) +
          ylab("") +
          xlab("") 

      }
    }
    
    
  }
  
  vert_US_all_plots <- list()
  
  for(i in 1:length(US_all_plots)){
    vert_US_all_plots[[2*(i-1) + 1]] <- US_all_plots[[i]] 
    vert_US_all_plots[[2*i]] <- FR_all_plots[[i]] 
  }
  
  my_plot <- grid.arrange(grobs = c(US_all_plots, FR_all_plots), nrow = 2, ncol = 5)
  print(my_plot)
  
  setwd("/Users/octavioustalbot/Desktop/PyTorch/plots")
  ggsave(filename = "Missing_Vis.pdf", 
         my_plot, 
         width = 5.2, 
         height = 2.5, 
         units = "in")
}


Missing_visuals(TN = 100, NG = 100, PI = 1.5)

