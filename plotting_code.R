# Loading libraries
library(dplyr)
library(igraph)
library(ggplot2)
library(ggraph)
library(gridExtra)
library(stringr)


Network_plots <- function(real_sim){
  if(real_sim == "sim"){
    # Identifying data
    setwd("~/Desktop/PyTorch/R/Data/Sim_Plotting_Data")
    all_files <- list.files()
    all_files <- all_files[-which(all_files == "Raw" | all_files == "Plots" | all_files == "Old")]
    
    
    # Removing prefixes
    iso_files <- unique(sub("Edge_List_|Nodal_Covariates_|Nodal_Positions_", "", all_files))
    iso_files <- iso_files[c(grep("NG_2", iso_files), 
                             grep("NG_5", iso_files),
                             grep("NG_100", iso_files))]
    
    
    all_plots <- list()
    for(i in 1:length(iso_files)){
      graph_files <- grep(iso_files[i], all_files, value = TRUE)
      
      # Reading in Data
      v1 <- read.csv(grep("Nodal_Cova", graph_files, value = TRUE), row.names = NULL, header = FALSE)
      #v1$Nodes <- 1:dim(v1)[1] 
      
      e1 <- read.csv(grep("Edge_Lis", graph_files, value = TRUE), header = FALSE)
      e1[, 1] <- as.character(e1[, 1])
      e1[, 2] <- as.character(e1[, 2])
      e1 <- as.matrix(e1)
      
      p1 <- read.csv(grep("Nodal_Posit", graph_files, value = TRUE), header = FALSE)
      p1 <- as.matrix(p1[, -1])
      
      gamma <- str_extract(iso_files[i], "(?<=Gamma_)\\d+\\.?\\d*")
      gamma <- round(as.numeric(gamma), 2)
      
      p_in <- paste(as.numeric(str_extract(iso_files[i], "(?<=PI_)\\d+\\.?\\d*"))*100, 
                    "%",
                    sep = "")
      
      cc <- str_extract(iso_files[i], "(?<=CC_)\\d+\\.?\\d*")
      
      NG <- str_extract(iso_files[i], "(?<=NG_)\\d+\\.?\\d*")
      
      # Constructing Network
      G <- make_empty_graph(n = dim(v1)[1],
                            directed = FALSE)
      V(G)$name <- as.character(v1[, 1]) # changing vertex names to that from the original graph
      G <- add_edges(G, t(e1))
      
      title <- bquote(.(paste0(letters[i], ")")) ~ gamma == .(gamma))
      FR <- layout_with_fr(G)
      # vertex.color needs a positive value
      #plot(G, layout = p1, vertex.color = v1[, "V1"] + max(abs(v1[, "V1"]))) 
      
      # Plotting Network
      if(NG == 2){
        palette <- ifelse(v1[, 2] == 0, "#1f77b4", 
                          ifelse(v1[, 2] == 1, "#d62728", NA))
      }else{
        if(NG == 5){
          palette <- ifelse(v1[, 2] == 0, "#1f77b4", 
                            ifelse(v1[, 2] == 1, "#d62728", 
                                   ifelse(v1[, 2] == 2, "#7f7f7f",
                                          ifelse(v1[, 2] == 3, "#9467bd",
                                                 ifelse(v1[, 2] == 4, "#ff7f0e",
                                                        NA)))))
        }else{# continuous 
          palette <- colorRampPalette(c("#1f77b4", "#d62728"))
        }
      }
      
      # Matching iteration
      j <- 3*floor((i-1)/3) + i
      if(cc == 1){
        all_plots[[j]] <- ggraph(G, layout = p1) +
          geom_edge_link(alpha = 0.2) +
          geom_node_point(size = 0.7, color = palette) +
          theme(panel.background = element_rect(color = "black", fill = "white"),
                plot.title = element_text(hjust = 0.0, size = 8)) +
          labs(title = title)
        
        all_plots[[j + 3]] <- ggraph(G, layout = FR) +
          geom_edge_link(alpha = 0.2) +
          geom_node_point(size = 0.7, color = palette) +
          theme(panel.background = element_rect(color = "black", fill = "white"),
                plot.title = element_text(hjust = 0.0, size = 8)) +
          #labs(title = title)
          labs(title = "")
      }else{
        if(cc == 2 | cc == 3){
          all_plots[[j]] <- ggraph(G, layout = p1) +
            geom_edge_link(alpha = 0.2) +
            geom_node_point(size = 0.7, color = palette(length(v1[, 2]))[rank(v1[, 2])]) +
            theme(panel.background = element_rect(color = "black", fill = "white"),
                  plot.title = element_text(hjust = 0.0, size = 8)) +
            labs(title = title)
          
          
          all_plots[[j + 3]] <- ggraph(G, layout = FR) +
            geom_edge_link(alpha = 0.2) +
            geom_node_point(size = 0.7, color = palette(length(v1[, 2]))[rank(v1[, 2])]) +
            theme(panel.background = element_rect(color = "black", fill = "white"),
                  plot.title = element_text(hjust = 0.0, size = 8)) +
            #labs(title = title)
            labs(title = "")
          
        }
      }
    }
    
    
    setwd("/Users/octavioustalbot/Desktop/PyTorch/plots")
    my_plot <- grid.arrange(grobs = all_plots, nrow = 6, ncol = 3)
    
    # Create row and column titles
    #col.titles = c("0%", "50%", "150%")
    #row.titles = c("2 Groups", "5 Groups", "Continuous")
    
    # Add row titles
    #all_plots[c(1, 4, 7)] = lapply(c(1, 2, 3), function(i) arrangeGrob(all_plots[[i + 2*(i-1)]], left=row.titles[i]))
    
    # Add column titles and lay out plots
    #my_plot <- grid.arrange(grobs=lapply(c(1,2, 3), function(i) {
    #  arrangeGrob(grobs=all_plots[c(i, i + 3, i + 6)], top=col.titles[i], ncol=1)
    #}), ncol=3)
    
    ggsave("Selection_Plots.pdf", 
           my_plot, 
           width = 5.6, 
           height = 7.2, 
           units = "in")
    
  }else{
    if(real_sim == "real"){
      # Identifying data
      setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Add_Data/Plotting_Data")
      all_files <- list.files()
      all_files <- all_files[-which(all_files %in% c("Plots", "Raw"))]
      
      # Removing prefixes
      iso_files <- unique(sub("Edge_List_|Nodal_Covariates_|Nodal_Positions_", "", all_files))
      
      
      for(i in 1:length(iso_files)){
        all_plots <- list()
        setwd("/Users/octavioustalbot/Desktop/PyTorch/R/Data/Add_Data/Plotting_Data")
        graph_files <- grep(iso_files[i], all_files, value = TRUE)
        
        # Reading in Data
        v1 <- read.csv(grep("Nodal_Cova", graph_files, value = TRUE), row.names = NULL, header = FALSE)
        #v1$Nodes <- 1:dim(v1)[1] 
        
        e1 <- read.csv(grep("Edge_Lis", graph_files, value = TRUE), header = FALSE)
        e1[, 1] <- as.character(e1[, 1])
        e1[, 2] <- as.character(e1[, 2])
        e1 <- as.matrix(e1)
        
        p1 <- read.csv(grep("Nodal_Posit", graph_files, value = TRUE), header = FALSE)
        p1 <- as.matrix(p1[, -1])
        
        gamma <- str_extract(iso_files[i], "(?<=Gamma_)\\d+\\.?\\d*")
        
        p_in <- paste(as.numeric(str_extract(iso_files[i], "(?<=PI_)\\d+\\.?\\d*"))*100, 
                      "%",
                      sep = "")
        
        cc <- str_extract(iso_files[i], "(?<=CC_)\\d+\\.?\\d*")
        
        NG <- str_extract(iso_files[i], "(?<=NG_)\\d+\\.?\\d*")
        
        # Constructing Network
        G <- make_empty_graph(n = dim(v1)[1],
                              directed = FALSE)
        V(G)$name <- as.character(v1[, 1]) # changing vertex names to that from the original graph
        G <- add_edges(G, t(e1))
        
        # FR Layout 
        FR <- layout_with_fr(G)
        
        # vertex.color needs a positive value
        # plot(G, layout = p1, vertex.color = v1[, "V4"] + max(abs(v1[, "V4"]))) 
        
        if(cc == 4){
          covs <- c("Sex", "Race", "Grade", "School")
          for(k in 1:4){
            # Plotting Network
            palette <- ifelse(v1[, k+1] == 0, "#1f77b4", 
                              ifelse(v1[, k+1] == 1, "#d62728", 
                                     ifelse(v1[, k+1] == 2, "#7f7f7f",
                                            ifelse(v1[, k+1] == 3, "#9467bd",
                                                   ifelse(v1[, k+1] == 4, "#ff7f0e",
                                                          ifelse(v1[, k+1] == 5, "#8c564b",
                                                                 NA))))))
            # US
            all_plots[[k]] <- ggraph(G, layout = p1) +
              geom_edge_link(alpha = 0.2) +
              geom_node_point(size = 0.7, color = palette) +
              theme(panel.background = element_rect(color = "black", fill = "white"),
                    plot.title = element_text(hjust = 0, size = 8)) +
              labs(title = paste(paste(letters[k], ")", sep = "")))
            
            # FR
            all_plots[[k + 4]] <- ggraph(G, layout = FR) +
              geom_edge_link(alpha = 0.2) +
              geom_node_point(size = 0.7, color = palette) +
              theme(panel.background = element_rect(color = "black", fill = "white"),
                    plot.title = element_text(hjust = 0, size = 8)) +
              #labs(title = paste(paste(letters[k + 4], ")", sep = "")))
              labs(title = "")
          }
          
          setwd("/Users/octavioustalbot/Desktop/PyTorch/plots")
          my_plot <- grid.arrange(grobs = all_plots, nrow = 2, ncol = 4)
          ggsave(filename = "Full_AddH_Selection_Plots.pdf", 
                 my_plot, 
                 width = 5.6, 
                 height = 2.7, 
                 units = "in")
        }else{
          if(cc == 5){
            covs <- c("Grade")
            for(k in 3){ # Grade covariate
              # Plotting Network
              palette <- ifelse(v1[, k+1] == 0, "#1f77b4", 
                                ifelse(v1[, k+1] == 1, "#d62728", 
                                       ifelse(v1[, k+1] == 2, "#7f7f7f",
                                              ifelse(v1[, k+1] == 3, "#9467bd",
                                                     ifelse(v1[, k+1] == 4, "#ff7f0e",
                                                            ifelse(v1[, k+1] == 5, "#8c564b",
                                                                   NA))))))
              # US
              all_plots[[1]] <- ggraph(G, layout = p1) +
                geom_edge_link(alpha = 0.2) +
                geom_node_point(size = 0.7, color = palette) +
                theme(panel.background = element_rect(color = "black", fill = "white"),
                      plot.title = element_text(hjust = 0, size = 8)) +
                labs(title = paste(paste(letters[1], ")", sep = "")))
              
              # FR
              all_plots[[2]] <- ggraph(G, layout = FR) +
                geom_edge_link(alpha = 0.2) +
                geom_node_point(size = 0.7, color = palette) +
                theme(panel.background = element_rect(color = "black", fill = "white"),
                      plot.title = element_text(hjust = 0, size = 8)) +
                labs(title = paste(paste(letters[2], ")", sep = "")))
                labs(title = "")
            }
            
            setwd("/Users/octavioustalbot/Desktop/PyTorch/plots")
            my_plot <- grid.arrange(grobs = all_plots, nrow = 1, ncol = 2)
            ggsave(filename = "Grade_AddH_Selection_Plots.pdf", 
                   my_plot, 
                   width = 3.0, 
                   height = 1.5, 
                   units = "in")
          }else{
            if(cc == 6){
              covs <- c("Sex")
              for(k in 1){ # Grade covariate
                # Plotting Network
                palette <- ifelse(v1[, k+1] == 0, "#1f77b4", 
                                  ifelse(v1[, k+1] == 1, "#d62728", 
                                         ifelse(v1[, k+1] == 2, "#7f7f7f",
                                                ifelse(v1[, k+1] == 3, "#9467bd",
                                                       ifelse(v1[, k+1] == 4, "#ff7f0e",
                                                              ifelse(v1[, k+1] == 5, "#8c564b",
                                                                     NA))))))
                # US
                all_plots[[1]] <- ggraph(G, layout = p1) +
                  geom_edge_link(alpha = 0.2) +
                  geom_node_point(size = 0.7, color = palette) +
                  theme(panel.background = element_rect(color = "black", fill = "white"),
                        plot.title = element_text(hjust = 0, size = 8)) +
                  labs(title = paste(paste(letters[1], ")", sep = "")))
                
                # FR
                all_plots[[2]] <- ggraph(G, layout = FR) +
                  geom_edge_link(alpha = 0.2) +
                  geom_node_point(size = 0.7, color = palette) +
                  theme(panel.background = element_rect(color = "black", fill = "white"),
                        plot.title = element_text(hjust = 0, size = 8)) +
                  labs(title = paste(paste(letters[2], ")", sep = "")))
                labs(title = "")
              }
              
              setwd("/Users/octavioustalbot/Desktop/PyTorch/plots")
              my_plot <- grid.arrange(grobs = all_plots, nrow = 1, ncol = 2)
              ggsave(filename = "Sex_AddH_Selection_Plots.pdf", 
                     my_plot, 
                     width = 3.0, 
                     height = 1.5, 
                     units = "in")
            }
          }
        }
      }
    }
  }
}


Network_plots(real_sim = "sim")


Network_plots(real_sim = "real")

