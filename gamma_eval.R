####################################################################
# This script calcuates the gamma eval plots where we see the visuals
# as a function of gamma. 
####################################################################

Gamma_Eval <- function(d_file_direc, r_file_direc){
  # d_file_path: Directory that contains the varying simulated data considering different gammas
  # r_file_direc: Directory to store the results
  
  # Identifying data
  setwd(d_file_direc)
  all_files <- list.files()
  
  
  # Removing prefixes
  iso_files <- unique(sub("Edge_List_|Nodal_Covariates_|Nodal_Positions_", "", all_files))
  
  all_plots <- list()
  for(i in 1:length(iso_files)){
    graph_files <- grep(iso_files[i], all_files, value = TRUE)
    
    # Reading in Data
    v1 <- read.csv(grep("Nodal_Cova", graph_files, value = TRUE), row.names = NULL, header = FALSE)
    
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
    
    cc <- as.numeric(str_extract(iso_files[i], "(?<=CC_)\\d+\\.?\\d*"))
    
    NG <- str_extract(iso_files[i], "(?<=NG_)\\d+\\.?\\d*")
    
    # Constructing Network
    G <- make_empty_graph(n = dim(v1)[1],
                          directed = FALSE)
    V(G)$name <- as.character(v1[, 1]) # changing vertex names to that from the original graph
    G <- add_edges(G, t(e1))
    
    title <- bquote(.(paste0(letters[i], ")")) ~ gamma == .(gamma))
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
    if(gamma == 0.8){
      color = "red"
    }else{
      color = "black"
    }
    
    if(cc == 1){
      all_plots[[i]] <- ggraph(G, layout = p1) +
        geom_edge_link(alpha = 0.2) +
        geom_node_point(size = 2, color = palette) +
        theme(panel.background = element_rect(color = "black", fill = "white"),
              plot.title = element_text(hjust = 0.0, size = 8, color = color)) +
        labs(title = title)
      
    }else{
      if(cc == 2 | cc == 3){
        all_plots[[i]] <- ggraph(G, layout = p1) +
          geom_edge_link(alpha = 0.2) +
          geom_node_point(size = 2, color = palette(length(v1[, 2]))[rank(v1[, 2])]) +
          theme(panel.background = element_rect(color = "black", fill = "white"),
                plot.title = element_text(hjust = 0.0, size = 8, color = color)) +
          labs(title = title)
        
      }
    }
  }
  
  setwd(r_file_direc)
  my_plot <- grid.arrange(grobs = all_plots, nrow = 3, ncol = 4)
  
  ggsave("Gamma_Eval_Select.pdf", 
         my_plot, 
         width = 5, 
         height = 5, 
         units = "in")
}
