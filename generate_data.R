library(anndata)
library(Matrix)
library(SymSim)
phyla <- Phyla5()

generate_symsim_true_counts <- function(ncells, ngenes, phyla, 
                                        seed = 123, outputpath = "./") {
  ngroups <- length(phyla$tip.label)
  
  # Generate true counts
  true_counts_res <- SimulateTrueCounts(
    ncells_total = ncells,
    min_popsize = floor(ncells / ngroups),
    i_minpop = 2,
    ngenes = ngenes,
    nevf = 40,
    evf_type = "distinct",
    n_de_evf = 20,
    vary = "s",
    Sigma = 0.5,
    phyla = phyla,
    randseed = seed,
    gene_effects_sd = 1,
    gene_effect_prob = 0.7
  )
  
  # Save true counts
  true_count_data <- data.frame(t(true_counts_res$counts))
  labels <- true_counts_res$cell_meta$pop
  
  #write.csv(true_count_data, paste0(outputpath, "symsim_true_counts_", ngenes, "genes_", ncells, "cells.csv"))
  #write.csv(labels, paste0(outputpath, "symsim_labels_", ngenes, "genes_", ncells, "cells.csv"))
  #write.csv(seed, paste0(outputpath, "symsim_seed_", ngenes, "genes_", ncells, "cells.csv"))
  
  ann_true <- AnnData(
    X = Matrix(t(true_counts_res$counts), sparse = TRUE),
    obs = data.frame(group = labels,
                     row.names = rownames(true_count_data)),
    var = data.frame(col.names = colnames(true_count_data))
  )
  write_h5ad(ann_true, paste0(outputpath, "symsim_observed_counts_",
                              ngenes, "genes_", ncells, "cells_0.h5ad"))
  
  return(list(true_counts_res = true_counts_res, labels = labels))
}

generate_symsim_observed_counts <- function(true_counts_res, labels,
                                            ngenes, ncells,
                                            seed = 123,
                                            outputpath = "./", 
                                            alpha_mean,
                                            set) {
  # Estimate gene lengths
  data(gene_len_pool)
  gene_len <- sample(gene_len_pool, ngenes, replace = FALSE)
  
  # Simulate observed counts
  set.seed(seed + 1)
  observed_counts_res <- True2ObservedCounts(
    true_counts_res$counts,
    true_counts_res$cell_meta,
    protocol = "UMI",
    gene_len = gene_len,
    depth_mean = 1e5,
    depth_sd = 3e3,
    alpha_mean
  )
  
  # Save observed counts
  observed_count_data <- data.frame(t(observed_counts_res$counts))
  sparsity <- sum(observed_counts_res$counts == 0) / length(observed_counts_res$counts)
  
  #write.csv(observed_count_data, paste0(outputpath, "symsim_observed_counts_",sparsity,"sparsity_", ngenes, "genes_", ncells, "cells.csv"))
  
  ann_observed <- AnnData(
    X = Matrix(t(observed_counts_res$counts), sparse = TRUE),
    obs = data.frame(group = labels,
                     row.names = rownames(observed_count_data)),
    var = data.frame(col.names = colnames(observed_count_data))
  )
  write_h5ad(ann_observed, paste0(outputpath, "symsim_observed_counts_", ngenes, "genes_", ncells, "cells_", set, ".h5ad"))
  
  return(observed_counts_res)
}



generate_symsim_datasets_with_sparsity <- function(n_datasets,
                                                   ncells,
                                                   ngenes,
                                                   phyla,
                                                   seed = 123,
                                                   outputpath = "./symsim_runs/") {
  dir.create(outputpath, showWarnings = FALSE, recursive = TRUE)
  
  # Step 1: Generate true counts once
  true_data <- generate_symsim_true_counts(
    ncells = ncells,
    ngenes = ngenes,
    phyla = phyla,
    seed = seed,
    outputpath = outputpath
  )
  
  # Step 2: Generate observed counts with varying alpha_mean (capture rate)
  alpha_vals <- c(0.1562, 0.0776, 0.0475, 0.0321, 0.0229, 0.0160, 0.0122, 0.0084, 0.0046, 0.0008)
  
  sparsities <- c()
  
  for (i in seq_len(n_datasets)) {
    alpha_mean <- alpha_vals[i]
    
    observed_res <- generate_symsim_observed_counts(
      true_counts_res = true_data$true_counts_res,
      labels = true_data$labels,
      ngenes = ngenes,
      ncells = ncells,
      seed = seed + i,  # different seed for variability in capture noise
      outputpath = outputpath,  # save directly in the main outputpath
      alpha_mean = alpha_mean,
      set = i  # pass set number
    )
    
    sparsity <- sum(observed_res$counts == 0) / length(observed_res$counts)
    sparsities[i] <- sparsity
    
    message(sprintf("Dataset %d: alpha_mean = %.4f → sparsity = %.4f", i, alpha_mean, sparsity))
  }
  
  return(data.frame(dataset = 1:n_datasets, alpha_mean = alpha_vals, sparsity = sparsities))
}

sparsity_summary <- generate_symsim_datasets_with_sparsity(
  n_datasets = 10,
  ncells = 5000,
  ngenes = 20000,
  phyla = phyla,
  seed = 123,
  outputpath = "/zhome/54/2/187738/Desktop/data/20k_genes"
)
