import os
import scanpy as sc
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score,homogeneity_score, completeness_score, v_measure_score
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from datetime import datetime
import random

print("Start time:", datetime.now())

# ------------------------------------------------------------------------------
# 1. CONFIG --------------------------------------------------------------------
# ------------------------------------------------------------------------------

sets            = list(range(11))                       # 0 … 10
data_dir        = "/zhome/54/2/187738/Desktop/data/20k_genes"
output_folder   = "/zhome/54/2/187738/Desktop/outputs3/normal/normalised/20k_genes"
os.makedirs(output_folder, exist_ok=True)

distance_metrics = [
    "euclidean",
    "manhattan",
    "cosine",
    "minkowski_p0.5",
    "minkowski_p0.7",
    "minkowski_p4",
]

# ------------------------------------------------------------------------------
# 2. LOAD DATASETS -------------------------------------------------------------
# ------------------------------------------------------------------------------

datasets = [
    sc.read_h5ad(
        f"{data_dir}/20k_genessymsim_observed_counts_20000genes_5000cells_{i}.h5ad"
    )
    for i in sets
]

# ------------------------------------------------------------------------------
# 3. ANALYSIS ------------------------------------------------------------------
# ------------------------------------------------------------------------------

results         = []   
cluster_counts  = []   

for i, adata in enumerate(datasets):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata)
    # make sure X is dense ndarray (needed for several sklearn metrics)
    adata.X = np.asarray(adata.X) if isinstance(adata.X, np.ndarray) else np.asarray(adata.X.toarray())
    true_labels = adata.obs["group"]
    # sparsity label for filenames / plotting
    nonzeros       = np.count_nonzero(adata.X)
    total_entries  = np.prod(adata.X.shape)
    sparsity       = round(1.0 - nonzeros / total_entries, 3)
    sparsity_label = f"Sparsity_{sparsity:.3f}"

    for dist in distance_metrics:
        # ----- build neighbor graph --------------------------------------
        if dist.startswith("minkowski"):
            p = float(dist.split("_p")[-1])
            sc.pp.neighbors(adata, metric="minkowski", metric_kwds={"p": p}, use_rep='X')
        else:
            sc.pp.neighbors(adata, metric=dist, use_rep='X')
        # ----- ONE canonical Leiden run --------
        if i == 0:
            sc.tl.leiden(adata, random_state=random.randint(0, 1_000_000), resolution=0.4,flavor="igraph", n_iterations=2, key_added="leiden_stability")
            stability_labels = adata.obs["leiden_stability"]

        # ----- 3c) mean ARI over 10 Leiden runs ------------------------------
        def run_leiden(seed: int):
            ad = sc.tl.leiden(
                adata, 
                random_state=seed, 
                resolution=0.4,
                flavor="igraph",
                n_iterations=2, 
                copy=True)
            return {
                "ari": adjusted_rand_score(true_labels, ad.obs["leiden"]),
                "ari_stability": adjusted_rand_score(stability_labels, ad.obs["leiden"]),
                "homogeneity": homogeneity_score(true_labels, ad.obs["leiden"]),
                "completeness": completeness_score(true_labels, ad.obs["leiden"]),
                "v_measure": v_measure_score(true_labels, ad.obs["leiden"]),
                "num_clusters": int(ad.obs["leiden"].nunique()),
            }

        # Run in parallel
        metrics_list = Parallel(n_jobs=6)(
            delayed(run_leiden)(random.randint(0, 1_000_000)) for _ in range(20)
        )

        # Extract mean and SE for each metric
        ari_scores = [m["ari"] for m in metrics_list]
        mean_ari  = float(np.mean(ari_scores))
        se_ari    = float(np.std(ari_scores, ddof=1) / np.sqrt(len(ari_scores)))

        ari_stability_scores = [m["ari_stability"] for m in metrics_list]
        if i == 0: 
            mean_ari_stability = 1
            se_ari_stability = 0
        else:
            mean_ari_stability  = float(np.mean(ari_stability_scores))
            se_ari_stability    = float(np.std(ari_stability_scores, ddof=1) / np.sqrt(len(ari_stability_scores)))

        hom_scores = [m["homogeneity"] for m in metrics_list]
        comp_scores = [m["completeness"] for m in metrics_list]
        v_scores = [m["v_measure"] for m in metrics_list]

        mean_hom_scores = float(np.mean(hom_scores))
        mean_comp_scores = float(np.mean(comp_scores))
        mean_v_scores = float(np.mean(v_scores))

        se_hom = float(np.std(hom_scores, ddof=1) / np.sqrt(len(hom_scores)))
        se_comp = float(np.std(comp_scores, ddof=1) / np.sqrt(len(comp_scores)))
        se_v = float(np.std(v_scores, ddof=1) / np.sqrt(len(v_scores)))

        num_clusters_list = [m["num_clusters"] for m in metrics_list]

        results.append({
            "Dataset": sparsity_label,
            "Distance": dist,
            "Mean_ARI": mean_ari,
            "Std_Error": se_ari,
            "Mean_ARI_Stability": mean_ari_stability,
            "Std_Error_Stability": se_ari_stability,
            "Mean_Homogeneity": mean_hom_scores,
            "Std_Error_Homogenity": se_hom,
            "Mean_Completeness": mean_comp_scores,
            "Std_Error_Completeness": se_comp,
            "Mean_V_Score": mean_v_scores,
            "Std_Error_V_Score": se_v,
        })

        cluster_counts.append(
            {
                "Dataset": sparsity_label,
                "Distance": dist,
                "Num_Clusters_Mean": float(np.mean(num_clusters_list)),
                "Num_Clusters_Std_Error": float(np.std(num_clusters_list, ddof=1) / np.sqrt(len(num_clusters_list))),
            }
        )

        sc.tl.leiden(adata, random_state=11, resolution=0.4,
                     flavor="igraph", n_iterations=2,key_added="leiden_umap")

        sc.tl.umap(adata)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        sc.pl.umap(adata, color="group",  title="True labels",   show=False, ax=axes[0])
        sc.pl.umap(adata, color="leiden_umap", title="Leiden clusters", show=False, ax=axes[1])
        fig.suptitle(f"{sparsity_label} — {dist}", fontsize=14)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(
            os.path.join(output_folder, f"UMAP_{sparsity_label}_{dist}.png"),
            dpi=300,
        )
        plt.close(fig)
        

# ------------------------------------------------------------------------------
# 4. SAVE CSV RESULTS ----------------------------------------------------------
# ------------------------------------------------------------------------------

df_ari      = pd.DataFrame(results)
df_clusters = pd.DataFrame(cluster_counts)

df_ari.to_csv(os.path.join(output_folder, "clustering_results.csv"), index=False)
df_clusters.to_csv(os.path.join(output_folder, "cluster_counts.csv"), index=False)

# ------------------------------------------------------------------------------
# 5. ORDER DATASETS BY SPARSITY FOR PLOTS --------------------------------------
# ------------------------------------------------------------------------------

def order_by_sparsity(df, value_col):
    df["Sparsity_Float"] = df["Dataset"].str.extract(r"([\d.]+)").astype(float)
    df = df.sort_values("Sparsity_Float")
    df["Dataset"] = pd.Categorical(df["Dataset"],
                                   categories=df["Dataset"].unique(),
                                   ordered=True)
    return df.drop(columns="Sparsity_Float")

df_ari      = order_by_sparsity(df_ari,      "Mean_ARI")
df_clusters = order_by_sparsity(df_clusters, "Num_Clusters_Mean")

# ------------------------------------------------------------------------------
# 6. PLOT MEAN ARI -------------------------------------------------------------
# ------------------------------------------------------------------------------

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_ari,
    x="Dataset",
    y="Mean_ARI",
    hue="Distance",
    marker="o",
    err_style="bars",
    errorbar=None,
    linewidth=2.5
)
# manual error bars
for (_, row) in df_ari.iterrows():
    plt.errorbar(row["Dataset"], row["Mean_ARI"], yerr=row["Std_Error"],
                 fmt="none", capsize=4, color="black", alpha=0.7)

plt.title("Mean ARI by Dataset and Distance Metric")
plt.ylabel("Mean Adjusted Rand Index (ARI)")
plt.xlabel("Dataset (increasing sparsity)")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title="Distance Metric", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # make space on the right

plt.savefig(os.path.join(output_folder, "mean_ari_plot.png"),
            bbox_inches="tight", dpi=300)
plt.close()

# ------------------------------------------------------------------------------
# 6-2. PLOT MEAN ARI Stability -------------------------------------------------
# ------------------------------------------------------------------------------

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_ari,
    x="Dataset",
    y="Mean_ARI_Stability",
    hue="Distance",
    marker="o",
    err_style="bars",
    errorbar=None,
    linewidth=2.5
)
# manual error bars
for (_, row) in df_ari.iterrows():
    plt.errorbar(row["Dataset"], row["Mean_ARI_Stability"], yerr=row["Std_Error_Stability"],
                 fmt="none", capsize=4, color="black", alpha=0.7)

plt.title("Mean ARI by Dataset and Distance Metric (Stability) ")
plt.ylabel("Mean Adjusted Rand Index (ARI)")
plt.xlabel("Dataset (increasing sparsity)")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title="Distance Metric", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # make space on the right

plt.savefig(os.path.join(output_folder, "mean_ari_plot_stability.png"),
            bbox_inches="tight", dpi=300)
plt.close()
# ------------------------------------------------------------------------------
# 7. PLOT NUMBER OF CLUSTERS ---------------------------------------------------
# ------------------------------------------------------------------------------

plt.figure(figsize=(12, 6))
sns.lineplot(
    data=df_clusters,
    x="Dataset",
    y="Num_Clusters_Mean",
    hue="Distance",
    marker="o",
    linewidth=2.5,
    errorbar=None  # We'll add manual error bars below
)

# manual error bars
for (_, row) in df_clusters.iterrows():
    plt.errorbar(
        row["Dataset"], 
        row["Num_Clusters_Mean"], 
        yerr=row["Num_Clusters_Std_Error"],
        fmt="none", capsize=4, color="black", alpha=0.7
    )

plt.title("Number of Leiden Clusters by Dataset and Distance Metric")
plt.ylabel("Number of Clusters (mean ± SE)")
plt.xlabel("Dataset (increasing sparsity)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title="Distance Metric", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(os.path.join(output_folder, "num_clusters_plot.png"),
            bbox_inches="tight", dpi=300)
plt.close()

# ------------------------------------------------------------------------------  
# 8. COMBINED PLOT: HOMOGENEITY & COMPLETENESS --------------------------------  
# ------------------------------------------------------------------------------  

fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, sharey=True)

sns.lineplot(
    data=df_ari,
    x="Dataset",
    y="Mean_Homogeneity",
    hue="Distance",
    marker="o",
    linewidth=2.5,
    ax=axes[0],
    err_style="bars",
    errorbar=None,
    legend=False
)
# Add manual error bars for Homogeneity
for _, row in df_ari.iterrows():
    axes[0].errorbar(
        row["Dataset"],
        row["Mean_Homogeneity"],
        yerr=row["Std_Error_Homogenity"],
        fmt="none",
        capsize=4,
        color="black",
        alpha=0.7
    )
axes[0].set_title("Mean Homogeneity by Dataset and Distance Metric")
axes[0].set_ylabel("Score")
axes[0].set_xlabel("Dataset (increasing sparsity)")
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylim(0, 1)

sns.lineplot(
    data=df_ari,
    x="Dataset",
    y="Mean_Completeness",
    hue="Distance",
    marker="o",
    linewidth=2.5,
    ax=axes[1],
    err_style="bars",
    errorbar=None,
)
# Add manual error bars for Completeness
for _, row in df_ari.iterrows():
    axes[1].errorbar(
        row["Dataset"],
        row["Mean_Completeness"],
        yerr=row["Std_Error_Completeness"],
        fmt="none",
        capsize=4,
        color="black",
        alpha=0.7
    )
axes[1].set_title("Mean Completeness by Dataset and Distance Metric")
axes[1].set_ylabel("Score")
axes[1].set_xlabel("Dataset (increasing sparsity)")
axes[1].tick_params(axis='x', rotation=45)
axes[1].set_ylim(0, 1)
axes[1].legend(title="Distance Metric", bbox_to_anchor=(1.02, 1), loc="upper left")

plt.suptitle("Homogeneity and Completeness by Dataset and Distance Metric", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 0.85, 0.95])  # Adjust layout for title and legend

plt.savefig(os.path.join(output_folder, "homogeneity_completeness_combined_plot.png"),
            bbox_inches="tight", dpi=300)
plt.close()


# ------------------------------------------------------------------------------  
# 9. PLOT MEAN V-MEASURE -------------------------------------------------------  
# ------------------------------------------------------------------------------  

sns.set_theme(style="whitegrid")
plt.figure(figsize=(12, 6))

sns.lineplot(
    data=df_ari,
    x="Dataset",
    y="Mean_V_Score",
    hue="Distance",
    marker="o",
    linewidth=2.5,
    err_style="bars",
    errorbar=None,
)

# manual error bars for V-measure
for _, row in df_ari.iterrows():
    plt.errorbar(
        row["Dataset"],
        row["Mean_V_Score"],
        yerr=row["Std_Error_V_Score"],
        fmt="none",
        capsize=4,
        color="black",
        alpha=0.7
    )

plt.title("Mean V-Measure by Dataset and Distance Metric")
plt.ylabel("Mean V-Measure")
plt.xlabel("Dataset (increasing sparsity)")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.legend(title="Distance Metric", bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.85, 1])  # make space on the right

plt.savefig(os.path.join(output_folder, "mean_v_measure_plot.png"),
            bbox_inches="tight", dpi=300)
plt.close()
