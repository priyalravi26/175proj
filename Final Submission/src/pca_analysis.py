import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from adjustText import adjust_text
from sklearn.ensemble import RandomForestRegressor

def pca_biplot_pipeline(Y_matrix, X_final, drug_metadata, show_plot=True,save_dir=None):
    """
    Full PCA biplot pipeline:
    - PCA on drug response matrix
    - PCA on target features
    - Drug + target visualization (biplot)
    - Returns objects for integration testing 
    """

    # PCA on drug response matrix
    scaler = StandardScaler()
    Yscaled = scaler.fit_transform(Y_matrix)

    pca = PCA(n_components=10)
    drug_scores = pca.fit_transform(Yscaled)

    explained_variance = pca.explained_variance_ratio_
    print(f"Explained variance ratio of first 3 PCs: {explained_variance[:3]}")

    # Biplot

    drug_df = drug_df = pd.DataFrame(
        drug_scores,  
        columns=[f"PC{i+1}" for i in range(drug_scores.shape[1])],
        index=Y_matrix.index
    )

    # Drug metadata alignment
    drug_metadata = drug_metadata.copy()
    drug_metadata["column_name"] = drug_metadata["column_name"].astype(str).str.strip()

    cat_map = drug_metadata.set_index("column_name")["drug_category"].to_dict()

    drug_df["Category"] = [
        cat_map.get(d, "Unknown") for d in drug_df.index
    ]

    colors = {
        "targeted cancer": "#E23C4A",
        "noncancer": "#748CAA",
        "chemo": "#2ca02c"
    }

    # PCA on target features
    X_scaled = StandardScaler().fit_transform(X_final)

    pca_targets = PCA(n_components=10)
    pca_targets.fit(X_scaled)

    target_loadings = pd.DataFrame(
        pca_targets.components_.T[:, :2],
        index=X_final.columns,
        columns=["PC1", "PC2"]
    ).dropna()

    target_loadings["magnitude"] = np.sqrt(
        target_loadings["PC1"]**2 + target_loadings["PC2"]**2
    )

    top_targets = target_loadings.sort_values(
        "magnitude", ascending=False
    ).head(5)

    # Plotting
    fig, ax = plt.subplots(figsize=(10, 7))

    # Scatter plot for drugs
    for cat, group in drug_df.groupby("Category"):
        ax.scatter(
            group["PC1"],
            group["PC2"],
            label=cat,
            color=colors.get(cat, "#7f7f7f"),
            alpha=0.6,
            s=20
        )

    texts = []
    arrow_scale = 40

    # Arrows for targets
    for target in top_targets.index:
        x = top_targets.loc[target, "PC1"] * arrow_scale
        y = top_targets.loc[target, "PC2"] * arrow_scale

        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                alpha=0.5,
                shrinkA=0,
                shrinkB=0
            )
        )

        txt = ax.text(
            x * 1.15,
            y * 1.15,
            target,
            fontsize=9,
            color="black",
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none")
        )

        texts.append(txt)

        ax.plot([x, x * 1.15], [y, y * 1.15],
                linestyle="dotted", color="black", alpha=0.5)

    adjust_text(
        texts,
        ax=ax,
        only_move={'texts': 'xy'},
        expand_text=(1.2, 1.4),
        expand_points=(1.2, 1.4),
        force_text=(0.5, 0.8),
        force_points=(0.3, 0.5),
    )

    # Axes styling
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("PCA Biplot: Drug Response + Target Structure")
    ax.legend()

    plt.tight_layout()

    if save_dir:
        plt.savefig(os.path.join(save_dir, "pca_biplot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    
    # Return values
    return {
        "drug_df": drug_df,
        "explained_variance": explained_variance,
        "top_targets": top_targets,
        "pca_drugs": pca,
        "pca_targets": pca_targets,
        "figure": fig
    }


# --- PCA Plots Using 5 PCs ---#
def plot_multiple_pca_views(drug_df, components=[(1,2), (2,3), (3,4), (4,5), (1,3)], save_dir=None):
    """
    Full PCA pipeline:
    - Graph 5 PCA plots with the selected components to see variation differences
    """
    # validate available PCs
    available_pcs = [col for col in drug_df.columns if col.startswith("PC")]

    filtered_components = []
    for i, j in components:
        pc_i, pc_j = f"PC{i}", f"PC{j}"

        if pc_i in available_pcs and pc_j in available_pcs:
            filtered_components.append((i, j))
        else:
            print(f"Warning: skipping {pc_i} vs {pc_j}")

    if not filtered_components:
        print("No valid PCA component pairs found.")
        return

    # Grid layout of 3x2
    n_plots = len(filtered_components)
    ncols = 2
    nrows = (n_plots + ncols - 1) // ncols

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows)
    )

    axes = axes.flatten()

    #Color map 
    colors = {
        "targeted cancer": "#E23C4A",
        "noncancer": "#748CAA",
        "chemo": "#2ca02c",
        "Unknown": "#999999"
    }

    #Plotting
    for ax, (i, j) in zip(axes, filtered_components):

        for cat, group in drug_df.groupby("Category"):
            ax.scatter(
                group[f"PC{i}"],
                group[f"PC{j}"],
                label=cat,
                color=colors.get(cat, "#7f7f7f"),
                s=15,
                alpha=0.6
            )

        ax.set_title(f"PC{i} vs PC{j}")
        ax.set_xlabel(f"PC{i}")
        ax.set_ylabel(f"PC{j}")

        ax.axhline(0, color="gray", lw=0.5)
        ax.axvline(0, color="gray", lw=0.5)

    #Hide unused axes
    for k in range(len(filtered_components), len(axes)):
        axes[k].axis("off")

    #Legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right")

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "pca_multiple_views.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()