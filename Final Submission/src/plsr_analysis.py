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
#--- Components Selection for PLSR ---#
"""
PLSR Components Selection Pipeline

Performs cross-validated model selection for Partial Least Squares Regression (PLSR)
by evaluating model performance across a range of latent components.

For each candidate number of components, the pipeline:
- Fits a PLSRegression model using K-fold cross-validation
- Computes CV Mean Squared Error (MSE)
- Computes Q² (predictive power metric based on PRESS vs TSS)
- Normalizes and combines metrics to select optimal complexity

The optimal number of components is chosen by minimizing a combined score
that balances prediction error and explained variance.

Parameters
----------
X_train : array-like or DataFrame
    Training feature matrix.

Y_train : array-like or DataFrame
    Training response matrix.

save_dir : str or None
    Directory to save the components selection plot. If None, plot is displayed.

Returns
-------
dict
    Contains:
    - best_n_components : int
    - cv_mse_scores : array
    - q2_scores : array
    - score : array (combined selection criterion)
"""
def pls_components_selection_pipeline(X_train, Y_train, save_dir=None):
    
    test_comps = range(1, 15)
    cv = KFold(n_splits=10, shuffle=True, random_state=42)

    cv_mse_scores = []
    Q2Ys = []

    # Compute baseline
    Y_mean = np.mean(Y_train, axis=0)
    tss = np.sum((Y_train - Y_mean) ** 2)

    # Model Selection
    for n_comp in test_comps:

        model = PLSRegression(n_components=n_comp)

        Y_cv_pred = cross_val_predict(
            model,
            X_train,
            Y_train,
            cv=cv
        )

        # MSE
        mse = mean_squared_error(Y_train, Y_cv_pred)
        cv_mse_scores.append(mse)

        # Q²Y
        press = np.sum((Y_train - Y_cv_pred) ** 2)
        q2 = 1 - (press / tss)
        Q2Ys.append(q2)

    cv_mse_scores = np.array(cv_mse_scores)
    Q2Ys = np.array(Q2Ys)

    # Normalize MSE and Q2Y value
    mse_norm = (cv_mse_scores - cv_mse_scores.min()) / (np.ptp(cv_mse_scores))
    q2_norm = (Q2Ys - Q2Ys.min()) / (np.ptp(Q2Ys))

    # Combined score
    score = mse_norm - q2_norm

    best_idx = np.argmin(score)
    best_n_comp = list(test_comps)[best_idx]

    print(f"Optimal number of components: {best_n_comp}")
    print(f"CV-MSE: {cv_mse_scores[best_idx]:.4f}")
    print(f"Q²Y: {Q2Ys[best_idx]:.4f}")

    # Plot results
    plt.figure(figsize=(8, 5))
    plt.plot(test_comps, Q2Ys, marker='o', label="Q²Y")
    plt.plot(test_comps, cv_mse_scores, marker='o', label="MSE")
    plt.axvline(best_n_comp, linestyle="--", color="red", label="Selected")
    plt.xlabel("Number of PLS Components")
    plt.ylabel("Score")
    plt.title("PLSR Model Selection")
    plt.legend()
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, "components_selection.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {
        "best_n_components": best_n_comp,
        "cv_mse_scores": cv_mse_scores,
        "q2_scores": Q2Ys,
        "score": score
    }
# --- PLSR Evaluation ---#
"""
PLSR Evaluation Pipeline

Trains a PLSRegression model using the selected number of components
and evaluates its predictive performance on held-out test data.

Computes:
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- Pearson correlation (linear association)
- Spearman correlation (rank-based association)

This provides both regression error metrics and biological agreement measures.

Parameters
----------
X_train : array-like
    Training features.

Y_train : array-like
    Training targets.

X_test : array-like
    Test features.

Y_test : array-like
    Test targets.

best_n_comp : int
    Number of PLS components selected from model selection step.

save_dir : str or None
    Directory to save evaluation metrics CSV.

Returns
-------
dict
    Contains trained model, predictions, flattened arrays,
    and performance metrics (RMSE, MAE, Pearson, Spearman).
"""
def plsr_evaluation_pipeline(X_train, Y_train, X_test, Y_test, best_n_comp, save_dir=None):

    # PLSR model
    plsr = PLSRegression(n_components=best_n_comp)

    # Fit
    plsr.fit(X_train, Y_train)

    # Predict
    Y_pred = plsr.predict(X_test)

    # Flatten for global metrics
    y_true = Y_test.flatten()
    y_pred = Y_pred.flatten()

    # Performance error metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # Correlation metrics
    pearson_corr, _ = pearsonr(y_true, y_pred)
    spearman_corr, _ = spearmanr(y_true, y_pred)

    if save_dir:
        pd.DataFrame([{
            "RMSE": rmse,
            "MAE": mae,
            "Pearson": pearson_corr,
            "Spearman": spearman_corr
        }]).to_csv(os.path.join(save_dir, "plsr_metrics.csv"), index=False)

    return {
        "model": plsr,
        "Y_pred": Y_pred,
        "y_true": y_true,
        "y_pred": y_pred,
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson_corr,
        "spearman": spearman_corr
    }
# --- PLSR Scores Plot ---#
"""
PLSR Scores Plot Pipeline

Visualizes the latent space of samples in PLSR component space,
highlighting separation between predicted anticancer drugs and background compounds.

Key steps:
- Projects test data into PLSR latent space
- Separates high-confidence anticancer drugs based on AUC_Killing_Score threshold
- Plots score distribution in first two PLS components
- Annotates top-ranked anticancer candidates

This visualization helps interpret clustering and separation in model-derived space.

Parameters
----------
results_df : DataFrame
    Contains drug-level metadata including AUC_Killing_Score and targets.

plsr : trained PLSRegression model
    Fitted model used for transformation.

X_test : array-like
    Test feature matrix.

adjust_text : function
    Annotation adjustment function for label overlap reduction.

save_dir : str or None
    Directory to save plot. If None, plot is shown.

Returns
-------
dict
    Contains:
    - strong_drugs : DataFrame of high-confidence candidates
    - anticancer_df : annotated projection table
    - X_scores_anticancer : projected anticancer drug scores
    - X_scores_bg : projected background scores
"""
def plsr_scores_plot_pipeline(
    results_df,
    plsr,
    X_test,
    adjust_text,
    save_dir=None
):

    # Define strong drugs
    strong_mask = results_df["AUC_Killing_Score"] >= 0.7
    strong_drugs = results_df[strong_mask]

    anticancer_idx = strong_drugs.index
    background_idx = results_df.index.difference(anticancer_idx)

    # Latent space projection
    X_test_scores = plsr.transform(X_test)

    # safe index mapping
    index_list = list(results_df.index)

    anticancer_pos = [index_list.index(i) for i in anticancer_idx]
    background_pos = [index_list.index(i) for i in background_idx]

    X_scores_anticancer = X_test_scores[anticancer_pos]
    X_scores_bg = X_test_scores[background_pos]

    # Plot Setup
    plt.figure(figsize=(6, 5))

    # Background drugs
    plt.scatter(
        X_scores_bg[:, 0],
        X_scores_bg[:, 1],
        color="lightgray",
        s=15,
        alpha=0.6,
        label="Other noncancer drugs"
    )

    # Strong anticancer drugs
    plt.scatter(
        X_scores_anticancer[:, 0],
        X_scores_anticancer[:, 1],
        color="#E23C4A",
        s=40,
        alpha=0.9,
        edgecolor="k",
        label="Predicted anticancer drugs"
    )

    # Label top candidates
    top_n = min(20, len(strong_drugs))
    top_hits = strong_drugs.sort_values(
        "AUC_Killing_Score",
        ascending=False
    ).head(top_n)

    texts = []

    for drug in top_hits.index:
        pos = anticancer_idx.get_loc(drug)

        texts.append(
            plt.text(
                X_scores_anticancer[pos, 0],
                X_scores_anticancer[pos, 1],
                drug,
                fontsize=8
            )
        )

    adjust_text(
        texts,
        arrowprops=dict(arrowstyle='-', color='gray', lw=0.5),
        expand_text=(1.2, 1.4),
        expand_points=(1.2, 1.4),
        force_text=(0.3, 0.5),
        force_points=(0.2, 0.3)
    )

    plt.title(
        "Scores Plot of Noncancer Drugs with \nStrong Anti-Cancer Potential vs Weak Anti-Cancer Potential",
        fontsize=15,
        fontweight="bold"
    )

    plt.xlabel("PLSR Component 1")
    plt.ylabel("PLSR Component 2")

    plt.axhline(0, color="gray", lw=0.5)
    plt.axvline(0, color="gray", lw=0.5)

    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "plsr_scores_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Output table
    anticancer_df = pd.DataFrame({
        "Drug": strong_drugs.index,
        "Target": strong_drugs["Target"].values,
        "AUC_Killing_Score": strong_drugs["AUC_Killing_Score"].values,
        "PLS1": X_scores_anticancer[:, 0],
        "PLS2": X_scores_anticancer[:, 1]
    })

    print(
        anticancer_df.sort_values(
            "AUC_Killing_Score",
            ascending=False
        )
    )

    return {
        "strong_drugs": strong_drugs,
        "anticancer_df": anticancer_df,
        "X_scores_anticancer": X_scores_anticancer,
        "X_scores_bg": X_scores_bg
    }
# --- PLSR Loadings Plot on Oncology Target Features ---#
"""
PLSR Loadings Plot Pipeline

Visualizes feature loadings from a trained PLSR model in the first two components,
highlighting the most influential features based on loading magnitude.

Steps:
- Extracts X-loadings from trained model
- Computes feature importance using Euclidean magnitude across first two components
- Identifies top contributing features
- Plots full feature space and highlights top features

This helps interpret which biological targets drive latent structure.

Parameters
----------
plsr : trained PLSRegression model
    Fitted model containing x_loadings_.

X_final : DataFrame
    Feature matrix used for training (provides feature names).

save_dir : str or None
    Directory to save plot. If None, plot is shown.

Returns
-------
dict
    Contains:
    - X_loadings : array
    - feature_names : array
    - top_loading_indices : array
    - loading_magnitudes : array
"""
def plsr_loadings_plot_pipeline(plsr, X_final, save_dir=None):

    # Extract loadings and feature names
    X_loadings = plsr.x_loadings_
    feature_names = X_final.columns.values

    # Compute feature importance (magnitude in first 2 components)
    loading_magnitudes = np.sqrt(
        X_loadings[:, 0] ** 2 + X_loadings[:, 1] ** 2
    )

    # Select top 10 features
    top_loading_indices = np.argsort(loading_magnitudes)[-10:]

    # Create figure
    plt.figure(figsize=(8, 6))

    # Plot background features
    plt.scatter(
        X_loadings[:, 0],
        X_loadings[:, 1],
        color="lightgray",
        alpha=0.5,
        label="All targets"
    )

    # Plot top features
    plt.scatter(
        X_loadings[top_loading_indices, 0],
        X_loadings[top_loading_indices, 1],
        color="darkblue",
        s=50,
        edgecolors="black",
        label="Top 10 targets"
    )

    # Annotate
    for idx in top_loading_indices:
        clean_label = feature_names[idx].replace("target_", "")

        plt.annotate(
            clean_label,
            (X_loadings[idx, 0], X_loadings[idx, 1]),
            textcoords="offset points",
            xytext=(5, 5),
            ha="right",
            fontsize=8,
            fontweight="semibold"
        )

    plt.title(
        "PLSR Loadings Plot of Oncology Target Features",
        fontsize=14,
        fontweight="bold"
    )

    plt.xlabel("PLSR Component 1")
    plt.ylabel("PLSR Component 2")

    plt.axhline(0, color="gray", linewidth=0.5)
    plt.axvline(0, color="gray", linewidth=0.5)

    plt.grid(True, linestyle=":", alpha=0.5)

    # Legend
    plt.legend()

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "plsr_loadings_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    # Return outputs
    return {
        "X_loadings": X_loadings,
        "feature_names": feature_names,
        "top_loading_indices": top_loading_indices,
        "loading_magnitudes": loading_magnitudes
    }

# -- PLSR Loadings Plot on Predicted Anti-Cancer Drug Target Features--#
"""
Anticancer Target Loadings Pipeline

Maps predicted anticancer drug targets onto PLSR loading space
to identify which biological features are enriched among high-confidence drugs.

Steps:
- Extracts unique targets from predicted anticancer drugs
- Maps targets to feature indices in loading matrix
- Visualizes selected targets within global loading structure
- Annotates biologically relevant features

This connects model predictions back to biological target space.

Parameters
----------
strong_drugs : DataFrame
    Subset of predicted high-confidence anticancer drugs.

X_loadings : array-like
    PLSR feature loading matrix.

feature_names : array-like
    Feature names corresponding to loading rows.

save_dir : str or None
    Directory to save plot. If None, plot is shown.

Returns
-------
dict
    Contains:
    - valid_indices : indices of mapped targets
    - encoded_targets : formatted target names
    - anticancer_targets : raw extracted targets
    - unique_anticancer_targets : deduplicated targets
    - feature_index_map : mapping of feature name → index
"""

def anticancer_target_loadings_pipeline(
    strong_drugs,
    X_loadings,
    feature_names,
    save_dir=None
):

    anticancer_targets = strong_drugs["Target"].values
    unique_anticancer_targets = np.unique(anticancer_targets)
    encoded_targets = np.array([f"target_{t}" for t in unique_anticancer_targets])

    # Map feature names to indices once
    feature_index_map = {name: i for i, name in enumerate(feature_names)}

    # Find valid indices
    valid_indices = [
        feature_index_map[t] for t in encoded_targets if t in feature_index_map
    ]

    valid_indices = np.array(valid_indices)

    # Plot Loadings
    plt.figure(figsize=(7, 6))

    plt.scatter(
        X_loadings[:, 0],
        X_loadings[:, 1],
        color="lightgray",
        alpha=0.5,
        label="All targets"
    )

    # Highlight anticancer targets
    plt.scatter(
        X_loadings[valid_indices, 0],
        X_loadings[valid_indices, 1],
        color="#E23C4A",
        s=50,
        edgecolors="black",
        label="Targets in anticancer-predicted drugs",
        zorder=3
    )

    # Create labels
    for idx in valid_indices:
        clean_label = feature_names[idx].replace("target_", "")

        plt.annotate(
            clean_label,
            (X_loadings[idx, 0], X_loadings[idx, 1]),
            textcoords="offset points",
            xytext=(8, 5),
            ha="left",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none")
        )

    plt.title(
        "Loadings Plot of Targets of Predicted Anticancer Drugs",
        fontsize=16,
        fontweight="bold"
    )

    plt.xlabel("PLS Component 1 Loading")
    plt.ylabel("PLS Component 2 Loading")

    plt.axhline(0, color="black", linewidth=0.5)
    plt.axvline(0, color="black", linewidth=0.5)

    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.4)

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "anticancer_target_loadings_plot.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    #Return values
    return {
        "valid_indices": valid_indices,
        "encoded_targets": encoded_targets,
        "anticancer_targets": anticancer_targets,
        "unique_anticancer_targets": unique_anticancer_targets,
        "feature_index_map": feature_index_map
    }


# -- Target Overlap --#
"""
Target Overlap Pipeline

Compares overlap between:
- Blue targets: top features identified from model loadings (e.g., PCA/PLS components)
- Red targets: known anticancer target set

Computes:
- Set overlap and union between both target groups
- Jaccard similarity index as a measure of agreement between sets

Parameters
----------
feature_names : array-like
    List/array of all feature names corresponding to model inputs.

top_loading_indices : array-like
    Indices of top features selected from model loadings (e.g., highest absolute weights).

anticancer_targets : list or set
    Ground-truth or literature-derived anticancer target genes/features.

save_dir : str or None
    Directory to save overlap summary CSV. If None, results are not saved.

Returns
-------
dict
    Dictionary containing:
    - red_targets (set)
    - blue_targets (set)
    - overlap (set intersection)
    - union (set union)
    - jaccard (float similarity score)
"""
def target_overlap_pipeline(
    feature_names,
    top_loading_indices,
    anticancer_targets,
    save_dir=None
):

    blue_targets = set(feature_names[top_loading_indices])
    red_targets = set(anticancer_targets)

    overlap = red_targets.intersection(blue_targets) # Identify shared targets between model and known biology
    union = red_targets.union(blue_targets) # Total unique targets across both sets (used for Jaccard denominator)

    print("Red targets:", len(red_targets))
    print("Blue targets:", len(blue_targets))
    print("Overlap:", len(overlap))
    print("Overlapping targets:", overlap)

    jaccard = len(overlap) / len(union)
    print("Jaccard similarity:", jaccard)
    
    if save_dir:
        pd.DataFrame({
            "Metric": ["Red Targets", "Blue Targets", "Overlap", "Jaccard"],
            "Value": [len(red_targets), len(blue_targets), len(overlap), jaccard]
        }).to_csv(os.path.join(save_dir, "target_overlap.csv"), index=False)

    return {
        "red_targets": red_targets,
        "blue_targets": blue_targets,
        "overlap": overlap,
        "union": union,
        "jaccard": jaccard
    }