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


# --- Drug Ranking Profile --#
def auc_drug_ranking_pipeline(
    Y_pred,
    drug_metadata_filtered,
    common_indices,
    is_noncancer,
    save_dir=None
):
    """
    Full Drug Ranking pipeline:
    - 
    """ 

    # AUC-like killing score
    auc_score = np.mean(Y_pred < -0.5, axis=1)

    # Align metadata
    noncancer_drug_names = drug_metadata_filtered.loc[common_indices][is_noncancer]['name'].values
    targets = drug_metadata_filtered.loc[common_indices][is_noncancer]['target'].values

    # Build results table
    results_df = pd.DataFrame({
        'Drug_Name': noncancer_drug_names,
        'AUC_Killing_Score': auc_score,
        'Target': targets
    }).sort_values(by='AUC_Killing_Score', ascending=False)

    # Print Top 10 list
    print("\nTop 10 noncancer drugs predicted to have the strongest anti-cancer profiles:")
    print(results_df.head(10))

    print("\nTop 10 noncancer drugs predicted to have the weakest anti-cancer profiles:")
    print(results_df.tail(10))

    #Print Strong hits
    strong_hits = results_df[results_df["AUC_Killing_Score"] >= 0.7]

    print(f"\nNumber of strong hits: {len(strong_hits)}")
    print(strong_hits[["Drug_Name", "AUC_Killing_Score", "Target"]])

    if save_dir:
        results_df.to_csv(os.path.join(save_dir, "drug_rankings.csv"), index=False)
        strong_hits.to_csv(os.path.join(save_dir, "strong_hits.csv"), index=False)

    return {
        "auc_score": auc_score,
        "results_df": results_df,
        "strong_hits": strong_hits
    }
