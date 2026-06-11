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


# --- Random Forest ---#
def random_forest_pipeline(X_train, Y_train, X_test, Y_test, save_dir=None):
    """
    Full Random Forest pipeline:
    - 
    """ 
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None
    )

    # Train model
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred_rf)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, Y_pred_rf)

    y_true_flat = Y_test.ravel()
    y_pred_flat = Y_pred_rf.ravel()

    pearson = pearsonr(y_true_flat, y_pred_flat)[0]
    spearman = spearmanr(y_true_flat, y_pred_flat)[0]

    if save_dir:
        pd.DataFrame([{
            "RMSE": rmse,
            "MAE": mae,
            "Pearson": pearson,
            "Spearman": spearman
        }]).to_csv(os.path.join(save_dir, "rf_metrics.csv"), index=False)

    return {
        "model": rf,
        "Y_pred_rf": Y_pred_rf,
        "rmse": rmse,
        "mae": mae,
        "pearson": pearson,
        "spearman": spearman
    }


# -- Random Forest Importance Pipeline --#
def random_forest_importance_pipeline(rf, feature_names, save_dir=None):
    """
    Full Random Forest Importance pipeline:
    - 
    """ 

    importances = rf.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)

    print(importance_df.head(20))

    top_n = 10
    top_features = importance_df.head(top_n)

    plt.figure(figsize=(8,6))
    plt.barh(top_features["Feature"][::-1], top_features["Importance"][::-1])
    plt.title("Top Random Forest Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "rf_importance.png"), dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

    return {
        "importance_df": importance_df,
        "top_features": top_features,
        "importances": importances
    }


# --- Random Forest Overap with PLSR ---#
def plsr_rf_overlap_pipeline(feature_names, loading_magnitudes, importance_df, save_dir=None):
    """
    Full Random Forest Overlap pipeline:
    - 
    """ 

    # Top 10 PLSR targets
    plsr_top = feature_names[np.argsort(loading_magnitudes)[-10:]]

    # Top 10 RF targets
    rf_top = importance_df.head(10)["Feature"].values

    # Overlap
    overlap = set(plsr_top).intersection(set(rf_top))

    overlap_df = pd.DataFrame(
        {"Shared_Targets": sorted(overlap)}
    )

    if save_dir:
        overlap_df.to_csv(os.path.join(save_dir, "plsr_rf_overlap.csv"), index=False)
        pd.DataFrame({"PLSR_Top": plsr_top}).to_csv(os.path.join(save_dir, "plsr_top_targets.csv"), index=False)
        pd.DataFrame({"RF_Top": rf_top}).to_csv(os.path.join(save_dir, "rf_top_targets.csv"), index=False)

    return {
        "plsr_top": plsr_top,
        "rf_top": rf_top,
        "overlap": overlap,
        "overlap_df": overlap_df
    }
