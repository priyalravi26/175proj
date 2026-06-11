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
"""
Random Forest Regression Pipeline

Trains and evaluates a Random Forest regressor for predicting continuous biological or drug response outcomes.

Steps:
- Fits a Random Forest model on training data
- Generates predictions on test data
- Computes regression performance metrics:
  - RMSE (Root Mean Squared Error)
  - MAE (Mean Absolute Error)
- Computes correlation-based metrics:
  - Pearson correlation (linear agreement)
  - Spearman correlation (rank-based agreement)

This provides a nonlinear baseline comparison against PLSR models.

Parameters
----------
X_train : array-like
    Training feature matrix.

Y_train : array-like
    Training target matrix.

X_test : array-like
    Test feature matrix.

Y_test : array-like
    Test target matrix.

save_dir : str or None
    Directory to save evaluation metrics CSV. If None, results are not saved.

Returns
-------
dict
    Contains:
    - model : trained RandomForestRegressor
    - Y_pred_rf : model predictions
    - rmse : float
    - mae : float
    - pearson : float
    - spearman : float
"""
def random_forest_pipeline(X_train, Y_train, X_test, Y_test, save_dir=None):
    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=None
    )

    # Train model
    rf.fit(X_train, Y_train)
    Y_pred_rf = rf.predict(X_test)

    mse = mean_squared_error(Y_test, Y_pred_rf) #standard regression error metrics
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(Y_test, Y_pred_rf)

    y_true_flat = Y_test.ravel() #flatten arrays to compute global correlation across all outputs
    y_pred_flat = Y_pred_rf.ravel()

    pearson = pearsonr(y_true_flat, y_pred_flat)[0]     # Pearson: linear agreement, Spearman: rank consistency
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
"""
Random Forest Feature Importance Pipeline

Extracts and visualizes feature importance scores from a trained Random Forest model.

Steps:
- Extracts feature importance values from trained model
- Constructs ranked feature importance table
- Displays top contributing features
- Plots top N features for interpretability

This identifies nonlinear feature contributions to model predictions.

Parameters
----------
rf : RandomForestRegressor
    Trained Random Forest model.

feature_names : array-like
    Names of input features corresponding to model inputs.

save_dir : str or None
    Directory to save feature importance plot. If None, plot is shown.

Returns
-------
dict
    Contains:
    - importance_df : full ranked importance table
    - top_features : top 10 features
    - importances : raw importance array
"""
def random_forest_importance_pipeline(rf, feature_names, save_dir=None):

    importances = rf.feature_importances_     # Extract built-in feature importance from trained RF model

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False)     # Pair each feature with its importance score and sort descending

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
"""
PLSR vs Random Forest Feature Overlap Pipeline

Compares top-ranked features identified by:
- PLSR loading magnitudes (linear latent structure)
- Random Forest feature importances (nonlinear predictive importance)

This provides insight into agreement between linear and nonlinear models
in identifying biologically relevant targets.

Steps:
- Extracts top features from PLSR loadings
- Extracts top features from Random Forest importances
- Computes intersection (shared predictive features)
- Saves ranked feature lists and overlap results

Parameters
----------
feature_names : array-like
    Feature names corresponding to model inputs.

loading_magnitudes : array-like
    Feature importance scores derived from PLSR loadings.

importance_df : DataFrame
    Ranked feature importance output from Random Forest pipeline.

save_dir : str or None
    Directory to save overlap results. If None, outputs are not saved.

Returns
-------
dict
    Contains:
    - plsr_top : top PLSR features
    - rf_top : top Random Forest features
    - overlap : set of shared features
    - overlap_df : DataFrame of shared targets
"""
def plsr_rf_overlap_pipeline(feature_names, loading_magnitudes, importance_df, save_dir=None):
    
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
