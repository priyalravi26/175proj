import pandas as pd
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # goes up one level from src/
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PCA_DIR = os.path.join(RESULTS_DIR, "PCA")
PLSR_DIR = os.path.join(RESULTS_DIR, "PLSR")
RF_DIR = os.path.join(RESULTS_DIR, "Random_Forest")
RANKING_DIR = os.path.join(RESULTS_DIR, "Ranking")

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from adjustText import adjust_text
from sklearn.ensemble import RandomForestRegressor
from preprocessing import load_and_prepare_data
from pca_analysis import pca_biplot_pipeline, plot_multiple_pca_views
from plsr_analysis import pls_components_selection_pipeline, plsr_evaluation_pipeline, plsr_scores_plot_pipeline, plsr_loadings_plot_pipeline, anticancer_target_loadings_pipeline, target_overlap_pipeline
from drug_ranking import auc_drug_ranking_pipeline
from random_forest import random_forest_pipeline, random_forest_importance_pipeline, plsr_rf_overlap_pipeline
sys.path.insert(0, os.path.dirname(__file__))

#Load datasets and preprocess
data = load_and_prepare_data(
    os.path.join(DATA_DIR, "primary-screen-replicate-collapsed-logfold-change.csv"),
    os.path.join(DATA_DIR, "biomarkers.csv")
)

X_train = data["X_train"]
X_test = data["X_test"]
Y_train = data["Y_train"]
Y_test = data["Y_test"]
X_final = data["X_final"]
Y_final = data["Y_final"]
drug_metadata = data ["metadata"]
drug_metadata_filtered = data["metadata_filtered"]
Y_matrix = data["Y_matrix"]
common_indices = data["common_indices"]
is_cancer = data["is_cancer"]
is_noncancer = data["is_noncancer"]

PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)  # creates the folder if it doesn't exist

np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
np.save(os.path.join(PROCESSED_DIR, "X_test.npy"), X_test)
np.save(os.path.join(PROCESSED_DIR, "Y_train.npy"), Y_train)
np.save(os.path.join(PROCESSED_DIR, "Y_test.npy"), Y_test)
np.save(os.path.join(PROCESSED_DIR, "X_final.npy"), X_final)
np.save(os.path.join(PROCESSED_DIR, "Y_final.npy"), Y_final)
np.save(os.path.join(PROCESSED_DIR, "Y_matrix.npy"), Y_matrix)

drug_metadata.to_csv(os.path.join(PROCESSED_DIR, "drug_metadata.csv"))
drug_metadata_filtered.to_csv(os.path.join(PROCESSED_DIR, "drug_metadata_filtered.csv"))

# Save boolean masks
np.save(os.path.join(PROCESSED_DIR, "is_cancer.npy"), is_cancer)
np.save(os.path.join(PROCESSED_DIR, "is_noncancer.npy"), is_noncancer)

# Save index mapping
common_indices.to_series().to_csv(os.path.join(PROCESSED_DIR, "common_indices.csv"), index=False)

#Size of the Cancer and Noncancer drugs after processing
print(f"Training set size (Cancer drugs): {X_train.shape[0]}")
print(f"Testing set size (Noncancer drugs): {X_test.shape[0]}")

#PLS component selection
results = pls_components_selection_pipeline(X_train, Y_train, save_dir=PLSR_DIR)
best_n_comp = results["best_n_components"]
cv_mse_scores = results["cv_mse_scores"]
Q2Ys = results["q2_scores"]

#PCA Biplot
results = pca_biplot_pipeline(Y_matrix, X_final, drug_metadata, save_dir=PCA_DIR)
drug_df = results["drug_df"]
explained_variance = results["explained_variance"]
top_targets = results["top_targets"]


#Plotting different PCA component pair view
plot_multiple_pca_views(drug_df, save_dir=PCA_DIR)


#Fits PLSR model on cancer drugs and evaluates predictive performance on noncancer drugs
results = plsr_evaluation_pipeline(
    X_train,
    Y_train,
    X_test,
    Y_test,
    best_n_comp,
    save_dir=PLSR_DIR
)
plsr = results["model"]
Y_pred = results["Y_pred"]

#Ranking noncancer drugs by predicted anticancer potential
auc_results = auc_drug_ranking_pipeline(
    Y_pred,
    drug_metadata_filtered,
    common_indices,
    is_noncancer,
    save_dir=RANKING_DIR
)

results_df = auc_results["results_df"]
strong_hits = auc_results["strong_hits"]

#PLSR scores
scores_results = plsr_scores_plot_pipeline(
    results_df,
    plsr,
    X_test,
    adjust_text,
    save_dir=PLSR_DIR
)
strong_drugs = scores_results["strong_drugs"]
anticancer_df = scores_results["anticancer_df"]


#PLSR loadings
loadings_results = plsr_loadings_plot_pipeline(plsr, X_final, save_dir=PLSR_DIR)
X_loadings = loadings_results["X_loadings"]
feature_names = loadings_results["feature_names"]
loading_magnitudes = loadings_results["loading_magnitudes"]
top_loading_indices = loadings_results["top_loading_indices"]

#Results
anticancer_loadings_results = anticancer_target_loadings_pipeline(
    strong_drugs,
    X_loadings,
    feature_names,
    save_dir=PLSR_DIR
)

anticancer_targets = anticancer_loadings_results["anticancer_targets"]
valid_indices = anticancer_loadings_results["valid_indices"]


#Computing similarity between top PLSR loading targets and predicted anticancer drug targets
overlap_results = target_overlap_pipeline(
    feature_names,
    top_loading_indices,
    anticancer_targets,
    save_dir=PLSR_DIR
)

jaccard = overlap_results["jaccard"]

#Trains Random Forest on cancer drugs and generates anticancer activity predictions for noncancer drugs
rf_results = random_forest_pipeline(X_train, Y_train, X_test, Y_test, save_dir=RF_DIR)
Y_pred_rf = rf_results["Y_pred_rf"]
rf = rf_results["model"]

#Extracts and visualizes the top Random Forest feature importances by drug target
importance_results = random_forest_importance_pipeline(rf, feature_names, save_dir=RF_DIR)
importance_df = importance_results["importance_df"]

#Comparing RF and PLSR results
overlap_results = plsr_rf_overlap_pipeline(
    feature_names,
    loading_magnitudes,
    importance_df,
    save_dir=RF_DIR)

overlap_df = overlap_results["overlap_df"]
