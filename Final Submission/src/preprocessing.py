import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import LeaveOneGroupOut, cross_val_predict, KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
from adjustText import adjust_text
from sklearn.ensemble import RandomForestRegressor

#---- Preprocessing Stage -----#
"""
Data Loading and Preprocessing Pipeline

Prepares matched feature (X) and response (Y) matrices for downstream modeling
by integrating log-fold change data with drug metadata.

This pipeline performs full data harmonization including cleaning, alignment,
encoding, and train/test splitting based on biological drug categories.

Processing steps:
- Loads log-fold change matrix and drug metadata
- Removes rows/columns with missing values
- Standardizes identifiers across datasets (whitespace cleanup)
- Aligns drugs present in both datasets
- Constructs response matrix (Y) as drug-by-feature transpose
- Handles missing target annotations
- One-hot encodes drug target labels
- Aggregates duplicate target encodings per drug
- Synchronizes feature and response matrices
- Splits dataset into:
  - Training set: targeted cancer drugs
  - Test set: noncancer drugs

This ensures biologically meaningful separation between model training
and evaluation sets.

Parameters
----------
logfold_file : str
    Path to log-fold change matrix CSV (features × drugs).

metadata_file : str
    Path to drug metadata CSV containing target and drug category annotations.

Returns
-------
dict
    Dictionary containing:

    X_train : ndarray
        Feature matrix for targeted cancer drugs.

    Y_train : ndarray
        Response matrix for targeted cancer drugs.

    X_test : ndarray
        Feature matrix for noncancer drugs.

    Y_test : ndarray
        Response matrix for noncancer drugs.

    X_final : DataFrame
        Full aligned feature matrix.

    Y_final : DataFrame
        Full aligned response matrix.

    metadata : DataFrame
        Raw drug metadata.

    metadata_filtered : DataFrame
        Cleaned and aligned metadata indexed by drug.

    Y_matrix : DataFrame
        Intermediate response matrix (drugs × features).

    common_indices : Index
        Shared drug identifiers across datasets.

    is_cancer : Series
        Boolean mask for targeted cancer drugs (training set).

    is_noncancer : Series
        Boolean mask for noncancer drugs (test set).
"""
def load_and_prepare_data(logfold_file,metadata_file):

    logfold_raw= pd.read_csv(logfold_file, index_col=0)
    drug_metadata= pd.read_csv(metadata_file)

    # drop cell lines with missing data NaN
    logfold_v2 = logfold_raw.dropna(axis=0).dropna(axis=1)

    print(f"Original shape: {logfold_raw.shape}")
    print(f"Shape after dropping drug and cell lines with NaNs: {logfold_v2.shape}")

    # cleaning drug dataset - clean missing identifiers and strip whitespace from matching columns
    drug_metadata = drug_metadata.dropna(subset=['column_name'])
    drug_metadata['column_name'] = drug_metadata['column_name'].str.strip()
    logfold_v2.columns = logfold_v2.columns.str.strip()

    # Extract the alignment keys that exist simultaneously in both datasets
    valid_drug_keys = logfold_v2.columns.intersection(drug_metadata['column_name'])

    # Filter target matrix to only retain matched compounds
    logfold_v3 = logfold_v2[valid_drug_keys]
    Y_matrix = logfold_v3.T # to make drug rows (to match biomarker data)

    # Set the index of your drug metadata to match the filtered layout
    drug_metadata_filtered = drug_metadata[drug_metadata['column_name'].isin(valid_drug_keys)].set_index('column_name')

    print(f"Shape of filtered metadata: {drug_metadata_filtered.shape}")

    # Fill missing targets with a placeholder string before encoding
    drug_metadata_filtered['target'] = drug_metadata_filtered['target'].fillna('Unknown_Target')

    # One-hot encode the target proteins
    X_features = pd.get_dummies(drug_metadata_filtered['target'], prefix='target')

    # Group by the index (column_name) just in case a drug maps to multiple targets
    X_features = X_features.groupby(X_features.index).sum()

    # Reindex both matrices to ensure identical row order
    common_indices = X_features.index.intersection(Y_matrix.index)
    X_final = X_features.loc[common_indices]
    Y_final = Y_matrix.loc[common_indices]

    # Create masks using the drug_category column from your filtered metadata
    is_cancer = drug_metadata_filtered.loc[common_indices, 'drug_category'] == 'targeted cancer'
    is_noncancer = drug_metadata_filtered.loc[common_indices, 'drug_category'] == 'noncancer'

    # Training Set (Targeted Cancer Drugs)
    X_train = X_final[is_cancer].values
    Y_train = Y_final[is_cancer].values

    # Testing Set (Noncancer Drugs)
    X_test = X_final[is_noncancer].values
    Y_test = Y_final[is_noncancer].values

    return {
        "X_train": X_train,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "X_final": X_final,
        "Y_final": Y_final,
        "metadata": drug_metadata,
        "metadata_filtered": drug_metadata_filtered,
        "Y_matrix": Y_matrix,
        "common_indices": common_indices,
        "is_cancer": is_cancer,
        "is_noncancer": is_noncancer}