Evaluating Anticancer Potential of Non-Oncology Drugs
Overview
This project builds a computational pipeline to identify FDA-approved non-oncology drugs that may have anticancer potential. Using the PRISM Repurposing Dataset, we train a Partial Least Squares Regression (PLSR) model on known cancer drugs and apply it to non-cancer drugs to predict anticancer activity based on drug target profiles.

Dependencies
Install required packages before running:
pip install numpy pandas matplotlib scikit-learn scipy adjustText

How to Run
Navigate to the src/ directory and run main.py:
cd "Final Submission/src"
python3 main.py

To run the integration test:
cd "Final Submission/tests"
python3 Integration

Data
Raw data files are included in the `data/` folder. Processed outputs are saved automatically to `data/processed/` when the pipeline is run.

File Structure
Final Submission/
├── data/                          
│   ├── processed/                 # Auto-generated preprocessed files
│   ├── biomarkers.csv             # Drug metadata including target proteins and categories
│   └── primary-screen-replicate-collapsed-logfold-change.csv  # PRISM drug response matrix
│
├── src/                           
│   ├── __init__.py               
│   ├── main.py                    # Main pipeline entry point
│   ├── preprocessing.py           # Data loading and preprocessing
│   ├── pca_analysis.py            # PCA biplot and multi-view plots
│   ├── plsr_analysis.py           # PLSR model, scores, and loadings
│   ├── drug_ranking.py            # AUC-based drug ranking
│   └── random_forest.py           # Random Forest model and feature importance
│
├── tests/
│   └── Integration                # Full integration test
│
├── results/                       
│   ├── PCA/                       # PCA biplot and multi-view figures
│   ├── PLSR/                      # PLSR scores, loadings, and metrics
│   ├── Random_Forest/             # RF metrics and feature importance
│   └── Ranking/                   # Drug ranking CSVs and strong hits
│
├── README.txt
└── Contributors.txt               # Contributions from group members 	

________________


Pipeline Summary
1. Preprocessing — Loads and aligns the PRISM response matrix with drug metadata, one-hot encodes drug targets, and splits into cancer (train) and noncancer (test) sets
2. PCA — Exploratory analysis of drug response profiles across cancer cell lines
3. PLSR — Trains on cancer drugs, projects noncancer drugs into latent space, identifies candidates with cancer-like target profiles
4. Drug Ranking — Ranks noncancer drugs by predicted anticancer killing score
5. Random Forest — Secondary model for validation and feature importance
6. Overlap Analysis — Compares top targets identified by PLSR and Random Forest