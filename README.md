# Bankruptcy Prediction — Gradient Boosting Ensemble with Optuna

A production-grade binary classification pipeline predicting company bankruptcy from 64 anonymised financial ratios. Three gradient boosting models (LightGBM, XGBoost, CatBoost) are independently tuned using **Bayesian hyperparameter optimisation (Optuna)** and ensembled for the final prediction — built for a Kaggle course competition.

## Highlights

- **0.9046 ROC-AUC** on the hold-out validation set (Optuna-tuned XGBoost) — a **+0.009 gain** over the default-parameter baseline
- **Leak-free cross-validation** design: the preprocessing pipeline (imputer, outlier clipper) is fit on each training fold independently and applied to the validation fold — a critical detail that many implementations get wrong
- Missing-value indicators treat the *pattern of missingness* as a predictive signal, not just a nuisance to be imputed away
- Three-model ensemble (XGB + LGBM + CatBoost) submitted as the final Kaggle entry

## Project Structure

```
bankruptcy-prediction-ensemble/
├── bankruptcy_prediction.ipynb          # Main modelling notebook (Optuna tuning + ensemble)
├── bankruptcy_prediction_pipeline.ipynb # Full EDA + 5-model benchmark (Kaggle version)
├── data/
│   ├── bankruptcy_Train.csv             # Training set — 10,000 rows × 65 cols
│   ├── bankruptcy_Test_X.csv            # Test features (unlabelled)
│   └── bankruptcy_sample_submission.csv # Kaggle submission format
├── submissions/
│   ├── submission_lgb.csv               # LightGBM-only predictions
│   ├── submission_xgb_lgbm.csv          # XGB + LGBM ensemble
│   └── submission_ensemble_3models.csv  # Final 3-model ensemble
└── README.md
```

## Pipeline

```
Raw Data (10,000 × 64 features)
        │
        ├─ [Per-fold, leak-free]
        │   ├── Missing value indicators  (columns with >10% missingness → binary flag)
        │   ├── Median imputation         (fit on train fold only)
        │   └── Percentile clipping       (1st–99th percentile per column)
        │
        ├── LightGBM   → Optuna (50 trials, TPE sampler) → AUC 0.9008
        ├── XGBoost    → Optuna (100 trials, TPE sampler) → AUC 0.9046  ← Best
        └── CatBoost   → Optuna (50 trials, TPE sampler)  → AUC 0.8985
                  │
              Ensemble (probability averaging)
                  │
           Final Kaggle Submission
```

## Results

| Model | Optuna Trials | Validation AUC | Key Hyperparameters |
|---|---|---|---|
| Baseline XGBoost (default) | 0 | 0.8956 | sklearn defaults |
| LightGBM | 50 | 0.9008 | lr=0.023, leaves=110, depth=15, n_est=928 |
| **XGBoost** | **100** | **0.9046** | lr=0.021, depth=9, n_est=679 |
| CatBoost | 50 | 0.8985 | lr=0.082, depth=5, iter=917 |
| XGB + LGBM Ensemble | — | ~0.904 | Simple average |
| XGB + LGBM + CatBoost | — | ~0.903 | Final submission |

## Key Design Decisions

**Leak-free cross-validation** — The imputer is fit on each training fold independently and applied to the validation fold. Fitting on the full dataset before splitting is a common but subtle data leakage mistake that inflates reported performance metrics.

**Missing value indicators** — For features with >10% missingness, a binary `_missing` flag is added alongside the imputed value. This allows the model to learn that the *absence* of a financial ratio can itself be a signal of financial distress.

**Optuna TPE sampler** — The Tree-structured Parzen Estimator builds a probabilistic model of which hyperparameter regions yield good results, focusing sampling budget on the most promising areas of the search space.

**Ensemble strategy** — Averaging probability scores from independently trained complementary models (XGBoost + LightGBM) reduces prediction variance and is the standard approach for maximising Kaggle leaderboard placement.

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| LightGBM | Gradient boosting (fast, leaf-wise tree growth) |
| XGBoost | Gradient boosting (level-wise, strong regularisation) |
| CatBoost | Gradient boosting (ordered boosting, robust to noise) |
| Optuna | Bayesian hyperparameter optimisation (TPE sampler) |
| Scikit-learn | CV, imputation, metrics |
| Pandas / NumPy | Data manipulation |

## Getting Started

```bash
git clone https://github.com/himanshuladdhad/bankruptcy-prediction-ensemble.git
cd bankruptcy-prediction-ensemble
pip install lightgbm xgboost catboost optuna optuna-integration[sklearn] scikit-learn pandas numpy jupyter
jupyter notebook bankruptcy_prediction.ipynb
```

> Place `bankruptcy_Train.csv` and `bankruptcy_Test_X.csv` in the `data/` directory before running. The Optuna tuning cells (50–100 trials each) are the most time-intensive steps — expect ~15–30 min total on a standard CPU.

## Dataset

Competition data from the **MGMT 571 Kaggle Competition** (Fall 2025). The dataset contains 64 anonymised financial ratio features for 10,000 companies with a binary bankruptcy label. The class distribution is highly imbalanced (~9:1 solvent-to-bankrupt ratio).

---

*Part of my data science portfolio. See also: [Telecom Churn EDA](https://github.com/himanshuladdhad/telecom-churn-eda) · [MLB Salary Prediction](https://github.com/himanshuladdhad/mlb-salary-prediction)*
