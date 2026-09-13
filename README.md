# Bankruptcy Prediction with Classification Models

An academic machine-learning project that compares several supervised classifiers for predicting corporate bankruptcy from financial indicators.

## Overview

The project prepares a mixed financial dataset, visualises class distribution and feature statistics, handles class imbalance with random undersampling of the training set, and evaluates candidate classifiers across stratified folds.

The implementation compares:

- K-Nearest Neighbours
- Gaussian Naive Bayes
- Support Vector Machine
- Linear Discriminant Analysis
- Logistic Regression
- Decision Tree
- Random Forest
- Multi-layer Perceptron

Each model is tuned with `GridSearchCV` and evaluated with accuracy, precision, recall, F1 score, ROC-AUC, and a confusion matrix.

## Project structure

```text
.
├── src/
│   └── train.py             # Reproducible training and evaluation workflow
├── requirements.txt         # Python dependencies
├── .gitignore               # Keeps data and generated results out of Git
└── README.md
```

## Dataset

The source dataset is not included in this repository. Place the Excel file supplied for the assignment at:

```text
data/Dataset2Use_Assignment1.xlsx
```

The code expects eight continuous predictors, three binary predictors, a target column, and a year column in the same column order as the original assignment dataset.

## Getting started

```bash
git clone https://github.com/<your-username>/bankruptcy-prediction-classification.git
cd bankruptcy-prediction-classification
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py --data data/Dataset2Use_Assignment1.xlsx --output results
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Methodology

1. Rename columns to concise predictor labels.
2. Inspect missing values and yearly class distribution.
3. Scale continuous features with Min-Max scaling inside each model pipeline.
4. Use stratified four-fold cross-validation to retain class proportions.
5. Apply random undersampling only to each training partition when its healthy-to-bankrupt ratio is greater than 3:1.
6. Tune model hyperparameters using grid search.
7. Export fold-level metrics and the selected hyperparameters to CSV.

## Notes

- This repository is a portfolio presentation of an academic project. It is not financial advice and should not be used for real lending or investment decisions.
- The supplied script has been refactored for local reproducibility. No original data is published.

## Technologies

Python, pandas, NumPy, scikit-learn, matplotlib, seaborn, and openpyxl.
