"""Train and compare bankruptcy-prediction classifiers.

The dataset itself is deliberately excluded from version control. Run this module
with the path to the assignment Excel file; metrics and figures are written to
the chosen output directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 1312


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to the Excel dataset.")
    parser.add_argument("--output", type=Path, default=Path("results"), help="Directory for CSV files and figures.")
    return parser.parse_args()


def prepare_data(data_path: Path) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Read and validate the assignment dataset while retaining yearly labels."""
    dataset = pd.read_excel(data_path)
    if dataset.shape[1] < 13:
        raise ValueError("Expected at least 13 columns: 8 continuous, 3 binary, target, and year.")

    columns = list(dataset.columns)
    renamed = {columns[index]: f"x{index + 1}" for index in range(8)}
    renamed.update({columns[index]: f"binary{index - 7}" for index in range(8, 11)})
    renamed[columns[-2]] = "target"
    renamed[columns[-1]] = "year"
    dataset = dataset.rename(columns=renamed)

    if dataset.isna().any().any():
        missing = dataset.isna().sum()
        raise ValueError(f"Dataset contains missing values:\\n{missing[missing > 0]}")

    features = dataset[[*(f"x{i}" for i in range(1, 9)), "binary1", "binary2", "binary3"]]
    target = dataset["target"]
    if target.nunique() != 2:
        raise ValueError("The target must contain exactly two classes.")
    return features, target, dataset


def save_class_distribution(dataset: pd.DataFrame, output_dir: Path) -> None:
    """Save yearly counts by outcome class."""
    distribution = dataset.groupby(["year", "target"]).size().unstack(fill_value=0)
    ax = distribution.plot(kind="bar", figsize=(9, 5), color=["#4C78A8", "#F58518"])
    ax.set_title("Yearly Bankruptcy Class Distribution")
    ax.set_xlabel("Year")
    ax.set_ylabel("Companies")
    ax.legend(title="Target")
    plt.tight_layout()
    plt.savefig(output_dir / "class_distribution.png", dpi=250)
    plt.close()


def undersample_training_data(
    features: pd.DataFrame, target: pd.Series
) -> tuple[pd.DataFrame, pd.Series]:
    """Undersample the majority class to at most a 3:1 ratio in training data."""
    counts = target.value_counts()
    majority_label, minority_label = counts.index[0], counts.index[-1]
    if counts[majority_label] <= 3 * counts[minority_label]:
        return features, target

    majority_index = target[target == majority_label].index
    minority_index = target[target == minority_label].index
    sampled_majority = pd.Series(majority_index).sample(
        n=3 * len(minority_index), random_state=RANDOM_STATE
    )
    selected_index = pd.Index(minority_index).append(pd.Index(sampled_majority)).sort_values()
    return features.loc[selected_index], target.loc[selected_index]


def model_candidates() -> dict[str, tuple[Pipeline, dict[str, list[object]]]]:
    """Return the estimators and grids used in the original assignment."""
    return {
        "KNN": (Pipeline([("scale", MinMaxScaler()), ("model", KNeighborsClassifier())]), {"model__n_neighbors": list(range(1, 50))}),
        "GaussianNB": (Pipeline([("scale", MinMaxScaler()), ("model", GaussianNB())]), {}),
        "SVM": (Pipeline([("scale", MinMaxScaler()), ("model", SVC(probability=True, random_state=RANDOM_STATE))]), {"model__C": [0.1, 1, 10, 100], "model__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1], "model__kernel": ["linear", "rbf", "poly", "sigmoid"]}),
        "LDA": (Pipeline([("scale", MinMaxScaler()), ("model", LinearDiscriminantAnalysis())]), {"model__solver": ["svd", "lsqr", "eigen"]}),
        "LogisticRegression": (Pipeline([("scale", MinMaxScaler()), ("model", LogisticRegression(max_iter=3000, solver="saga", random_state=RANDOM_STATE))]), {"model__C": list(np.logspace(-3, 3, 7)), "model__penalty": ["l1", "l2"]}),
        "DecisionTree": (Pipeline([("scale", MinMaxScaler()), ("model", DecisionTreeClassifier(random_state=RANDOM_STATE))]), {"model__max_features": [None, "sqrt", "log2"], "model__ccp_alpha": [0.1, 0.01, 0.001], "model__max_depth": [5, 6, 7, 8, 9], "model__criterion": ["gini", "entropy"]}),
        "RandomForest": (Pipeline([("scale", MinMaxScaler()), ("model", RandomForestClassifier(random_state=RANDOM_STATE))]), {"model__n_estimators": [200, 500], "model__max_features": [None, "sqrt", "log2"], "model__max_depth": [4, 5, 6, 7, 8], "model__criterion": ["gini", "entropy"]}),
        "MLP": (Pipeline([("scale", MinMaxScaler()), ("model", MLPClassifier(max_iter=2000, random_state=RANDOM_STATE))]), {"model__hidden_layer_sizes": [(size,) for size in range(1, 6)], "model__alpha": list(10.0 ** (-np.arange(1, 10))), "model__activation": ["identity", "logistic", "tanh", "relu"]}),
    }


def positive_label(target: pd.Series) -> object:
    """Treat the larger assignment label as the bankruptcy (positive) class."""
    return sorted(target.unique())[-1]


def evaluate(y_true: pd.Series, predicted: np.ndarray, scores: np.ndarray, label: object) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, predicted),
        "precision": precision_score(y_true, predicted, pos_label=label, zero_division=0),
        "recall": recall_score(y_true, predicted, pos_label=label, zero_division=0),
        "f1": f1_score(y_true, predicted, pos_label=label, zero_division=0),
        "roc_auc": roc_auc_score((y_true == label).astype(int), scores),
    }


def main() -> None:
    args = parse_args()
    if not args.data.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.data}")
    args.output.mkdir(parents=True, exist_ok=True)

    features, target, dataset = prepare_data(args.data)
    save_class_distribution(dataset, args.output)
    splitter = StratifiedKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    records: list[dict[str, object]] = []
    best_params: list[dict[str, object]] = []
    label = positive_label(target)

    for fold, (train_index, test_index) in enumerate(splitter.split(features, target), start=1):
        x_train, y_train = features.iloc[train_index], target.iloc[train_index]
        x_test, y_test = features.iloc[test_index], target.iloc[test_index]
        x_train, y_train = undersample_training_data(x_train, y_train)

        for name, (estimator, parameters) in model_candidates().items():
            search = GridSearchCV(clone(estimator), parameters, scoring="roc_auc", n_jobs=-1)
            search.fit(x_train, y_train)
            predictions = search.predict(x_test)
            probabilities = search.predict_proba(x_test)
            positive_column = list(search.classes_).index(label)
            metrics = evaluate(y_test, predictions, probabilities[:, positive_column], label)
            records.append({"fold": fold, "model": name, **metrics})
            best_params.append({"fold": fold, "model": name, "parameters": json.dumps(search.best_params_, default=str)})

            matrix = confusion_matrix(y_test, predictions, labels=sorted(target.unique()))
            plt.figure(figsize=(4, 3))
            sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
            plt.title(f"{name} - Fold {fold}")
            plt.xlabel("Predicted class")
            plt.ylabel("Actual class")
            plt.tight_layout()
            plt.savefig(args.output / f"confusion_matrix_{name}_fold_{fold}.png", dpi=200)
            plt.close()

    metrics_frame = pd.DataFrame(records)
    metrics_frame.to_csv(args.output / "fold_metrics.csv", index=False)
    pd.DataFrame(best_params).to_csv(args.output / "best_parameters.csv", index=False)
    summary = metrics_frame.groupby("model")[["accuracy", "precision", "recall", "f1", "roc_auc"]].agg(["mean", "std"])
    summary.to_csv(args.output / "model_summary.csv")
    print("Saved metrics, hyperparameters, figures, and summary to", args.output)


if __name__ == "__main__":
    main()
