import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier

from histobench.evaluation.compute_embeddings_tcga_ut import load_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = Path(__file__).parents[2].resolve()


def get_split_idx(df_split, image_paths, split_type, no_val=False):
    split_column = f"split_{split_type}"
    patient_splits = df_split.groupby("patient")[split_column].nunique()
    multi_split_patients = patient_splits[patient_splits > 1]
    if not multi_split_patients.empty:
        raise ValueError(
            f"Some patients are assigned to multiple splits: {multi_split_patients.index.tolist()}. "
            "Please ensure each patient is assigned to only one split."
        )

    assert "patient" in df_split.columns and "path" in df_split.columns, (
        "CSV must have 'patient' and 'path' columns"
    )
    patient_to_split = df_split.groupby("patient")[split_column].first().to_dict()
    splits = [
        patient_to_split.get("-".join(path.parent.name.split("-")[:3]), None)
        for path in image_paths
    ]
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    val_idx = [i for i, s in enumerate(splits) if s == "valid"]
    test_idx = [i for i, s in enumerate(splits) if s == "test"]
    if no_val:
        train_idx.extend(val_idx)
        return train_idx, test_idx
    return train_idx, val_idx, test_idx


def compute_metrics(y_true, y_pred, y_score=None):
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_score is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def evaluate_split(train_idx, test_idx, embeddings, y, knn_n_neighbors=5):
    X_train, X_test = embeddings[train_idx, :], embeddings[test_idx, :]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]

    logger.info(
        f"Evaluating split: train={len(train_idx)}, test={len(test_idx)}, classes={set(y_train)}"
    )
    logger.info(f"Train labels distribution: {pd.Series(y_train).value_counts().to_dict()}")
    logger.info(f"Test labels distribution: {pd.Series(y_test).value_counts().to_dict()}")

    report_rows = []

    # KNN Classifier
    knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors)
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_score_knn = knn.predict_proba(X_test)[:, 1] if len(set(y)) == 2 else None
    metrics_knn = compute_metrics(y_test, y_pred_knn, y_score_knn)
    metrics_knn["classifier"] = "knn"
    report_rows.append(metrics_knn)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_score_lr = lr.predict_proba(X_test)[:, 1] if len(set(y)) == 2 else None
    metrics_lr = compute_metrics(y_test, y_pred_lr, y_score_lr)
    metrics_lr["classifier"] = "logistic_regression"
    report_rows.append(metrics_lr)

    return report_rows


@click.command()
@click.option(
    "--csv-split-path",
    default="data/tcga-ut/train_val_test_split.csv",
    type=click.Path(exists=True),
    help="Path to the CSV file containing the train/val/test split.",
)
@click.option(
    "--embeddings-path",
    default=None,
    type=click.Path(exists=True),
    show_default=True,
    help="Path to the .h5 embeddings file.",
)
@click.option(
    "--knn-n-neighbors", default=5, show_default=True, help="Number of neighbors for KNN."
)
@click.option(
    "--report-path",
    type=click.Path(),
    show_default=True,
    help="Path to save the metrics report (CSV).",
)
def main(
    csv_split_path,
    embeddings_path,
    knn_n_neighbors,
    report_path,
):
    csv_split_path = project_dir / csv_split_path
    df_split = pd.read_csv(csv_split_path)
    embeddings_path = project_dir / embeddings_path if embeddings_path else None
    report_path = project_dir / report_path

    embeddings, image_paths = load_embeddings(embeddings_path)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # add epsilon to avoid division by zero

    image_paths = [Path(path) for path in image_paths]
    labels = [path.parents[2].name for path in image_paths]

    # Convert labels to binary (0/1) for two-class problem
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = [label_map[label] for label in labels]

    all_report_rows = []

    for split_type in ["internal", "external"]:
        logger.info(f"Evaluating split: {split_type}")
        try:
            train_idx, test_idx = get_split_idx(df_split, image_paths, split_type, no_val=True)
        except Exception as e:
            logger.warning(f"Skipping split '{split_type}': {e}")
            continue

        report_rows = evaluate_split(
            train_idx, test_idx, embeddings, y, knn_n_neighbors=knn_n_neighbors
        )
        for row in report_rows:
            row["split_type"] = split_type
        all_report_rows.extend(report_rows)

    # Save combined report as CSV
    df_report = pd.DataFrame(all_report_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(report_path, index=False)
    logger.info(f"Metrics report saved to {report_path}")


if __name__ == "__main__":
    main()
