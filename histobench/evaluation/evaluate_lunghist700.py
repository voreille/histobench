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


def custom_balanced_group_kfold(X, y, groups, n_splits=5, random_state=42):
    """Ensure all classes appear in each fold"""
    rng = np.random.RandomState(random_state)
    unique_classes = np.unique(y)
    unique_groups = np.unique(groups)

    # Create a mapping from string labels to integers if needed
    if not np.issubdtype(unique_classes.dtype, np.number):
        class_mapping = {label: i for i, label in enumerate(unique_classes)}
        y_numeric = np.array([class_mapping[label] for label in y])
    else:
        y_numeric = y

    # Group patients by class
    class_to_groups = {c: [] for c in unique_classes}
    for group in unique_groups:
        group_mask = groups == group
        group_classes = y[group_mask]

        # Count occurrences of each class in this group
        if not np.issubdtype(unique_classes.dtype, np.number):
            counts = pd.Series(group_classes).value_counts()
            most_common_class = counts.idxmax()
        else:
            most_common_class = np.bincount(y_numeric[group_mask]).argmax()
            most_common_class = unique_classes[most_common_class]

        class_to_groups[most_common_class].append(group)

    # Create folds ensuring each class is represented
    folds = [[] for _ in range(n_splits)]
    for cls, cls_groups in class_to_groups.items():
        cls_groups = list(cls_groups)
        rng.shuffle(cls_groups)
        for i, group in enumerate(cls_groups):
            fold_idx = i % n_splits
            folds[fold_idx].append(group)

    # Generate train/test indices for each fold
    for i in range(n_splits):
        test_groups = folds[i]
        test_mask = np.isin(groups, test_groups)
        test_indices = np.where(test_mask)[0]
        train_indices = np.where(~test_mask)[0]
        yield train_indices, test_indices


def compute_metrics(y_true, y_pred, y_score=None):
    from sklearn.utils.multiclass import type_of_target

    # Determine if binary or multiclass
    target_type = type_of_target(y_true)
    if target_type == "binary":
        average = "binary"
    else:
        average = "macro"

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0, average=average),
        "recall": recall_score(y_true, y_pred, zero_division=0, average=average),
        "f1": f1_score(y_true, y_pred, zero_division=0, average=average),
    }
    if y_score is not None and target_type == "binary":
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_score)
        except Exception:
            metrics["roc_auc"] = None
    else:
        metrics["roc_auc"] = None
    return metrics


def evaluate_split(train_idx, test_idx, embeddings, y, knn_n_neighbors=20):
    X_train, X_test = embeddings[train_idx, :], embeddings[test_idx, :]
    y_train, y_test = [y[i] for i in train_idx], [y[i] for i in test_idx]
    embedding_dim = X_train.shape[1]
    n_classes = len(set(y_train))
    logger.info(f"Embedding dimension: {embedding_dim}, Number of classes: {n_classes}")
    logger.info(
        f"Evaluating split: train={len(train_idx)}, test={len(test_idx)}, classes={set(y_train)}"
    )
    logger.info(f"Train labels distribution: {pd.Series(y_train).value_counts().to_dict()}")
    logger.info(f"Test labels distribution: {pd.Series(y_test).value_counts().to_dict()}")

    report_rows = []

    # KNN Probing (K=20, Euclidean)
    knn = KNeighborsClassifier(n_neighbors=knn_n_neighbors, metric="euclidean")
    knn.fit(X_train, y_train)
    y_pred_knn = knn.predict(X_test)
    y_score_knn = knn.predict_proba(X_test)[:, 1] if n_classes == 2 else None
    metrics_knn = compute_metrics(y_test, y_pred_knn, y_score_knn)
    metrics_knn["classifier"] = "knn"
    report_rows.append(metrics_knn)

    # Linear Probing (Logistic Regression)
    # λ = 100 * M / C, so C = 1 / λ
    l2_lambda = 100 / embedding_dim / n_classes
    C = 1.0 / l2_lambda
    lr = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=1000,
        multi_class="auto",
    )
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_score_lr = lr.predict_proba(X_test)[:, 1] if n_classes == 2 else None
    metrics_lr = compute_metrics(y_test, y_pred_lr, y_score_lr)
    metrics_lr["classifier"] = "logistic_regression"
    metrics_lr["l2_lambda"] = l2_lambda
    metrics_lr["C"] = C
    report_rows.append(metrics_lr)

    return report_rows


@click.command()
@click.option(
    "--csv-metadata",
    default="data/LungHist700/metadata.csv",
    type=click.Path(exists=True),
    help="Path to the CSV metadata file with patient information.",
)
@click.option(
    "--embeddings-path",
    default=None,
    type=click.Path(exists=True),
    show_default=True,
    help="Path to the .h5 embeddings file.",
)
@click.option(
    "--knn-n-neighbors", default=20, show_default=True, help="Number of neighbors for KNN."
)
@click.option(
    "--report-path",
    type=click.Path(),
    show_default=True,
    help="Path to save the metrics report (CSV).",
)
@click.option("--n-splits", default=5, show_default=True, help="Number of CV folds.")
def main(
    csv_metadata,
    embeddings_path,
    knn_n_neighbors,
    report_path,
    n_splits,
):
    csv_metadata_path = project_dir / csv_metadata
    metadata_df = pd.read_csv(csv_metadata_path).set_index("filename")
    embeddings_path = project_dir / embeddings_path if embeddings_path else None
    report_path = project_dir / report_path

    embeddings, image_paths = load_embeddings(embeddings_path)
    filenames = [Path(path).stem for path in image_paths]

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)  # add epsilon to avoid division by zero

    # Get patient_id and label for each embedding
    patient_ids = metadata_df.loc[filenames, "patient_id"].values
    labels = metadata_df.loc[filenames, "superclass"].values

    # Convert labels to binary (0/1) for two-class problem
    unique_labels = sorted(set(labels))
    label_map = {label: idx for idx, label in enumerate(unique_labels)}
    y = np.array([label_map[label] for label in labels])

    all_report_rows = []
    cv = custom_balanced_group_kfold(
        embeddings, y, patient_ids, n_splits=n_splits, random_state=42
    )
    for fold_idx, (train_idx, test_idx) in enumerate(cv):
        report_rows = evaluate_split(
            train_idx, test_idx, embeddings, y, knn_n_neighbors=knn_n_neighbors
        )
        for row in report_rows:
            row["fold"] = fold_idx
        all_report_rows.extend(report_rows)

    # Save combined report as CSV
    df_report = pd.DataFrame(all_report_rows)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    df_report.to_csv(report_path, index=False)
    logger.info(f"Metrics report saved to {report_path}")


if __name__ == "__main__":
    main()
