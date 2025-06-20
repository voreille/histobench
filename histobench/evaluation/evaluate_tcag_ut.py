import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from histobench.evaluation.compute_embeddings_tcga_ut import load_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = Path(__file__).parents[2].resolve()


def get_split_idx(df_split, image_paths, split_type, no_val=False):
    """Get the positional indices of the images for a specific split type."""
    split_column = f"split_{split_type}"

    patient_splits = df_split.groupby("patient")[split_column].nunique()
    multi_split_patients = patient_splits[patient_splits > 1]

    if not multi_split_patients.empty:
        logger.error("Patients in multiple splits:", multi_split_patients.index.tolist())
    else:
        logger.info("All patients are assigned to only one split.")

    # Ensure 'patient' and 'path' columns exist
    assert "patient" in df_split.columns and "path" in df_split.columns, (
        "CSV must have 'patient' and 'path' columns"
    )

    # Map patient to split
    patient_to_split = df_split.groupby("patient")[split_column].first().to_dict()

    # Get patient name from image path (assuming parent dir is patient)
    splits = [
        patient_to_split.get("-".join(path.parent.name.split("-")[:3]), None)
        for path in image_paths
    ]

    # Get indices for each split
    train_idx = [i for i, s in enumerate(splits) if s == "train"]
    val_idx = [i for i, s in enumerate(splits) if s == "valid"]
    test_idx = [i for i, s in enumerate(splits) if s == "test"]
    if no_val:
        # If no validation set, merge train and validation indices
        train_idx.extend(val_idx)
        return train_idx, test_idx

    return train_idx, val_idx, test_idx


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
    help="Name of the model to use.",
)
@click.option(
    "--split-type",
    default="internal",
    show_default=True,
    help="Either 'internal' or 'external'",
)
@click.option(
    "--knn-n-neighbors", default=5, show_default=True, help="Number of neighbors for KNN."
)
@click.option(
    "--knn-num-workers",
    default=24,
    show_default=True,
    help="Number of workers for KNN (not used yet).",
)
def main(csv_split_path, embeddings_path, split_type, knn_n_neighbors, knn_num_workers):
    """Simple CLI program to greet someone"""
    csv_split_path = project_dir / csv_split_path
    df_split = pd.read_csv(csv_split_path)
    embeddings_path = project_dir / embeddings_path if embeddings_path else None

    embeddings, image_paths = load_embeddings(embeddings_path)
    image_paths = [Path(path) for path in image_paths]
    labels = [path.parents[2].name for path in image_paths]
    train_idx, test_idx = get_split_idx(df_split, image_paths, split_type, no_val=True)

    embeddings_train = embeddings[train_idx, :]
    embeddings_test = embeddings[test_idx, :]
    labels_train = [labels[i] for i in train_idx]
    labels_test = [labels[i] for i in test_idx]

    classifier = KNeighborsClassifier(n_neighbors=knn_n_neighbors, n_jobs=knn_num_workers)
    classifier.fit(embeddings_train, labels_train)
    predictions = classifier.predict(embeddings_test)
    accuracy = (predictions == labels_test).mean()
    logger.info(f"Accuracy for {split_type} split: {accuracy:.4f}")


if __name__ == "__main__":
    main()
