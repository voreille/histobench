from collections import Counter
import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from histobench.finetuning.finetune_lunghist700_avg_pool_with_finetuner import (
    CLASS_MAPPING,
    SUPERCLASS_MAPPING,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory containing ROI PNG images.",
)
@click.option(
    "--csv-metadata",
    required=True,
    type=click.Path(exists=True),
    help="Path to the CSV metadata file with patient information.",
)
@click.option(
    "--output-path",
    required=True,
    type=click.Path(),
    help="Path to save the generated CSV file with fold information.",
)
@click.option("--label-column", default="superclass", show_default=True, help="Column for labels.")
@click.option(
    "--n-folds", default=5, show_default=True, help="Number of folds for cross-validation."
)
def generate_cv_splits(input_dir, csv_metadata, output_path, label_column, n_folds):
    """
    Generate a CSV file with cross-validation fold information and print class ratios.
    """
    input_dir = Path(input_dir).resolve()
    image_paths = list(input_dir.glob("*.png"))
    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    metadata = pd.read_csv(csv_metadata).set_index("filename")
    class_mapping = CLASS_MAPPING if label_column == "class_name" else SUPERCLASS_MAPPING

    labels = (
        metadata.loc[[path.stem for path in image_paths], label_column].map(class_mapping).values
    )
    patient_ids = metadata.loc[[path.stem for path in image_paths], "patient_id"].values

    # Custom splitting logic to improve class distribution
    def custom_split(image_paths, labels, patient_ids, n_folds):
        unique_patients = pd.unique(patient_ids)
        patient_to_class = {pid: labels[patient_ids == pid] for pid in unique_patients}

        # Sort patients by class distribution
        sorted_patients = sorted(unique_patients, key=lambda pid: Counter(patient_to_class[pid]))

        # Assign patients to folds
        folds = [[] for _ in range(n_folds)]
        for i, pid in enumerate(sorted_patients):
            folds[i % n_folds].append(pid)

        # Generate splits
        splits = []
        for fold_idx in range(n_folds):
            test_patients = folds[fold_idx]
            train_val_patients = [pid for pid in unique_patients if pid not in test_patients]

            # Further split train_val into train and validation
            train_patients = train_val_patients[: len(train_val_patients) // 2]
            val_patients = train_val_patients[len(train_val_patients) // 2 :]

            splits.append(
                {
                    "train": train_patients,
                    "validation": val_patients,
                    "test": test_patients,
                }
            )

        return splits

    # Generate custom splits
    splits = custom_split(image_paths, labels, patient_ids, n_folds)

    # Prepare DataFrame to store fold information
    fold_data = []

    def print_class_ratios(labels, split_name, fold):
        label_counts = Counter(labels)
        total_labels = len(labels)
        label_ratios = {label: count / total_labels for label, count in label_counts.items()}
        logger.info(f"Class ratios for fold {fold}, split {split_name}: {label_ratios}")

    for fold, split in enumerate(splits):
        train_image_paths = [
            path for path in image_paths if metadata.loc[path.stem, "patient_id"] in split["train"]
        ]
        val_image_paths = [
            path for path in image_paths if metadata.loc[path.stem, "patient_id"] in split["validation"]
        ]
        test_image_paths = [
            path for path in image_paths if metadata.loc[path.stem, "patient_id"] in split["test"]
        ]

        train_labels = [class_mapping[metadata.loc[Path(path).stem, label_column]] for path in train_image_paths]
        val_labels = [class_mapping[metadata.loc[Path(path).stem, label_column]] for path in val_image_paths]
        test_labels = [class_mapping[metadata.loc[Path(path).stem, label_column]] for path in test_image_paths]

        # Print class ratios for each split
        print_class_ratios(train_labels, "train", fold)
        print_class_ratios(val_labels, "validation", fold)
        print_class_ratios(test_labels, "test", fold)

        # Add train split
        for path in train_image_paths:
            fold_data.append(
                {"original_filename": Path(path).name, "fold": fold, "split": "train"}
            )

        # Add validation split
        for path in val_image_paths:
            fold_data.append(
                {"original_filename": Path(path).name, "fold": fold, "split": "validation"}
            )

        # Add test split
        for path in test_image_paths:
            fold_data.append({"original_filename": Path(path).name, "fold": fold, "split": "test"})

    # Save to CSV
    fold_df = pd.DataFrame(fold_data)
    fold_df.to_csv(output_path, index=False)
    logger.info(f"Saved cross-validation splits to {output_path}")


if __name__ == "__main__":
    generate_cv_splits()