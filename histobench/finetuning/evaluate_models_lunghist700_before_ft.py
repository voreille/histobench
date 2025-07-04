import logging
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import KNeighborsClassifier
import torch
from torch.utils.data import DataLoader

from histobench.darya_code.get_model_inference import model_cls_map
from histobench.data.torch_datasets import LabelledTileOnTheFlyDataset
from histobench.finetuning.finetune_lunghist700_no_avg_pool_with_finetuner import (
    CLASS_MAPPING,
    SUPERCLASS_MAPPING,
    get_train_val_test_ids,
)
from histobench.models.foundation_models import FOUNDATION_MODEL_NAMES
from histobench.models.foundation_models import load_model as load_foundation_model
from histobench.models.load_models import load_pretrained_encoder
from histobench.utils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_dataloaders(
    image_paths, metadata, preprocess, batch_size, num_workers, label_column="superclass"
):
    class_mapping = CLASS_MAPPING if label_column == "class_name" else SUPERCLASS_MAPPING
    train_ids, val_ids, test_ids = get_train_val_test_ids("all")
    train_ids = train_ids + val_ids  # Combine train and val for training

    train_image_paths = [
        path for path in image_paths if metadata.loc[path.stem, "patient_id"] in train_ids
    ]
    test_image_paths = [
        path for path in image_paths if metadata.loc[path.stem, "patient_id"] in test_ids
    ]

    train_labels = [
        class_mapping[metadata.loc[path.stem, label_column]] for path in train_image_paths
    ]
    test_labels = [
        class_mapping[metadata.loc[path.stem, label_column]] for path in test_image_paths
    ]

    train_dataset = LabelledTileOnTheFlyDataset(
        roi_paths=train_image_paths, labels=train_labels, tile_transform=preprocess
    )
    test_dataset = LabelledTileOnTheFlyDataset(
        roi_paths=test_image_paths, labels=test_labels, tile_transform=preprocess
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=LabelledTileOnTheFlyDataset.get_collate_fn_ragged(),
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        collate_fn=LabelledTileOnTheFlyDataset.get_collate_fn_ragged(),
    )
    return train_dataloader, test_dataloader


def bootstrap_metrics(y_true, y_pred, n_bootstraps=1000, average="macro", seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(y_true)
    metrics_list = []

    for _ in range(n_bootstraps):
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        y_true_bs = y_true[indices]
        y_pred_bs = y_pred[indices]

        metrics_list.append(
            {
                "accuracy": accuracy_score(y_true_bs, y_pred_bs),
                "precision": precision_score(
                    y_true_bs, y_pred_bs, average=average, zero_division=0
                ),
                "recall": recall_score(y_true_bs, y_pred_bs, average=average, zero_division=0),
                "f1": f1_score(y_true_bs, y_pred_bs, average=average, zero_division=0),
            }
        )

    # Convert list of dicts to dict of arrays
    all_metrics = {k: np.array([m[k] for m in metrics_list]) for k in metrics_list[0].keys()}

    # Compute mean and 95% confidence intervals
    stats = {}
    for k, values in all_metrics.items():
        stats[k] = {
            "mean": np.mean(values),
            "ci_lower": np.percentile(values, 2.5),
            "ci_upper": np.percentile(values, 97.5),
        }

    return stats


def compute_embeddings(encoder, dataloader, device, normalize_embeddings=True):
    all_embeddings = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            embeddings = encoder(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    all_embeddings = torch.cat(all_embeddings).numpy()
    all_labels = torch.cat(all_labels).numpy()

    if normalize_embeddings:
        logger.info("Normalizing embeddings...")
        # Normalize embeddings to unit length
        norms = np.linalg.norm(all_embeddings, axis=1, keepdims=True)
        all_embeddings = all_embeddings / (norms + 1e-8)  # add epsilon to avoid division by zero
    return all_embeddings, all_labels


def compute_embeddings_on_tiles(model, dataloader, device="cuda", normalize_embeddings=True):
    model.to(device)
    model.eval()
    embeddings, labels = [], []

    with torch.inference_mode():
        for batch_of_bags, batch_labels in dataloader:
            mean_embeddings = torch.stack(
                [model(tiles.to(device)).mean(dim=0).cpu() for tiles in batch_of_bags], dim=0
            )
            embeddings.append(mean_embeddings.cpu())
            labels.append(batch_labels.cpu())

    embeddings = torch.cat(embeddings).numpy()
    labels = torch.cat(labels).numpy()
    if normalize_embeddings:
        logger.info("Normalizing embeddings...")
        # Normalize embeddings to unit length
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / (norms + 1e-8)  # add epsilon to avoid division by zero

    return embeddings, labels


@click.command()
@click.option("--model", default="UNI2", help="Name of the model to use for embeddings.")
@click.option(
    "--model-weights-path",
    default=None,
    help="Either a model name or a path to the model weights.",
)
@click.option(
    "--input-dir",
    required=True,
    type=click.Path(exists=True),
    help="Directory with ROI PNG images.",
)
@click.option(
    "--csv-metadata", required=True, type=click.Path(exists=True), help="Path to metadata CSV."
)
@click.option("--batch-size", default=32, show_default=True)
@click.option("--num-workers", default=0, show_default=True)
@click.option("--gpu-id", default=0, show_default=True)
@click.option("--label-column", default="superclass", show_default=True)
@click.option("--normalize-embeddings", is_flag=True, default=False, help="Normalize embeddings.")
@click.option(
    "--confusion-matrix-path",
    required=True,
    type=click.Path(),
    help="Path to save the confusion matrix plot.",
)
def evaluate(
    model,
    model_weights_path,
    input_dir,
    csv_metadata,
    batch_size,
    num_workers,
    gpu_id,
    label_column,
    normalize_embeddings,
    confusion_matrix_path,
):
    """
    Evaluate a trained model checkpoint on the LungHist700 test set.
    """
    device = get_device(gpu_id)
    logger.info(f"Using device: {device}")

    if label_column == "superclass":
        n_classes = len(SUPERCLASS_MAPPING)
    elif label_column == "class_name":
        n_classes = len(CLASS_MAPPING)
    else:
        raise ValueError(
            f"Unsupported label column: {label_column}. Use 'superclass' or 'class_name'."
        )

    if model in FOUNDATION_MODEL_NAMES:
        encoder, preprocess, embedding_dim, _ = load_foundation_model(model, device)
    elif model in model_cls_map and model_weights_path is not None:
        logger.info(f"Loading model weights from {model_weights_path}")
        encoder, preprocess = load_pretrained_encoder(model, model_weights_path, device=device)
        embedding_dim = 2048
    else:
        raise ValueError(
            f"Model {model} is not supported or model_weights_path is required for this model. \n"
            f"model_weights_path: {model_weights_path}."
            f"Foundation models: {FOUNDATION_MODEL_NAMES} \n"
            f"Custom models: {model_cls_map.keys()} \n"
        )

    encoder.eval()
    encoder.to(device)

    # Load data
    input_dir = Path(input_dir).resolve()
    image_paths = list(input_dir.glob("*.png"))
    metadata = pd.read_csv(csv_metadata).set_index("filename")

    train_loader, test_loader = get_dataloaders(
        image_paths=image_paths,
        metadata=metadata,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
        label_column=label_column,
    )

    # # Choose mapping
    # output_dim = len(CLASS_MAPPING) if label_column == "class_name" else len(SUPERCLASS_MAPPING)
    # metric = MulticlassAccuracy(num_classes=output_dim, average="macro").to(device)

    # Collect predictions and true labels
    train_embeddings, train_labels = compute_embeddings_on_tiles(
        encoder, train_loader, device, normalize_embeddings=normalize_embeddings
    )

    # Linear Probing (Logistic Regression)
    # λ = 100 * M / C, so C = 1 / λ
    l2_lambda = 100 / embedding_dim / n_classes
    C = 1.0 / l2_lambda
    lr = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        max_iter=1000,
        # class_weight="balanced",
    )
    knn = KNeighborsClassifier(n_neighbors=20)

    lr.fit(train_embeddings, train_labels)
    knn.fit(train_embeddings, train_labels)

    test_embeddings, y_true = compute_embeddings_on_tiles(
        encoder, test_loader, device, normalize_embeddings=normalize_embeddings
    )

    # Concatenate all batches
    y_pred_lr = lr.predict(test_embeddings)
    y_pred_knn = knn.predict(test_embeddings)

    logger.info(f"Saving confusion matrix to {confusion_matrix_path}")
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred_lr, display_labels=list(CLASS_MAPPING.keys())
    )
    Path(confusion_matrix_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(confusion_matrix_path)
    plt.close()

    stats = bootstrap_metrics(y_true, y_pred_lr, n_bootstraps=1000, average="macro")
    print("Number of test samples:", len(y_true))
    print("Number of classes:", n_classes)

    print("\n✅ Bootstrapped Metrics (95% CI) for linear:")
    for k, v in stats.items():
        print(f"{k:>10}: {v['mean']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")

    stats = bootstrap_metrics(y_true, y_pred_knn, n_bootstraps=1000, average="macro")

    print("\n✅ Bootstrapped Metrics (95% CI) for kNN:")
    for k, v in stats.items():
        print(f"{k:>10}: {v['mean']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")


if __name__ == "__main__":
    evaluate()
