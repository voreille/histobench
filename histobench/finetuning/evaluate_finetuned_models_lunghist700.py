import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch

from histobench.finetuning.finetune_lunghist700_avg_pool_with_finetuner import (
    FineTuneModelGAP,
    get_dataloaders,
)
from histobench.models.foundation_models import load_model as load_foundation_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


@click.command()
@click.option(
    "--checkpoint-path",
    required=True,
    type=click.Path(exists=True),
    help="Path to the trained model checkpoint (.ckpt).",
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
def evaluate(
    checkpoint_path,
    input_dir,
    csv_metadata,
    batch_size,
    num_workers,
    gpu_id,
    label_column,
):
    """
    Evaluate a trained model checkpoint on the LungHist700 test set.
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model directly from checkpoint
    encoder, preprocess, feature_dim, _ = load_foundation_model("resnet50", device)

    initial_encoder_state = {k: v.clone() for k, v in encoder.state_dict().items()}

    model = FineTuneModelGAP.load_from_checkpoint(
        checkpoint_path, encoder=encoder, map_location=device
    )
    model.eval()
    model.to(device)

    # Check if encoder weights have changed
    changed_weights = []
    for k, v in model.encoder.state_dict().items():
        if not torch.equal(initial_encoder_state[k], v):
            changed_weights.append(k)

    if changed_weights:
        logger.info("Encoder weights have changed after loading the checkpoint.")
    else:
        logger.info("Encoder weights remain unchanged after loading the checkpoint.")

    # Load data
    input_dir = Path(input_dir).resolve()
    image_paths = list(input_dir.glob("*.png"))
    metadata = pd.read_csv(csv_metadata).set_index("filename")

    _, _, test_loader = get_dataloaders(
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
    all_preds = []
    all_labels = []
    all_embeddings = []

    with torch.inference_mode():
        for tile_bags, labels in test_loader:
            labels.to(device)
            tile_bags = [bag.to(device) for bag in tile_bags]
            embeddings = model.compute_embeddings(tile_bags)
            all_embeddings.append(embeddings.cpu())
            logits = model.classifier(embeddings)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    # Concatenate all batches
    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()

    stats = bootstrap_metrics(y_true, y_pred, n_bootstraps=1000, average="macro")
    print("Number of test samples:", len(y_true))

    print("\n✅ Bootstrapped Metrics (95% CI):")
    for k, v in stats.items():
        print(f"{k:>10}: {v['mean']:.4f} [{v['ci_lower']:.4f}, {v['ci_upper']:.4f}]")


if __name__ == "__main__":
    evaluate()
