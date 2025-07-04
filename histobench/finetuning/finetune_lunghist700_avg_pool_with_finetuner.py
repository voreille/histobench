from collections import Counter
import logging
from pathlib import Path

import click
from finetuning_scheduler import FinetuningScheduler
import lightning as L
from lightning.pytorch import seed_everything
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

from histobench.data.torch_datasets import LabelledTileOnTheFlyDataset
from histobench.models.foundation_models import FOUNDATION_MODEL_NAMES
from histobench.models.foundation_models import load_model as load_foundation_model
from histobench.models.load_models import load_pretrained_encoder
from histobench.utils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = Path(__file__).parents[2].resolve()

SUPERCLASS_MAPPING = {
    "nor": 0,
    "aca": 1,
    "scc": 2,
}

CLASS_MAPPING = {
    "nor": 0,
    "aca_bd": 1,
    "aca_md": 2,
    "aca_pd": 3,
    "scc_bd": 4,
    "scc_md": 5,
    "scc_pd": 6,
}


class FineTuneModelGAP(L.LightningModule):
    def __init__(self, encoder, input_dim, output_dim, lr, weight_decay, unfreeze_epoch=5):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Linear(input_dim, output_dim)
        self.lr = lr
        self.weight_decay = weight_decay
        self.unfreeze_epoch = unfreeze_epoch

        self.train_accuracy = MulticlassAccuracy(num_classes=output_dim, average="macro")
        self.val_accuracy = MulticlassAccuracy(num_classes=output_dim, average="macro")

        # Initially freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.save_hyperparameters(ignore=["encoder"])

    def forward(self, x):
        embeddings = self.compute_embeddings(x)
        logits = self.classifier(embeddings)
        return logits

    def compute_embeddings(self, x):
        all_embeddings = []
        for tile_bag in x:
            with (
                torch.no_grad()
                if not any(p.requires_grad for p in self.encoder.parameters())
                else torch.enable_grad()
            ):
                roi_embedding = self.encoder(tile_bag).mean(dim=0)

            all_embeddings.append(roi_embedding)
        return torch.stack(all_embeddings, dim=0)

    def training_step(self, batch, batch_idx):
        batch_of_bags, labels = batch
        batch_size = labels.size(0)
        logits = self(batch_of_bags)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log("train_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(
            "train_acc", self.train_accuracy(logits, labels), prog_bar=True, batch_size=batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        batch_size = labels.size(0)
        logits = self(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(
            "val_acc", self.val_accuracy(logits, labels), prog_bar=True, batch_size=batch_size
        )
        return loss

    def configure_optimizers(self):
        # At init, only classifier is trainable
        return torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )


def fine_tune_lunghist700_from_csv(
    model,
    csv_metadata,
    model_weights_path,
    input_dir,
    fold_csv_path,
    gpu_id,
    batch_size,
    num_workers,
    lr,
    weight_decay,
    max_epochs,
    checkpoint_dir,
    label_column="superclass",
    seed=42,
    fold=0,
):
    """
    Fine-tune a model using cross-validation splits loaded from a CSV file.
    """
    if seed is not None:
        seed_everything(seed + fold)

    input_dir = Path(input_dir).resolve()
    image_paths = list(input_dir.glob("*.png"))
    logger.info(f"Found {len(image_paths)} images in {input_dir}")
    metadata = pd.read_csv(csv_metadata).set_index("filename")

    # Load fold information from CSV
    fold_df = pd.read_csv(fold_csv_path)
    fold_df = fold_df[fold_df["fold"] == fold]

    train_image_paths = [
        input_dir / row["original_filename"]
        for _, row in fold_df[fold_df["split"] == "train"].iterrows()
    ]
    val_image_paths = [
        input_dir / row["original_filename"]
        for _, row in fold_df[fold_df["split"] == "validation"].iterrows()
    ]
    test_image_paths = [
        input_dir / row["original_filename"]
        for _, row in fold_df[fold_df["split"] == "test"].iterrows()
    ]

    device = get_device(gpu_id)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {model}")

    if model in FOUNDATION_MODEL_NAMES:
        encoder, preprocess, feature_dim, _ = load_foundation_model(
            model, device, apply_torch_scripting=False
        )
    elif model_weights_path is not None:
        logger.info(f"Loading model weights from {model_weights_path}")
        encoder, preprocess = load_pretrained_encoder(model, model_weights_path, device=device)
        feature_dim = 2048
    else:
        raise ValueError(
            f"Model {model} is not supported or model_weights_path is required for this model."
        )

    class_mapping = CLASS_MAPPING if label_column == "class_name" else SUPERCLASS_MAPPING

    train_labels = [
        class_mapping[metadata.loc[Path(path).stem, label_column]] for path in train_image_paths
    ]
    val_labels = [
        class_mapping[metadata.loc[Path(path).stem, label_column]] for path in val_image_paths
    ]
    test_labels = [
        class_mapping[metadata.loc[Path(path).stem, label_column]] for path in test_image_paths
    ]

    # Print class ratios for each split
    def print_class_ratios(labels, split_name):
        label_counts = Counter(labels)
        total_labels = len(labels)
        label_ratios = {label: count / total_labels for label, count in label_counts.items()}
        logger.info(f"Class ratios for {split_name}: {label_ratios}")

    print_class_ratios(train_labels, "train")
    print_class_ratios(val_labels, "validation")
    print_class_ratios(test_labels, "test")

    # Create datasets
    train_dataset = LabelledTileOnTheFlyDataset(
        roi_paths=train_image_paths, labels=train_labels, tile_transform=preprocess
    )
    val_dataset = LabelledTileOnTheFlyDataset(
        roi_paths=val_image_paths, labels=val_labels, tile_transform=preprocess
    )
    test_dataset = LabelledTileOnTheFlyDataset(
        roi_paths=test_image_paths, labels=test_labels, tile_transform=preprocess
    )

    # Create dataloaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
        collate_fn=LabelledTileOnTheFlyDataset.get_collate_fn_ragged(),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
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

    if label_column == "class_name":
        output_dim = len(CLASS_MAPPING)
    else:
        output_dim = len(SUPERCLASS_MAPPING)

    model = FineTuneModelGAP(
        encoder=encoder,
        input_dim=feature_dim,
        output_dim=output_dim,
        lr=lr,
        weight_decay=weight_decay,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    finetuning_callback = FinetuningScheduler(
        ft_schedule="/home/valentin/workspaces/histobench/configs/LungHist700_finetuning/FineTuneModel_ft_schedule.yaml"
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        precision="16-mixed",
        callbacks=[checkpoint_callback, finetuning_callback],
    )

    trainer.fit(model, train_dataloader, val_dataloader)


@click.command()
@click.option("--model", default="resnet50", help="Name of the model to fine-tune.")
@click.option(
    "--model-weights-path",
    default=None,
    help="Path to the model weights.",
)
@click.option(
    "--input-dir",
    default="data/LungHist700/LungHist700_10x",
    type=click.Path(exists=True),
    help="Directory containing ROI images.",
)
@click.option(
    "--csv-metadata",
    default="data/LungHist700/metadata.csv",
    type=click.Path(exists=True),
    help="Path to the CSV metadata file with patient information.",
)
@click.option("--gpu-id", default=0, show_default=True, help="GPU ID to use for computation.")
@click.option("--batch-size", default=32, show_default=True, help="Batch size for the dataloader.")
@click.option(
    "--num-workers", default=0, show_default=True, help="Number of workers for the dataloader."
)
@click.option("--lr", default=1e-3, show_default=True, help="Learning rate for fine-tuning.")
@click.option(
    "--weight-decay", default=1e-5, show_default=True, help="Weight decay for the optimizer."
)
@click.option("--max-epochs", default=20, show_default=True, help="Maximum number of epochs.")
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    type=click.Path(),
    show_default=True,
    help="Directory to save model checkpoints.",
)
@click.option("--label-column", default="superclass", show_default=True, help="Column for labels.")
@click.option(
    "--fold-csv-path",
    default=None,
    show_default=True,
    help="Path to the CSV with fold information.",
)
def main(
    model,
    model_weights_path,
    input_dir,
    csv_metadata,
    gpu_id,
    batch_size,
    num_workers,
    lr,
    weight_decay,
    max_epochs,
    checkpoint_dir,
    label_column,
    fold_csv_path,
):
    """
    Fine-tune a model on the LungHist700 dataset using an MLP classifier.
    """
    n_folds = 4
    for fold in range(n_folds):
        logger.info(f"Fine-tuning fold {fold + 1}/{n_folds}")
        checkpoint_dir_fold = Path(checkpoint_dir) / f"fold_{fold}"
        checkpoint_dir_fold.mkdir(parents=True, exist_ok=True)
        fine_tune_lunghist700_from_csv(
            model=model,
            csv_metadata=csv_metadata,
            model_weights_path=model_weights_path,
            input_dir=input_dir,
            gpu_id=gpu_id,
            batch_size=batch_size,
            num_workers=num_workers,
            lr=lr,
            weight_decay=weight_decay,
            max_epochs=max_epochs,
            checkpoint_dir=checkpoint_dir_fold,
            label_column=label_column,
            fold_csv_path=fold_csv_path,
            fold=fold,
        )
        # Clean CUDA memory after each fold
        torch.cuda.empty_cache()
        logger.info(f"CUDA memory cleaned after fold {fold + 1}/{n_folds}")

if __name__ == "__main__":
    main()
