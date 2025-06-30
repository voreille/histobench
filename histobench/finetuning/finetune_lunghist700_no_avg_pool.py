import logging
from pathlib import Path

import click
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torch
from torch.utils.data import DataLoader

from histobench.data.torch_datasets import LabelledImageDataset
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


def get_train_val_test_ids(resolution):
    # fmt: off
    if resolution == '20x':
        train_ids = [2, 3, 4, 5, 7, 8, 12, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 28, 29, 30, 33, 36, 37, 38, 39, 41, 42, 45]
        val_ids = [1, 6, 27, 32, 44]
        test_ids = [9, 13, 31, 40]
    elif resolution == '40x':
        train_ids = [2, 6, 8, 9, 10, 12, 13, 14, 16, 18, 19, 21, 22, 24, 28, 29, 31, 33, 34, 35, 36, 38, 40, 44]
        val_ids = [1, 4, 17, 26, 30, 37, 45]
        test_ids = [11, 15, 20, 25, 32, 43]
    elif resolution == 'all':
        train_ids_20x = [2, 3, 4, 5, 7, 8, 12, 14, 15, 16, 17, 18, 20, 21, 23, 24, 25, 26, 28, 29, 30, 33, 36, 37, 38, 39, 41, 42, 45]
        val_ids_20x = [1, 6, 27, 32, 44]
        test_ids_20x = [9, 13, 31, 40]

        train_ids_40x = [2, 6, 8, 9, 10, 12, 13, 14, 16, 18, 19, 21, 22, 24, 28, 29, 31, 33, 34, 35, 36, 38, 40, 44]
        val_ids_40x = [1, 4, 17, 26, 30, 37, 45]
        test_ids_40x = [11, 15, 20, 25, 32, 43]

        train_ids = list(set(train_ids_20x + train_ids_40x))
        val_ids = list(set(val_ids_20x + val_ids_40x))
        test_ids = list(set(test_ids_20x + test_ids_40x))
    else:
        raise ValueError(f"Unsupported resolution: {resolution}")
    # fmt: on
    return train_ids, val_ids, test_ids


class FineTuneModel(pl.LightningModule):
    def __init__(self, encoder, input_dim, output_dim, lr, weight_decay):
        super().__init__()
        self.encoder = encoder
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(input_dim, output_dim),
        )
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        features = self.encoder(x)
        return self.classifier(features)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        logits = self(images)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


def get_dataloaders(
    image_paths, metadata, preprocess, batch_size, num_workers, label_column="superclass"
):
    class_mapping = CLASS_MAPPING if label_column == "class_name" else SUPERCLASS_MAPPING
    train_ids, val_ids, test_ids = get_train_val_test_ids("all")
    train_image_paths = [
        path for path in image_paths if metadata.loc[path.name, "patient_id"] in train_ids
    ]
    val_image_paths = [
        path for path in image_paths if metadata.loc[path.name, "patient_id"] in val_ids
    ]
    test_image_paths = [
        path for path in image_paths if metadata.loc[path.name, "patient_id"] in test_ids
    ]
    train_labels = metadata.loc[train_image_paths, label_column].values
    val_labels = metadata.loc[val_image_paths, label_column].values
    test_labels = metadata.loc[test_image_paths, label_column].values

    train_labels = [class_mapping[label] for label in train_labels]
    val_labels = [class_mapping[label] for label in val_labels]
    test_labels = [class_mapping[label] for label in test_labels]

    train_dataset = LabelledImageDataset(
        image_paths=train_image_paths, labels=train_labels, transform=preprocess
    )
    val_dataset = LabelledImageDataset(
        image_paths=val_image_paths, labels=val_labels, transform=preprocess
    )
    test_dataset = LabelledImageDataset(
        image_paths=test_image_paths, labels=test_labels, transform=preprocess
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
    )
    return train_dataloader, val_dataloader, test_dataloader


def fine_tune_lunghist700(
    model,
    csv_metadata,
    model_weights_path,
    input_dir,
    gpu_id,
    batch_size,
    num_workers,
    hidden_dim,
    lr,
    weight_decay,
    max_epochs,
    output_dim,
    checkpoint_dir,
):
    input_dir = Path(input_dir).resolve()
    image_paths = list(input_dir.glob("*.png"))
    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    device = get_device(gpu_id)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {model}")

    if model in FOUNDATION_MODEL_NAMES:
        encoder, preprocess, _, _ = load_foundation_model(model, device)
    elif model_weights_path is not None:
        logger.info(f"Loading model weights from {model_weights_path}")
        encoder, preprocess = load_pretrained_encoder(model, model_weights_path, device=device)
    else:
        raise ValueError(
            f"Model {model} is not supported or model_weights_path is required for this model."
        )

    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(
        image_paths=image_paths,
        metadata=csv_metadata,
        preprocess=preprocess,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    model = FineTuneModel(
        encoder=encoder,
        input_dim=encoder.output_dim,
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
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=[gpu_id],
        precision="16-mixed",
        check_val_every_n_epoch=10,
        callbacks=[checkpoint_callback, early_stopping_callback],
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
@click.option("--hidden-dim", default=128, show_default=True, help="Hidden dimension for MLP.")
@click.option("--lr", default=1e-3, show_default=True, help="Learning rate for fine-tuning.")
@click.option(
    "--weight-decay", default=1e-5, show_default=True, help="Weight decay for the optimizer."
)
@click.option("--max-epochs", default=20, show_default=True, help="Maximum number of epochs.")
@click.option("--output-dim", default=2, show_default=True, help="Output dimension for MLP.")
@click.option(
    "--checkpoint-dir",
    default="checkpoints",
    type=click.Path(),
    show_default=True,
    help="Directory to save model checkpoints.",
)
def main(
    model,
    model_weights_path,
    input_dir,
    csv_metadata,
    gpu_id,
    batch_size,
    num_workers,
    hidden_dim,
    lr,
    weight_decay,
    max_epochs,
    output_dim,
    checkpoint_dir,
):
    """
    Fine-tune a model on the LungHist700 dataset using an MLP classifier.
    """
    fine_tune_lunghist700(
        model=model,
        csv_metadata=csv_metadata,
        model_weights_path=model_weights_path,
        input_dir=input_dir,
        gpu_id=gpu_id,
        batch_size=batch_size,
        num_workers=num_workers,
        hidden_dim=hidden_dim,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        output_dim=output_dim,
        checkpoint_dir=checkpoint_dir,
    )


if __name__ == "__main__":
    main()
