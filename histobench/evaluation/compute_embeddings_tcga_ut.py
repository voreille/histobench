import logging
from pathlib import Path

import click
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from histobench.darya_code.get_model_inference import model_cls_map
from histobench.data.torch_datasets import ImageDataset
from histobench.models.foundation_models import FOUNDATION_MODEL_NAMES
from histobench.models.foundation_models import load_model as load_foundation_model
from histobench.models.load_models import load_pretrained_encoder
from histobench.utils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = Path(__file__).parents[2].resolve()


def compute_embeddings(model, dataloader, device="cuda"):
    """Compute embeddings dynamically for the given model."""

    model.to(device)
    model.eval()
    embeddings, image_paths = [], []

    with torch.inference_mode():
        for images, batch_image_paths in tqdm(dataloader, desc="Computing embeddings"):
            embeddings.append(model(images.to(device)).cpu())
            image_paths.extend(batch_image_paths)

    return torch.cat(embeddings, dim=0).numpy(), np.array(image_paths)


def save_embeddings(embeddings, image_paths, embeddings_path):
    """Save embeddings and image paths to an HDF5 file."""
    if embeddings_path is None:
        raise ValueError("embeddings_path must be specified to save embeddings.")

    embeddings_path = Path(embeddings_path).with_suffix(".h5").resolve()
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(embeddings_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, compression="gzip")
        # Save image paths as fixed-length ASCII strings
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("image_paths", data=image_paths.astype("S"), dtype=dt)
    print(f"Embeddings saved to {embeddings_path}")


def load_embeddings(embeddings_path):
    """Load embeddings and image paths from an HDF5 file."""
    embeddings_path = Path(embeddings_path).resolve()
    with h5py.File(embeddings_path, "r") as f:
        embeddings = f["embeddings"][:]
        image_paths = f["image_paths"][:].astype(str)
    return embeddings, image_paths


@click.command()
@click.option("--model", default="UNI2", help="Name of the model to use for embeddings.")
@click.option(
    "--model-weights-path",
    default=None,
    help="Either a model name or a path to the model weights.",
)
@click.option("--magnification-key", default=5, help="Magnification key for the model")
@click.option(
    "--tcga-ut-dir",
    default="data/tcga-ut",
    type=click.Path(exists=True),
    help="Magnification key for the model",
)
@click.option("--magnification-key", default=5, help="Magnification key for the model")
@click.option("--gpu-id", default=0, show_default=True, help="GPU ID to use for computation.")
@click.option(
    "--batch-size", default=256, show_default=True, help="Batch size for the dataloader."
)
@click.option(
    "--num-workers", default=0, show_default=True, help="Number of workers for the dataloader."
)
@click.option(
    "--embeddings-path",
    default=None,
    show_default=True,
    required=True,
    help="Name of the model to use.",
)
def main(
    model,
    model_weights_path,
    magnification_key,
    tcga_ut_dir,
    gpu_id,
    batch_size,
    num_workers,
    embeddings_path,
):
    """Simple CLI program to greet someone"""
    tcga_ut_rootdir = Path(tcga_ut_dir).resolve()
    image_paths = list(tcga_ut_rootdir.rglob(f"*/{magnification_key}/*/*.jpg"))
    embeddings_path = project_dir / embeddings_path if embeddings_path else None

    if embeddings_path.suffix != ".h5":
        logger.warning(
            f"Embeddings path should have .h5 suffix, changing {embeddings_path} to {embeddings_path.with_suffix('.h5')}"
        )
        embeddings_path = embeddings_path.with_suffix(".h5")

    logger.info(f"Found {len(image_paths)} images in {tcga_ut_rootdir}")

    device = get_device(gpu_id)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {model}")

    if model in FOUNDATION_MODEL_NAMES:
        encoder, preprocess, _, _ = load_foundation_model(model, get_device(gpu_id))

    elif model in model_cls_map and model_weights_path is not None:
        encoder, preprocess = load_pretrained_encoder(model, model_weights_path, device=device)

    else:
        raise ValueError(
            f"Model {model} is not supported or model_weights_path is required for this model."
        )

    dataset = ImageDataset(
        image_paths=image_paths,
        transform=preprocess,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=False,
        prefetch_factor=4 if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    embeddings, image_paths = compute_embeddings(encoder, dataloader, device=device)

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(embeddings, image_paths, embeddings_path)
    logger.info(f"Embeddings saved to {embeddings_path}")


if __name__ == "__main__":
    main()
