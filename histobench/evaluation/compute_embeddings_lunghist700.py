import logging
from pathlib import Path

import click
import h5py
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from histobench.darya_code.get_model_inference import model_cls_map
from histobench.data.torch_datasets import ImageDataset, TileOnTheFlyDataset
from histobench.models.foundation_models import FOUNDATION_MODEL_NAMES
from histobench.models.foundation_models import load_model as load_foundation_model
from histobench.models.load_models import load_pretrained_encoder
from histobench.utils import get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

project_dir = Path(__file__).parents[2].resolve()


def compute_embeddings_whole_roi(model, dataloader, device="cuda"):
    model.to(device)
    model.eval()
    embeddings, image_paths = [], []

    with torch.inference_mode():
        for images, batch_image_paths in tqdm(dataloader, desc="Computing embeddings (whole ROI)"):
            embeddings.append(model(images.to(device)).cpu())
            image_paths.extend(batch_image_paths)

    return torch.cat(embeddings, dim=0).numpy(), np.array(image_paths)


def compute_embeddings_on_tiles(model, dataloader, device="cuda", overlap=False):
    model.to(device)
    model.eval()
    embeddings, image_paths = [], []

    with torch.inference_mode():
        for batch_of_bags, batch_image_paths in tqdm(
            dataloader, desc=f"Computing embeddings ({'overlap' if overlap else 'no overlap'})"
        ):
            # Each bag is a tensor of shape (num_tiles, C, H, W)
            mean_embeddings = torch.stack(
                [model(tiles.to(device)).mean(dim=0).cpu() for tiles in batch_of_bags], dim=0
            )
            embeddings.append(mean_embeddings)
            image_paths.extend(batch_image_paths)

    return torch.cat(embeddings, dim=0).numpy(), np.array(image_paths)


def save_embeddings(embeddings, image_paths, embeddings_path):
    if embeddings_path is None:
        raise ValueError("embeddings_path must be specified to save embeddings.")

    embeddings_path = Path(embeddings_path).with_suffix(".h5").resolve()
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(embeddings_path, "w") as f:
        f.create_dataset("embeddings", data=embeddings, compression="gzip")
        dt = h5py.string_dtype(encoding="utf-8")
        f.create_dataset("image_paths", data=image_paths.astype("S"), dtype=dt)
    print(f"Embeddings saved to {embeddings_path}")


def precrop_to_multiple_of_tile(image: Image.Image, tile_size: int) -> Image.Image:
    """Crop the image so that both dimensions are multiples of tile_size, centered."""
    width, height = image.size
    new_width = (width // tile_size) * tile_size
    new_height = (height // tile_size) * tile_size
    left = (width - new_width) // 2
    top = (height - new_height) // 2
    right = left + new_width
    bottom = top + new_height
    return image.crop((left, top, right, bottom))


@click.command()
@click.option("--model", default="UNI2", help="Name of the model to use for embeddings.")
@click.option(
    "--model-weights-path",
    default=None,
    help="Either a model name or a path to the model weights.",
)
@click.option(
    "--input-dir",
    default="data/LungHist700/LungHist700_10x",
    type=click.Path(exists=True),
    help="Directory containing ROI images.",
)
@click.option("--gpu-id", default=0, show_default=True, help="GPU ID to use for computation.")
@click.option(
    "--aggregation",
    type=click.Choice(["whole_roi", "tile_no_overlap", "tile_with_overlap"]),
    default="whole_roi",
    show_default=True,
    help="Aggregation method: 'whole_roi', 'tile_no_overlap', or 'tile_with_overlap'.",
)
@click.option(
    "--tile-size",
    default=224,
    show_default=True,
    help="Tile size for tiling strategies.",
)
@click.option("--batch-size", default=32, show_default=True, help="Batch size for the dataloader.")
@click.option(
    "--num-workers", default=0, show_default=True, help="Number of workers for the dataloader."
)
@click.option(
    "--embeddings-path",
    default=None,
    show_default=True,
    required=True,
    help="Path to save the embeddings (.h5).",
)
def main(
    model,
    model_weights_path,
    input_dir,
    gpu_id,
    aggregation,
    tile_size,
    batch_size,
    num_workers,
    embeddings_path,
):
    input_dir = Path(input_dir).resolve()
    image_paths = list(input_dir.glob("*.png"))
    embeddings_path = project_dir / embeddings_path if embeddings_path else None

    if embeddings_path.suffix != ".h5":
        logger.warning(
            f"Embeddings path should have .h5 suffix, changing {embeddings_path} to {embeddings_path.with_suffix('.h5')}"
        )
        embeddings_path = embeddings_path.with_suffix(".h5")

    logger.info(f"Found {len(image_paths)} images in {input_dir}")

    device = get_device(gpu_id)
    logger.info(f"Using device: {device}")

    logger.info(f"Loading model: {model}")

    if model in FOUNDATION_MODEL_NAMES:
        encoder, preprocess, _, _ = load_foundation_model(model, device)
    elif model in model_cls_map and model_weights_path is not None:
        encoder, preprocess = load_pretrained_encoder(model, model_weights_path, device=device)
    else:
        raise ValueError(
            f"Model {model} is not supported or model_weights_path is required for this model."
        )

    if aggregation == "whole_roi":
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
        embeddings, image_paths = compute_embeddings_whole_roi(encoder, dataloader, device=device)

    elif aggregation == "tile_no_overlap":
        # Pre-crop images to multiples of tile_size
        def roi_transform(image: Image.Image) -> Image.Image:
            return precrop_to_multiple_of_tile(image, tile_size)

        dataset = TileOnTheFlyDataset(
            roi_paths=image_paths,
            tile_transform=preprocess,
            roi_transform=roi_transform,
            tile_size=tile_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=TileOnTheFlyDataset.get_collate_fn_ragged(),
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        embeddings, image_paths = compute_embeddings_on_tiles(
            encoder, dataloader, device=device, overlap=False
        )

    elif aggregation == "tile_with_overlap":
        dataset = TileOnTheFlyDataset(
            roi_paths=image_paths,
            tile_transform=preprocess,
            roi_transform=None,
            tile_size=tile_size,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            collate_fn=TileOnTheFlyDataset.get_collate_fn_ragged(),
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
        embeddings, image_paths = compute_embeddings_on_tiles(
            encoder, dataloader, device=device, overlap=True
        )
    else:
        raise ValueError(
            f"Invalid aggregation method: {aggregation}. Choose from 'whole_roi', 'tile_no_overlap', or 'tile_with_overlap'."
        )

    embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    save_embeddings(embeddings, image_paths, embeddings_path)
    logger.info(f"Embeddings saved to {embeddings_path}")


if __name__ == "__main__":
    main()
