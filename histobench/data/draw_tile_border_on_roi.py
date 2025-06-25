from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from histobench.data.torch_datasets import TileOnTheFlyDataset
from histobench.evaluation.compute_embeddings_lunghist700 import precrop_to_multiple_of_tile


def draw_tiles_on_image(image_path, output_path, tile_size=224, precrop=False, save_tiles=True):
    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Load image
    image = Image.open(image_path).convert("RGB")

    if precrop:
        image = precrop_to_multiple_of_tile(image, tile_size)
    W, H = image.size

    # Instantiate dataset and get strides
    dataset = TileOnTheFlyDataset([Path(image_path)], tile_size=tile_size)
    stride_y = dataset.calculate_stride(H)
    stride_x = dataset.calculate_stride(W)

    # Define colors for visualization
    colors = ["red", "green", "blue", "yellow", "magenta", "cyan"]

    # Draw rectangles
    draw = ImageDraw.Draw(image)
    idx = 0
    for top in range(0, H - tile_size + 1, stride_y):
        for left in range(0, W - tile_size + 1, stride_x):
            color = colors[idx % len(colors)]
            draw.rectangle(
                [left, top, left + tile_size - 1, top + tile_size - 1], outline=color, width=2
            )
            idx += 1

    # Save tiles using DataLoader iterator
    if save_tiles:
        tiles_dir = output_path.parent / (output_path.stem + "_tiles")
        tiles_dir.mkdir(parents=True, exist_ok=True)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        for i, tile in enumerate(dataloader):
            # tile is a batch of shape (1, C, H, W), convert to PIL Image
            tiles = tile[0].squeeze(0).numpy()  # remove batch dimension
            for j in range(tiles.shape[0]):
                tile_img = tiles[j, ...]
                tile_img = np.transpose(tile_img, (1, 2, 0))
                tile_pil = Image.fromarray((tile_img * 255).astype("uint8"))
                tile_pil.save(tiles_dir / f"tile_{j:04d}.png")
        print(f"Saved {i + 1} tiles to {tiles_dir}")

    image.save(output_path)
    print(f"Saved annotated image with {idx} tiles to {output_path}")


if __name__ == "__main__":
    roi_path = (
        "/home/valentin/workspaces/histobench/data/LungHist700/LungHist700_10x/aca_bd_40x_11.png"
    )
    output_path = "/home/valentin/workspaces/histobench/data/outlines/LungHist700_10x/aca_bd_40x_11_cropped.png"
    tile_size = 224
    precrop = False  # Whether to crop image to multiple of tile size

    draw_tiles_on_image(roi_path, output_path, tile_size, precrop=precrop)
