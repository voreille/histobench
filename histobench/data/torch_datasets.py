import math

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform=None):
        self.image_paths = [str(path.resolve()) for path in image_paths]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Load as PIL image

        if self.transform:
            image = self.transform(image)  # Apply augmentation

        return image, image_path


class LabelledImageDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = [str(path.resolve()) for path in image_paths]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")  # Load as PIL image

        if self.transform:
            image = self.transform(image)  # Apply augmentation

        return image, torch.tensor(self.labels[idx], dtype=torch.long)


class TileOnTheFlyDataset(torch.utils.data.Dataset):
    def __init__(self, roi_paths, tile_transform=None, roi_transform=None, tile_size=224):
        self.roi_paths = [str(path.resolve()) for path in roi_paths]
        self.tile_transform = tile_transform
        self.roi_transform = roi_transform
        self.tile_size = tile_size

    def __len__(self):
        return len(self.roi_paths)

    @staticmethod
    def get_collate_fn_ragged():
        def collate_fn_ragged(batch):
            tile_bags, image_paths = zip(*batch)
            return list(tile_bags), list(image_paths)

        return collate_fn_ragged

    def calculate_stride(self, image_dim):
        """Calculate stride to align tiles symmetrically with borders."""
        if image_dim <= self.tile_size:
            raise ValueError("Image dimension must be larger than tile size")

        n_tiles = math.ceil(image_dim / self.tile_size)
        if n_tiles == 1:
            return 0  # Single tile, no stride needed
        total_stride_space = image_dim - self.tile_size * n_tiles
        stride = self.tile_size + total_stride_space // (n_tiles - 1)
        return int(stride)

    def tile_image(self, image: Image.Image) -> torch.Tensor:
        """
        Tile a PIL image into patches of shape (N, C, tile_size, tile_size).

        Args:
            image (PIL.Image): Input image of shape (H, W, 3)

        Returns:
            torch.Tensor: Tensor of tiles (num_tiles, C, tile_size, tile_size)
        """
        # Convert PIL image to tensor: [C, H, W]
        image_tensor = TF.to_tensor(image)  # range [0, 1]
        C, H, W = image_tensor.shape

        # Compute symmetric stride
        stride_y = self.calculate_stride(H)
        stride_x = self.calculate_stride(W)

        # Tiling with unfold
        tiles = image_tensor.unfold(1, self.tile_size, stride_y).unfold(
            2, self.tile_size, stride_x
        )
        # shape: [C, n_tiles_y, n_tiles_x, tile_size, tile_size]

        tiles = tiles.permute(1, 2, 0, 3, 4).contiguous()
        # shape: [n_tiles_y, n_tiles_x, C, tile_size, tile_size]

        tiles = tiles.view(-1, C, self.tile_size, self.tile_size)
        # shape: [num_tiles, C, tile_size, tile_size]

        return tiles

    def __getitem__(self, idx):
        image_path = self.roi_paths[idx]

        # Load image as Tensor [C, H, W]
        image = Image.open(image_path).convert("RGB")  # Load as PIL image

        if self.roi_transform:
            image = self.roi_transform(image)

        tiles = self.tile_image(image)
        if self.tile_transform:
            tiles = torch.stack(
                [self.tile_transform(TF.to_pil_image(tile)) for tile in tiles], dim=0
            )

        return tiles, image_path
