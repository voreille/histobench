from PIL import Image
from torch.utils.data import Dataset


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
