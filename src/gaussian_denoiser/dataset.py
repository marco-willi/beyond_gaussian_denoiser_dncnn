"""Define Datasets."""
from pathlib import Path
from typing import Callable

import datasets
import torchvision.transforms as transforms
from icecream import ic
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

IMAGE_EXT = ("*.png", "*.jpg", "*.jpeg")


class ImageFolderDataset(Dataset):
    """Dataset which searches images from a folder."""

    def __init__(self, path: str, cache_images: bool = False, transform: Callable | None = None):
        self.path = Path(path)
        self.cache_images = cache_images
        self.transform = transform

        # Find all images in the directory and subdirectories
        self.image_files = []
        for ext in IMAGE_EXT:
            self.image_files.extend(self.path.rglob(ext))

        # Cache images
        if cache_images:
            self.images = {
                image_path: Image.open(image_path).convert("RGB")
                for image_path in self.image_files
            }

    def read_image(self, image_path: str) -> Image:
        if self.cache_images:
            return self.images[image_path]
        else:
            return Image.open(image_path).convert("RGB")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        image = self.read_image(image_path)
        if self.transform:
            image = self.transform(image)
        return image


class HFDataset(Dataset):
    def __init__(self, path: str, **kwargs):
        self.path = path
        self.ds = datasets.load_from_disk(path)
        ic(f"Loaded HFDataset with: {len(self.ds)} obs")

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index) -> Image.Image:
        return self.ds[index]["image"]


class PreComputedTestDataset(Dataset):
    """Dataset with pre-computed noisy images."""

    def __init__(
        self,
        path_original: str,
        path_noisy: str,
        transform: Callable | None = lambda x: TF.to_tensor(x),
    ):

        self.path_original = Path(path_original)
        self.path_noisy = Path(path_noisy)
        self.transform = transform

        # Find all images in the original and noisy directories
        self.original_images = self._find_images(self.path_original)
        self.noisy_images = self._find_images(self.path_noisy)

        # Match images by name without postfix
        self.image_pairs = self._match_images(self.original_images, self.noisy_images)

    def _find_images(self, path: Path) -> dict:
        """Find all images in the directory and subdirectories, returning a dictionary with image
        names as keys."""
        image_dict = {}
        for ext in IMAGE_EXT:
            for image_path in path.rglob(ext):
                image_name = image_path.stem.split(".")[0]  # Remove the extension
                image_dict[image_name] = image_path
        return image_dict

    def _match_images(self, originals: dict, noisys: dict) -> list[tuple[Path, Path]]:
        """Match images by name without postfix, return a list of tuples."""
        matched_pairs = []
        for name, original_path in originals.items():
            if name in noisys:
                matched_pairs.append((original_path, noisys[name]))
        return matched_pairs

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx: int) -> tuple[Image.Image, Image.Image]:
        original_path, noisy_path = self.image_pairs[idx]
        original_image = Image.open(original_path).convert("RGB")
        noisy_image = Image.open(noisy_path).convert("RGB")

        if self.transform:
            original_image = self.transform(original_image)
            noisy_image = self.transform(noisy_image)

        return original_image, noisy_image
