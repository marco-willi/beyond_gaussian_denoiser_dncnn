"""Define Datasets and Data Modules."""
import math
from typing import Callable, Iterable, Protocol

import lightning as L
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from icecream import ic
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .transforms import RandomRotationTransform
from .utils import patchify_coordinates


class NoiseCallable(Protocol):
    def __call__(self, image: Image.Image) -> tuple[Image.Image, np.ndarray]:
        """Apply noise to the input image.

        Args:
            image (PIL.Image.Image): The input image.

        Returns:
            tuple: A tuple containing the noisy image and the noise residual.
        """
        pass


class DenoisingDataModule(L.LightningDataModule):
    """Class to handle training, validation and test datasets for image denoising."""

    def __init__(
        self,
        ds_train: Iterable[Image.Image],
        ds_val: Iterable[Image.Image],
        ds_test: Iterable[Image.Image],
        batch_size: int,
        patch_size: int,
        train_noise: NoiseCallable,
        val_noise: NoiseCallable,
        max_patches_per_image: int = None,
    ):
        super().__init__()

        self.ds_train = ds_train
        self.ds_val = ds_val
        self.ds_test = ds_test

        self.max_patches_per_image = max_patches_per_image

        self.patch_size = patch_size
        self.batch_size = batch_size

        self.train_noise = train_noise
        self.val_noise = val_noise

        self.train_transform = transforms.Compose(
            [
                transforms.RandomCrop(self.patch_size),
                transforms.RandomHorizontalFlip(p=0.5),
                RandomRotationTransform(degrees=[0, 90, 180, 270]),
                transforms.ColorJitter(
                    brightness=(0.8, 1.2),
                    contrast=(0.8, 1.2),
                    saturation=(0.8, 1.2),
                    hue=(-0.1, 0.1),
                ),
            ]
        )

        self.test_transform = transforms.Compose([lambda x: x])

    def setup(self, stage: str):

        if stage == "fit":

            self.train_dataset = ImagePatchDenoiseDataset(
                ds=self.ds_train,
                transform=self.train_transform,
                patch_size=self.patch_size,
                noise_transform=self.train_noise,
                max_patches_per_image=self.max_patches_per_image,
            )

            self.val_dataset = ImagePatchDenoiseDataset(
                ds=self.ds_val,
                transform=self.test_transform,
                patch_size=self.patch_size,
                noise_transform=self.val_noise,
                max_patches_per_image=self.max_patches_per_image,
            )

        if stage == "test":

            self.test_dataset = ImagePatchDenoiseDataset(
                ds=self.ds_test,
                transform=self.test_transform,
                patch_size=self.patch_size,
                noise_transform=self.val_noise,
                max_patches_per_image=self.max_patches_per_image,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=6,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=6,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=6
        )


class ImagePatchDenoiseDataset(Dataset):
    """Crops Regular Patches from Images and adds Noise."""

    def __init__(
        self,
        ds: Iterable[Image.Image],
        transform: transforms.Compose,
        patch_size: int,
        noise_transform: NoiseCallable,
        max_patches_per_image: int = None,
    ):
        self.ds = ds
        self.transform = transform
        self.noise_transform = noise_transform
        self.patch_size = patch_size
        self.max_patches_per_image = max_patches_per_image

        self.rng = np.random.default_rng(123)

        # Prepare a list to hold all patches information
        self.patch_coordinates = []
        self.prepare_patches()
        ic(f"created {len(self.patch_coordinates)} patches")

    def prepare_patches(self):
        """Calculate Patches per Image."""
        for image_idx, image in enumerate(self.ds):
            image_tensor = TF.to_tensor(image)
            patches = patchify_coordinates(image_tensor, patch_size=self.patch_size)
            # add image index
            patches = [(image_idx, x, y) for x, y in patches]

            if self.max_patches_per_image:
                if self.max_patches_per_image > len(patches):
                    self.rng.shuffle(patches)
                    patches = patches[: self.max_patches_per_image]

            self.patch_coordinates.extend(patches)

    def __len__(self):
        return len(self.patch_coordinates)

    def __getitem__(self, idx):
        # Read image
        image_index, start_h, start_w = self.patch_coordinates[idx]
        image = self.ds[image_index]
        image = TF.to_tensor(image)

        # Extract Patch
        patch = image[:, start_h : start_h + self.patch_size, start_w : start_w + self.patch_size]
        patch = TF.to_pil_image(patch)
        patch = self.transform(patch)

        # Add Noise
        noisy_patch, noise_added = self.noise_transform(patch)
        noisy_patch = TF.to_tensor(noisy_patch)
        patch = TF.to_tensor(patch)
        noise_added /= 255.0

        return patch, noisy_patch, noise_added


class ImageDenoiseDataset(Dataset):
    """Adds Noise to Images."""

    def __init__(
        self,
        ds: Iterable[Image.Image],
        noise_transform: NoiseCallable,
        transform: Callable = lambda x: x,
    ):
        self.ds = ds
        self.transform = transform
        self.noise_transform = noise_transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        x = self.ds[idx]
        x = self.transform(x)

        # Add Noise
        noisy_x, noise_added = self.noise_transform(x)
        noisy_x = TF.to_tensor(noisy_x)
        x = TF.to_tensor(x)
        noise_added /= 255.0

        return x, noisy_x, noise_added
