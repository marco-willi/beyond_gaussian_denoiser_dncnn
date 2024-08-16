"""Define Data Transformations."""
import io
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


class RandomRotationTransform:
    """Rotate by one of the given degrees."""

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, x):
        angle = random.choice(self.degrees)
        return TF.rotate(x, angle)


class AWGNOnlyTransform:
    def __init__(self, min_variance, max_variance):
        self.min_variance = min_variance
        self.max_variance = max_variance

    def __call__(self, image: Image.Image) -> tuple[Image.Image, torch.Tensor]:

        # converts to range [0, 1]
        x = TF.to_tensor(image)

        sampled_variance = np.random.uniform(low=self.min_variance, high=self.max_variance)

        noise_sampled = torch.randn_like(x) * (sampled_variance / 255.0)

        x_with_noise = torch.clip(x + noise_sampled, 0, 1.0)

        image_with_noise = TF.to_pil_image(x_with_noise)

        noise_added = (TF.to_tensor(image_with_noise) - x) * 255.0

        return image_with_noise, noise_added


class DownUpsampleTransform:
    def __init__(self, factors=[2, 3, 4]):
        self.factors = factors

    def __call__(self, image: Image.Image) -> tuple[Image.Image, torch.Tensor]:
        """Downsample and upsample an image by a random factor.

        Args:
            image (PIL.Image): The input image.
            factors (list of int): The list of factors to choose from.

        Returns:
            PIL.Image: The transformed image.
            torch.Tensor The added noise (image_with_noise - image)
        """
        factor = random.choice(self.factors)
        width, height = image.size
        downsampled_size = (width // factor, height // factor)
        downsampled_image = image.resize(downsampled_size, Image.BICUBIC)
        upsampled_image = downsampled_image.resize((width, height), Image.BICUBIC)

        noise_added = (TF.to_tensor(upsampled_image) - TF.to_tensor(image)) * 255.0
        return upsampled_image, noise_added


class JPEGTransform:
    def __init__(self, min_quality=5, max_quality=99):
        self.min_quality = min_quality
        self.max_quality = max_quality

    def __call__(self, image: Image.Image) -> tuple[Image.Image, np.ndarray]:

        quality = random.randint(self.min_quality, self.max_quality)

        # Save image to a buffer with JPEG compression
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)

        # Load the compressed image
        compressed_image = Image.open(buffer)

        noise_added = (TF.to_tensor(compressed_image) - TF.to_tensor(image)) * 255.0
        return compressed_image, noise_added


class CombinedTransform:
    def __init__(
        self, min_variance, max_variance, factors=[2, 3, 4], min_quality=5, max_quality=99
    ):
        self.transforms = [
            AWGNOnlyTransform(min_variance, max_variance),
            DownUpsampleTransform(factors),
            JPEGTransform(min_quality, max_quality),
        ]

    def __call__(self, image: Image.Image) -> tuple[Image.Image, np.ndarray]:
        transform = random.choice(self.transforms)
        return transform(image)
