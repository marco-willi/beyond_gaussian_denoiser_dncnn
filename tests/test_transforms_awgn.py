import numpy as np
import pytest
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from gaussian_denoiser.transforms import AWGNOnlyTransform


def create_test_image(size=(100, 100), color=(255, 255, 255)):
    """Creates a plain white test image."""
    return Image.new("RGB", size, color)


def test_awgn_only_transform():
    # Create a test image
    image = create_test_image()

    # Initialize the transform
    min_variance = 10
    max_variance = 50
    transform = AWGNOnlyTransform(min_variance, max_variance)

    # Apply the transform
    noisy_image, noise_residual = transform(image)

    # Assertions to check the output types
    assert isinstance(noisy_image, Image.Image), "Output image should be a PIL Image"
    assert isinstance(noise_residual, torch.Tensor), "Noise residual should be a torch Tensor"

    # Convert images to tensors for comparison
    original_tensor = TF.to_tensor(image)
    noisy_tensor = TF.to_tensor(noisy_image)

    # Ensure the noise residual has the same shape as the input
    assert (
        noise_residual.shape == original_tensor.shape
    ), "Noise residual should have the same shape as the input image tensor"

    # Ensure noise_residual is difference of original and noisy image
    torch.testing.assert_close(
        original_tensor, (noisy_tensor - (noise_residual / 255.0))
    ), "Noise Residual does not match difference of image with noisy image"


if __name__ == "__main__":
    pytest.main()
