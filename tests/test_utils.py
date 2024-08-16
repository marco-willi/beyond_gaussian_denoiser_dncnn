import math

import pytest
import torch
import torch.nn.functional as F

from gaussian_denoiser import utils

# Assuming the functions patchify, depatchify, and patchify_coordinates are imported from the module


def test_patchify_basic():
    # Test with a simple image tensor
    image_tensor = torch.arange(16).view(1, 4, 4).float()  # 1x4x4 image
    patch_size = 2

    expected_patches = torch.tensor(
        [
            [[0.0, 1.0], [4.0, 5.0]],
            [[2.0, 3.0], [6.0, 7.0]],
            [[8.0, 9.0], [12.0, 13.0]],
            [[10.0, 11.0], [14.0, 15.0]],
        ]
    ).view(4, 1, 2, 2)
    expected_padding = (0, 0)

    patches, padding = utils.patchify(image_tensor, patch_size)

    assert torch.equal(patches, expected_patches)
    assert padding == expected_padding


def test_patchify_padding():
    # Test with an image tensor that requires padding
    image_tensor = torch.arange(15).view(1, 3, 5).float()  # 1x3x5 image
    patch_size = 2

    expected_patches = torch.tensor(
        [
            [[0.0, 1.0], [5.0, 6.0]],
            [[2.0, 3.0], [7.0, 8.0]],
            [[4.0, 0.0], [9.0, 0.0]],
            [[10.0, 11.0], [0.0, 0.0]],
            [[12.0, 13.0], [0.0, 0.0]],
            [[14.0, 0.0], [0.0, 0.0]],
        ]
    ).view(6, 1, 2, 2)
    expected_padding = (1, 1)

    patches, padding = utils.patchify(image_tensor, patch_size)

    assert torch.equal(patches, expected_patches)
    assert padding == expected_padding


def test_depatchify_basic():
    # Test with simple patches and no padding
    patches = torch.tensor(
        [
            [[0.0, 1.0], [4.0, 5.0]],
            [[2.0, 3.0], [6.0, 7.0]],
            [[8.0, 9.0], [12.0, 13.0]],
            [[10.0, 11.0], [14.0, 15.0]],
        ]
    ).view(4, 1, 2, 2)
    original_size = (1, 4, 4)
    patch_size = 2
    padding = (0, 0)

    expected_image = torch.arange(16).view(1, 4, 4).float()

    reconstructed_image = utils.depatchify(patches, original_size, patch_size, padding)

    assert torch.equal(reconstructed_image, expected_image)


def test_depatchify_padding():
    # Test with patches and padding
    patches = torch.tensor(
        [
            [[0.0, 1.0], [5.0, 6.0]],
            [[2.0, 3.0], [7.0, 8.0]],
            [[4.0, 0.0], [9.0, 0.0]],
            [[10.0, 11.0], [0.0, 0.0]],
            [[12.0, 13.0], [0.0, 0.0]],
            [[14.0, 0.0], [0.0, 0.0]],
        ]
    ).view(6, 1, 2, 2)
    original_size = (1, 3, 5)
    patch_size = 2
    padding = (1, 1)

    expected_image = torch.arange(15).view(1, 3, 5).float()

    reconstructed_image = utils.depatchify(patches, original_size, patch_size, padding)

    assert torch.equal(reconstructed_image, expected_image)


def test_patchify_coordinates_basic():
    # Test with a simple image
    image_tensor = torch.arange(16).view(1, 4, 4).float()  # 1x4x4 image
    patch_size = 2

    expected_coordinates = [(0, 0), (0, 2), (2, 0), (2, 2)]

    coordinates = utils.patchify_coordinates(image_tensor, patch_size)

    assert coordinates == expected_coordinates


def test_patchify_coordinates_padding():
    # Test with an image tensor that is not a multiple of the patch size
    # the image is not padded, only coordinates for full patches are
    # considered
    image_tensor = torch.arange(15).view(1, 3, 5).float()  # 1x3x5 image
    patch_size = 2

    expected_coordinates = [(0, 0), (0, 2)]

    coordinates = utils.patchify_coordinates(image_tensor, patch_size)

    assert coordinates == expected_coordinates


def test_patchify_depatchify_consistency():
    images = [
        torch.arange(16).view(1, 4, 4).float(),  # 1x4x4 grayscale image
        torch.arange(15).view(1, 3, 5).float(),  # 1x3x5 grayscale image
        torch.arange(24).view(1, 4, 6).float(),  # 1x4x6 grayscale image
        torch.rand(1, 7, 7),  # 1x7x7 grayscale random image
        torch.rand(3, 8, 8),  # 3x8x8 color image
        torch.arange(27).view(3, 3, 3).float(),  # 3x3x3 color image
        torch.rand(3, 5, 7),  # 3x5x7 color image
        torch.rand(3, 10, 10),  # 3x10x10 color image
    ]
    patch_sizes = [2, 2, 3, 3, 4, 2, 3, 5]

    for image_tensor, patch_size in zip(images, patch_sizes):
        # Apply patchify
        patches, padding = utils.patchify(image_tensor, patch_size)

        # Get the original size
        original_size = image_tensor.shape

        # Reconstruct the image using depatchify
        reconstructed_image = utils.depatchify(patches, original_size, patch_size, padding)

        # Ensure the reconstructed image is the same as the original
        assert torch.equal(
            reconstructed_image, image_tensor
        ), f"Inconsistency found for image of shape {original_size} with patch size {patch_size}"


if __name__ == "__main__":
    pytest.main()
