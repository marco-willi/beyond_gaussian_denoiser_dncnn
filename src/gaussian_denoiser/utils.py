import math
from pathlib import Path

import torch
import torch.nn.functional as F


def patchify(image_tensor: torch.Tensor, patch_size: int) -> tuple[torch.Tensor, tuple[int, int]]:
    # Get the original height and width
    _, h, w = image_tensor.shape

    # Calculate padding needed to make dimensions a multiple of patch_size
    pad_h = (patch_size - h % patch_size) % patch_size
    pad_w = (patch_size - w % patch_size) % patch_size

    # Apply padding
    padded_image = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode="constant", value=0)

    # Unfold the padded image to extract patches
    patches = padded_image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)

    # Reshape the patches to the desired output format
    patches = patches.permute(1, 2, 0, 3, 4).reshape(
        -1, image_tensor.size(0), patch_size, patch_size
    )

    return patches, (pad_h, pad_w)


def depatchify(
    patches: torch.Tensor,
    original_size: tuple[int, int, int],
    patch_size: int,
    padding: tuple[int, int],
):
    channels, h, w = original_size
    pad_h, pad_w = padding

    # Calculate padded dimensions
    padded_h = h + pad_h
    padded_w = w + pad_w

    # Calculate the number of patches along height and width for the padded image
    num_patches_h = padded_h // patch_size
    num_patches_w = padded_w // patch_size

    # Reshape patches back to their grid form
    patches = patches.view(num_patches_h, num_patches_w, channels, patch_size, patch_size)
    patches = patches.permute(2, 0, 3, 1, 4).contiguous()
    patches = patches.view(channels, num_patches_h * patch_size, num_patches_w * patch_size)

    # Slice to remove padding and return the image to its original size
    reconstructed_image = patches[:, :h, :w]

    return reconstructed_image


def patchify_coordinates(image: torch.Tensor, patch_size: int) -> tuple[int, int]:
    _, h, w = image.shape

    num_patches_h = math.ceil(h / patch_size)
    num_patches_w = math.ceil(w / patch_size)

    patches = []

    for i in range(num_patches_h):
        for j in range(num_patches_w):
            start_h = i * patch_size
            start_w = j * patch_size
            if start_h + patch_size <= h and start_w + patch_size <= w:
                patches.append((start_h, start_w))
    return patches


def find_all_ckpt_files(directory: Path) -> list[Path]:
    # Find all files with .ckpt extension recursively
    ckpt_files = list(directory.rglob("*.ckpt"))
    return ckpt_files


def get_ckpt(ckpt_list: list[Path]):
    if ckpt_list:
        print("Found checkpoint files:")
        for file in ckpt_list:
            print(file)
    else:
        print("No checkpoint files found.")

    return ckpt_list[0]
