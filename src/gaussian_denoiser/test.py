import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchmetrics import MeanSquaredError, image
from tqdm import tqdm

from gaussian_denoiser import dataset


def load_exported_model(model_path, device="cpu"):
    assert Path(model_path).exists(), f"Exported model path {model_path} does not exist."
    model = torch.jit.load(model_path)
    model = model.to(device).eval()
    return model


def evaluate_exported_model(
    exported_model: torch.nn.Module,
    test_data_original_dir: str,
    test_data_noisy_dir: str,
    device="cpu",
):

    assert Path(
        test_data_original_dir
    ).exists(), f"test_data_original_dir: {test_data_original_dir} does not exist"
    assert Path(
        test_data_noisy_dir
    ).exists(), f"test_data_noisy_dir: {test_data_noisy_dir} does not exist"

    # Load dataset
    ds_precomputed = dataset.PreComputedTestDataset(test_data_original_dir, test_data_noisy_dir)
    dl_precomputed = DataLoader(ds_precomputed, batch_size=1, shuffle=False, num_workers=1)

    print(f"Found {len(ds_precomputed)} test images")

    # Initialize metrics
    psnr = image.PeakSignalNoiseRatio((0, 1), dim=(1, 2, 3), reduction="elementwise_mean")
    ssim = image.StructuralSimilarityIndexMeasure(data_range=1.0)
    mssim = image.MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0)
    mse = MeanSquaredError()

    with torch.no_grad():
        for original_image, noisy_image in tqdm(dl_precomputed):
            noisy_image = noisy_image.to(device)
            noise_estimate = exported_model(noisy_image)
            denoised_image = noisy_image - noise_estimate

            original_image = original_image.to("cpu")
            denoised_image = torch.clip(denoised_image, 0, 1.0).to("cpu")

            # Calculate metrics
            psnr(denoised_image, original_image)
            ssim(denoised_image, original_image)
            mse(denoised_image, original_image)
            mssim(denoised_image, original_image)

    # Output results
    print("Evaluation Results:")
    print(f"Test Noise-Free: {test_data_original_dir}")
    print(f"Test Noise-Free: {test_data_noisy_dir}")
    print(f"PSNR: {psnr.compute():.4f}")
    print(f"SSIM: {ssim.compute():.4f}")
    print(f"MSSIM: {mssim.compute():.4f}")
    print(f"MSE: {mse.compute():.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate an already exported denoising model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the exported model file (TorchScript).",
    )
    parser.add_argument(
        "--test_data_original_dir",
        type=str,
        required=True,
        help="Path to the test dataset directory.",
    )
    parser.add_argument(
        "--test_data_noisy_dir",
        type=str,
        required=True,
        help="Path to the noisy test dataset directory.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the evaluation on ('cpu' or 'cuda').",
    )

    args = parser.parse_args()

    # Load model
    model = load_exported_model(args.model_path, args.device)

    # Evaluate the model
    evaluate_exported_model(
        model, args.test_data_original_dir, args.test_data_noisy_dir, args.device
    )
