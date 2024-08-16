import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from gaussian_denoiser import dataset


def load_exported_model(model_path: str, device: str = "cpu"):
    assert Path(model_path).exists(), f"Exported model path {model_path} does not exist."
    model = torch.jit.load(model_path, map_location=device)
    model = model.to(device).eval()
    return model


def denoise_image(
    exported_model: torch.nn.Module, image_tensor: torch.Tensor, device: str = "cpu"
):
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        noise_estimate = exported_model(image_tensor)
        print(f"Noise device: {noise_estimate.device}")
        print(f"Image Tensor device: {image_tensor.device}")
        denoised_image = torch.clip(image_tensor - noise_estimate, 0, 1.0).squeeze(0).cpu()
    return denoised_image


def save_denoised_image(denoised_image: torch.Tensor, output_image_path: str):
    denoised_pil_image = TF.to_pil_image(denoised_image)
    denoised_pil_image.save(output_image_path)
    print(f"Denoised image saved to {output_image_path}")


def process_single_image(
    exported_model: torch.nn.Module,
    input_image_path: str,
    output_image_path: str,
    device: str = "cpu",
):
    image = Image.open(input_image_path)
    image_tensor = TF.to_tensor(image).unsqueeze(0)
    denoised_image = denoise_image(exported_model, image_tensor, device)
    save_denoised_image(denoised_image, output_image_path)


def process_directory(
    exported_model: torch.nn.Module, input_dir: str, output_dir: str, device: str = "cpu"
):
    ds = dataset.ImageFolderDataset(input_dir)
    for i, image in enumerate(ds):
        image_path = ds.image_files[i]
        x = TF.to_tensor(image).unsqueeze(0)
        denoised_image = denoise_image(exported_model, x, device)
        output_image_path = Path(output_dir).joinpath(image_path.name)
        save_denoised_image(denoised_image, output_image_path)


def process_hf_dataset(
    exported_model: torch.nn.Module, dataset_path: str, output_dir: str, device: str = "cpu"
):
    ds = dataset.HFDataset(dataset_path)
    for i, image in enumerate(ds):
        image_name = ds.ds[i]["id"]
        x = TF.to_tensor(image).unsqueeze(0)
        denoised_image = denoise_image(exported_model, x, device)
        output_image_path = Path(output_dir).joinpath(f"{image_name}.jpg")
        save_denoised_image(denoised_image, output_image_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Perform inference using an exported denoising model."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the exported model file (TorchScript).",
    )
    parser.add_argument("--input_image_path", type=str, help="Path to the input noisy image.")
    parser.add_argument("--input_dir", type=str, help="Path to the directory with noisy images.")
    parser.add_argument(
        "--hf_dataset_path", type=str, help="Path to the local Hugging Face dataset."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True, help="Directory to save the denoised images."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the inference on ('cpu' or 'cuda').",
    )

    args = parser.parse_args()

    # Load the exported model
    model = load_exported_model(args.model_path, args.device)

    # Process inputs based on the provided arguments
    if args.input_image_path:
        output_image_path = Path(args.output_dir) / Path(args.input_image_path).name
        process_single_image(model, args.input_image_path, output_image_path, args.device)
    elif args.input_dir:
        process_directory(model, args.input_dir, args.output_dir, args.device)
    elif args.hf_dataset_path:
        process_hf_dataset(model, args.hf_dataset_path, args.output_dir, args.device)
    else:
        raise ValueError(
            "You must provide either --input_image_path, --input_dir, or --hf_dataset_path."
        )
