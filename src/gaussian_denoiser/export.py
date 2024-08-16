import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms import functional as TF

from gaussian_denoiser import dncnn, utils


def export_model(
    model_export_dir: str,
    save_path: str,
    device: str,
    verify_export: bool = False,
    test_image_path: str | None = None,
):

    model_export_dir = Path(model_export_dir)
    save_path = Path(save_path)

    assert model_export_dir.exists(), f"model_export_dir: {model_export_dir} does not exist"
    assert save_path.parent.exists(), f"parent dir of save_path: {save_path} does not exist"

    # CFG_PATH = Path(model_export_dir).joinpath(".hydra/config.yaml")

    # Load model checkpoint
    ckpt_path_list = utils.find_all_ckpt_files(model_export_dir)
    ckpt_path = utils.get_ckpt(ckpt_path_list)
    model = dncnn.DnCNNModule.load_from_checkpoint(ckpt_path)
    model = model.to(args.device)

    # Export model
    compiled_model = model.to_torchscript()
    compiled_model.save(save_path)
    print(f"Model exported to {save_path}")

    if verify_export:
        assert Path(test_image_path).exists(), f"test_image {test_image_path} does not exist"

        test_image = Image.open(test_image_path)
        if any(test_image.size) > 1024:
            test_image.thumbnail((1024, 1024))

        # Load exported model
        loaded_model = torch.jit.load(save_path)
        loaded_model = loaded_model.to(device).eval()
        model = model.to(device).eval()

        with torch.no_grad():
            x = TF.to_tensor(test_image).unsqueeze(0).to(device)
            output_actual = loaded_model(x).cpu()
            output_expected = model(x).cpu()

        # Test consistency
        torch.testing.assert_close(output_actual, output_expected)
        print("Consistency test passed: original and exported model outputs match.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export a model using TorchScript and test its consistency."
    )
    parser.add_argument(
        "--model_export_dir", type=str, required=True, help="Path to the model directory."
    )
    parser.add_argument(
        "--save_path", type=str, required=True, help="Path to save the exported model."
    )
    parser.add_argument(
        "--device",
        type=str,
        required=True,
        default="cpu",
        help="Torchscript exports are device specific. Specify the desired device for this export.",
    )
    parser.add_argument(
        "--verify_export",
        action="store_true",
        default=False,
        help="Whether to verify export with a sample image",
    )
    parser.add_argument(
        "--test_image_path", type=str, default=None, help="Path to the test image."
    )

    args = parser.parse_args()

    export_model(
        args.model_export_dir,
        args.save_path,
        args.device,
        args.verify_export,
        args.test_image_path,
    )
