import argparse
import os
from io import BytesIO
from pathlib import Path

import pandas as pd

# Import the functions you defined earlier
from datasets import Dataset, Features, Image, Value, load_from_disk
from dotenv import load_dotenv
from img2dataset import download
from PIL import Image as PILImage
from tqdm import tqdm


def image_to_bytes(img):
    """Convert a PIL image to bytes."""
    with BytesIO() as buffer:
        img.save(buffer, format="JPEG")
        return buffer.getvalue()


def download_images(csv_path: str, output_dir: str) -> None:
    download(
        processes_count=1,
        thread_count=4,
        url_list=csv_path,
        image_size=2048,
        resize_mode="no",
        resize_only_if_bigger=True,
        output_folder=output_dir,
        output_format="files",
        input_format="csv",
        url_col="TIFF",
        save_additional_columns=["Keywords", "File"],
        enable_wandb=False,
        number_sample_per_shard=1000,
        extract_exif=False,
        distributor="multiprocessing",
    )


def create_dataset(csv_path, output_dir):
    """Create a Hugging Face dataset from the RAISE_1k CSV file."""
    df = pd.read_csv(csv_path)

    file_idx = "00000"
    df_index = pd.read_parquet(Path(output_dir).joinpath(f"{file_idx}.parquet"))
    file_root_path = Path(output_dir).joinpath(file_idx)

    observations = []
    for _, row in tqdm(df_index.iterrows(), total=df_index.shape[0]):
        file_path = file_root_path.joinpath(f"{row['key']}.jpg")
        if file_path.exists():
            img = PILImage.open(file_path)
            img.thumbnail((1024, 1024))
            img_bytes = image_to_bytes(img)

            meta_data = {
                "image": img_bytes,
                "file": row["File"],
                "keywords": row["Keywords"],
                "width": img.size[0],
                "height": img.size[1],
            }
            observations.append(meta_data)

    features = Features(
        {
            "image": Image(decode=True),
            "file": Value("string"),
            "keywords": Value("string"),
            "width": Value("int32"),
            "height": Value("int32"),
        }
    )

    hf_dataset = Dataset.from_list(observations, features=features)
    return hf_dataset


def save_dataset(dataset, save_path):
    """Save the dataset to disk."""
    dataset.save_to_disk(save_path)


def create_splits(dataset, save_path, seed=123, test_size=0.5):
    """Create and save train/val/test splits of the dataset."""
    ds_splitted = dataset.train_test_split(test_size=test_size, seed=seed)
    ds_val_test = ds_splitted["test"].train_test_split(test_size=0.5, seed=seed)
    ds_val_test["val"] = ds_val_test["train"]
    _ = ds_val_test.pop("train")
    ds_splitted["val"] = ds_val_test["val"]
    ds_splitted["test"] = ds_val_test["test"]

    ds_splitted.save_to_disk(save_path)


def main(csv_path, output_path, download):
    # Define paths
    output_dir = output_path.joinpath("original_images")
    ds_save_path = output_path.joinpath("dataset")

    # Download Images
    if download:
        download_images(str(csv_path), output_dir=str(output_dir))

    # Create dataset
    dataset = create_dataset(csv_path, output_dir)

    # Save dataset
    save_dataset(dataset, ds_save_path)

    # Create and save dataset splits
    create_splits(dataset, ds_save_path)


if __name__ == "__main__":
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Process RAISE-1k dataset and create Hugging Face datasets."
    )
    parser.add_argument("--csv_path", type=Path, help="Path to the RAISE_1k.csv file.")
    parser.add_argument(
        "--output_path", type=Path, help="Directory where the output dataset will be saved."
    )
    parser.add_argument(
        "--download", action="store_true", default=False, help="whether to download the images"
    )

    args = parser.parse_args()

    main(csv_path=args.csv_path, output_path=args.output_path, download=args.download)
