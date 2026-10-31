"""Load the Roadwork Cones dataset from Hugging Face.

https://huggingface.co/datasets/StarLineResearch/Roadwork_Cones_Dataset
"""
import json
import random
import zipfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO_ID = "StarLineResearch/Roadwork_Cones_Dataset"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATASET_DIR = DATA_DIR / "roadwork_cones_dataset"
SPLIT_FILE = DATA_DIR / "split.json"
SEED = 42
# Dataset label -> our class (the Vistas template classes). vertical_pannel has no templates, so it is dropped.
LABEL_MAP = {"cone": "cone", "roadworks": "sign"}


def load_dataset():
    """Download and extract the dataset (only if it isn't on disk yet), then return its annotations."""
    if not (DATASET_DIR / "annotations.parquet").exists():
        zip_path = hf_hub_download(REPO_ID, "roadwork_cones_dataset.zip", repo_type="dataset", local_dir=DATA_DIR)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(DATA_DIR)
        Path(zip_path).unlink()
    return pd.read_parquet(DATASET_DIR / "annotations.parquet")


def map_labels(df):
    """Keep only labels in LABEL_MAP and add a `cls` column with our class name (cone / sign)."""
    df = df[df["label"].isin(LABEL_MAP)].copy()
    df["cls"] = df["label"].map(LABEL_MAP)
    return df.reset_index(drop=True)


def split_data(df, val_ratio=0.2, seed=SEED):
    """Split images into train / validation / test and save the image IDs to SPLIT_FILE.

    The dataset's own test/ images stay the test set, untouched. Validation is a random
    val_ratio of the train/ images. Image ID = image_path, so all boxes of an image stay together.
    """
    train_images = sorted(df.loc[df["image_path"].str.startswith("train/"), "image_path"].unique())
    test_images = sorted(df.loc[df["image_path"].str.startswith("test/"), "image_path"].unique())
    random.Random(seed).shuffle(train_images)
    n_val = round(len(train_images) * val_ratio)
    split = {"seed": seed,
             "train": sorted(train_images[n_val:]),
             "validation": sorted(train_images[:n_val]),
             "test": test_images}
    SPLIT_FILE.write_text(json.dumps(split, indent=1))
    return split
