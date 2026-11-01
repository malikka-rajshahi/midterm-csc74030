"""Load the Roadwork Cones dataset from Hugging Face, and the NYC street-level images.

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
NYC_DIR = Path(__file__).resolve().parent.parent / "nyc_data"
# NYC window sizes: there are no boxes to measure, so a fixed width range per class, up to a quarter of the 1024 px
# image width. Cone windows are twice as tall as wide, sign windows square. Windows slide by a quarter of their width.
# Sign windows start at 48 px: smaller ones let tiny all-orange patches (lights, buses, bike frames) beat whole objects
# in NMS. Cone windows start at 16 px, since most NYC cones are narrower than 48 px.
NYC_WIDTHS = {"cone": (16, 256), "sign": (48, 256)}
NYC_ASPECTS = {"cone": 2.0, "sign": 1.0}
NYC_STRIDE_FRACTION = 0.25
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



def nyc_images():
    """All 300 NYC street-level image paths (relative to NYC_DIR). Their annotations are not used."""
    return sorted(str(p.relative_to(NYC_DIR)) for p in (NYC_DIR / "images").glob("*.jpg"))
