#!/usr/bin/env python3
"""Unpack the midterm datasets made by build_datasets.py.

    python data_extraction.py

vistas_templates.zip              -> templates/
nyc_dataset_part1.zip, part2.zip  -> nyc_data/   (merged into one folder)
"""
import csv, sys, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGETS = {"templates": ["vistas_templates.zip"],
           "nyc_data": sorted(p.name for p in ROOT.glob("nyc_dataset_part*.zip"))}


def extract(zip_path, dest):
    """Extract zip_path into dest, dropping the zip's top-level folder (vistas_templates/, nyc_dataset/)."""
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            rel = Path(*Path(info.filename).parts[1:])
            target = (dest / rel).resolve()
            if info.is_dir() or not rel.parts:
                continue
            if not target.is_relative_to(dest.resolve()):
                sys.exit(f"{zip_path.name}: unsafe path {info.filename}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(z.read(info))
    print(f"extracted {zip_path.name} -> {dest.name}/")


def check(dest, csv_name, column):
    """Every file listed in the CSV should be on disk."""
    with open(dest / csv_name) as f:
        rows = list(csv.DictReader(f))
    missing = [r[column] for r in rows if not (dest / r[column]).exists()]
    if missing:
        sys.exit(f"{dest.name}/: {len(missing)} of {len(rows)} files in {csv_name} missing, e.g. {missing[0]}")
    print(f"{dest.name}/: all {len(rows)} files in {csv_name} present")


if __name__ == "__main__":
    for folder, zips in TARGETS.items():
        if not zips or not all((ROOT / z).exists() for z in zips):
            sys.exit(f"missing zip(s) for {folder}/: expected {zips or 'nyc_dataset_part*.zip'} next to this script")
        for z in zips:
            extract(ROOT / z, ROOT / folder)
    check(ROOT / "templates", "templates.csv", "crop")
    check(ROOT / "templates", "templates.csv", "mask")
    check(ROOT / "nyc_data", "images.csv", "image")
