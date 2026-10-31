#!/usr/bin/env python3
"""Reproducible stratified NYC dataset split; originals are preserved."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def write_csv(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).parent / 'nyc_dataset')
    parser.add_argument('--output', type=Path, default=Path(__file__).parent)
    parser.add_argument('--validation-ratio', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if not 0 < args.validation_ratio < 1:
        parser.error('validation ratio must be between 0 and 1')
    image_fields, images = read_csv(args.source / 'images.csv')
    annotation_fields, annotations = read_csv(args.source / 'annotations.csv')
    categories = {'cone_only': 'cone', 'sign_only': 'construction_sign',
                  'both': 'combination', 'negative': 'negative'}
    assert set(row['group'] for row in images) == set(categories)
    assert len({row['image'] for row in images}) == len(images)
    assert all((args.source / row['image']).is_file() for row in images)
    assert {row['image'] for row in annotations} <= {row['image'] for row in images}
    assert all(row['sequence'] for row in images)
    assert len({row['sequence'] for row in images}) == len(images), 'Duplicate capture sequences require grouped splitting'
    for name in ('validation', 'test'):
        if (args.output / name).exists():
            raise FileExistsError(f'Refusing to overwrite {args.output / name}')
    splits = {'validation': [], 'test': []}
    rng = random.Random(args.seed)
    for group in categories:
        rows = sorted((r for r in images if r['group'] == group), key=lambda r: r['id'])
        rng.shuffle(rows)
        count = round(len(rows) * args.validation_ratio)
        if abs(count - len(rows) * args.validation_ratio) > 1e-9:
            raise ValueError(f'{group}: ratio cannot preserve exact class proportions with integer image counts')
        splits['validation'].extend(rows[:count])
        splits['test'].extend(rows[count:])
    assert not ({r['id'] for r in splits['validation']} & {r['id'] for r in splits['test']})
    assert sum(map(len, splits.values())) == len(images)
    summary = {'seed': args.seed, 'validation_ratio': args.validation_ratio, 'splits': {}}
    for name, rows in splits.items():
        root = args.output / name
        for category in categories.values():
            (root / category).mkdir(parents=True, exist_ok=False)
        paths = {}
        output_rows = []
        for row in rows:
            relative = f"{categories[row['group']]}/{Path(row['image']).name}"
            paths[row['image']] = relative
            source, target = args.source / row['image'], root / relative
            shutil.copy2(source, target)
            assert hashlib.sha256(source.read_bytes()).digest() == hashlib.sha256(target.read_bytes()).digest()
            output_rows.append(dict(row, image=relative))
        output_annotations = [dict(row, image=paths[row['image']]) for row in annotations if row['image'] in paths]
        write_csv(root / 'images.csv', image_fields, output_rows)
        write_csv(root / 'annotations.csv', annotation_fields, output_annotations)
        shutil.copy2(args.source / 'README.md', root / 'SOURCE_README.md')
        counts = dict(Counter(categories[row['group']] for row in rows))
        summary['splits'][name] = {'total': len(rows), 'categories': counts, 'annotations': len(output_annotations)}
        (root / 'README.md').write_text(
            f'# NYC {name} split\n\n'
            f'Seed: {args.seed}; validation fraction: {args.validation_ratio}.\n'
            'Stratified by the original images.csv group; source images are copied unchanged.\n'
            'cone = cone_only; construction_sign = sign_only; combination = both.\n'
            'images.csv and annotations.csv paths are relative to this folder.\n'
            'See SOURCE_README.md for provenance, licensing and annotation limitations.\n\n'
            + json.dumps(counts, indent=2) + '\n', encoding='utf-8')
    (args.output / 'split_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
