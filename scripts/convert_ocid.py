"""
Convert OCID (Object Clutter Indoor Dataset) to COCO annotation format.

OCID has its own annotation structure — this script reads it and spits out
the standard COCO JSON that our dataloader expects.

Usage:
    python scripts/convert_ocid.py \
        --ocid-root data/raw/OCID \
        --output-dir data/ \
        --val-split 0.15 \
        --test-split 0.15

After running, you should have:
    data/train/images/ + annotations.json
    data/val/images/   + annotations.json
    data/test/images/  + annotations.json
"""

import argparse
import os
import json
import glob
import random
import shutil
import numpy as np
from PIL import Image


def load_ocid_scene(scene_dir):
    """
    Load a single OCID scene. Returns list of (image_path, label_path) pairs.
    OCID stores per-pixel label images where each pixel value = object id.
    """
    rgb_dir = os.path.join(scene_dir, 'rgb')
    label_dir = os.path.join(scene_dir, 'label')

    if not os.path.isdir(rgb_dir) or not os.path.isdir(label_dir):
        return []

    pairs = []
    for rgb_file in sorted(glob.glob(os.path.join(rgb_dir, '*.png'))):
        fname = os.path.basename(rgb_file)
        label_file = os.path.join(label_dir, fname)
        if os.path.exists(label_file):
            pairs.append((rgb_file, label_file))

    return pairs


def label_image_to_coco_anns(label_path, image_id, start_ann_id, min_area=256):
    """
    Convert a label image to COCO-format annotations.
    Each unique nonzero pixel value = one object instance.
    """
    label_img = np.array(Image.open(label_path))
    annotations = []

    obj_ids = np.unique(label_img)
    obj_ids = obj_ids[obj_ids != 0]  # 0 = background

    ann_id = start_ann_id
    for obj_id in obj_ids:
        mask = (label_img == obj_id).astype(np.uint8)
        area = int(mask.sum())

        if area < min_area:
            continue

        # bounding box
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            continue
        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        bbox = [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]

        # polygon segmentation (simplified — contour based)
        from cv2 import findContours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE
        contours, _ = findContours(mask, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)

        segmentation = []
        for contour in contours:
            if len(contour) < 6:  # need at least 3 points
                continue
            seg = contour.flatten().tolist()
            segmentation.append(seg)

        if not segmentation:
            continue

        ann = {
            'id': ann_id,
            'image_id': image_id,
            'category_id': 1,  # single "object" category for now
            'segmentation': segmentation,
            'bbox': bbox,
            'area': area,
            'iscrowd': 0,
        }
        annotations.append(ann)
        ann_id += 1

    return annotations, ann_id


def convert(ocid_root, output_dir, val_split=0.15, test_split=0.15, seed=42):
    random.seed(seed)

    # find all scenes
    all_pairs = []
    for scene_type in ['ARID10', 'ARID20', 'YCB10', 'YCB100']:
        scene_type_dir = os.path.join(ocid_root, scene_type)
        if not os.path.isdir(scene_type_dir):
            continue
        for scene_name in sorted(os.listdir(scene_type_dir)):
            scene_dir = os.path.join(scene_type_dir, scene_name)
            pairs = load_ocid_scene(scene_dir)
            all_pairs.extend(pairs)

    if not all_pairs:
        print(f"No scenes found in {ocid_root}. Check the path.")
        print("Expected structure: OCID/ARID10/scene_01/rgb/*.png")
        return

    print(f"Found {len(all_pairs)} image-label pairs")

    # shuffle and split
    random.shuffle(all_pairs)
    n = len(all_pairs)
    n_test = int(n * test_split)
    n_val = int(n * val_split)

    test_pairs = all_pairs[:n_test]
    val_pairs = all_pairs[n_test:n_test + n_val]
    train_pairs = all_pairs[n_test + n_val:]

    print(f"Split: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")

    categories = [{'id': 1, 'name': 'object', 'supercategory': 'item'}]

    for split_name, pairs in [('train', train_pairs), ('val', val_pairs), ('test', test_pairs)]:
        split_dir = os.path.join(output_dir, split_name)
        img_dir = os.path.join(split_dir, 'images')
        os.makedirs(img_dir, exist_ok=True)

        images = []
        annotations = []
        ann_id = 1

        for i, (rgb_path, label_path) in enumerate(pairs):
            img = Image.open(rgb_path)
            w, h = img.size

            image_id = i + 1
            new_fname = f'scene_{image_id:06d}.png'

            # copy image
            shutil.copy2(rgb_path, os.path.join(img_dir, new_fname))

            images.append({
                'id': image_id,
                'file_name': new_fname,
                'width': w,
                'height': h,
            })

            anns, ann_id = label_image_to_coco_anns(
                label_path, image_id, ann_id, min_area=256
            )
            annotations.extend(anns)

            if (i + 1) % 100 == 0:
                print(f"  {split_name}: processed {i+1}/{len(pairs)}")

        coco_json = {
            'images': images,
            'annotations': annotations,
            'categories': categories,
        }

        ann_path = os.path.join(split_dir, 'annotations.json')
        with open(ann_path, 'w') as f:
            json.dump(coco_json, f)

        print(f"  {split_name}: {len(images)} images, {len(annotations)} annotations -> {ann_path}")

    print("\nDone! Dataset ready.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ocid-root', required=True, help='Path to OCID dataset root')
    parser.add_argument('--output-dir', required=True, help='Where to write train/val/test')
    parser.add_argument('--val-split', type=float, default=0.15)
    parser.add_argument('--test-split', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    convert(args.ocid_root, args.output_dir,
            val_split=args.val_split, test_split=args.test_split, seed=args.seed)
