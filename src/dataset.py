import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask_utils
# NOTE: pycocotools prints a bunch of stuff to stdout when loading,
# haven't found a clean way to suppress it


class BinDataset(Dataset):
    """COCO-format dataset for cluttered bin images."""

    def __init__(self, root_dir, transforms=None, min_area=256):
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'images')
        self.transforms = transforms
        self.min_area = min_area  # skip tiny annotations

        ann_file = os.path.join(root_dir, 'annotations.json')
        self.coco = COCO(ann_file)

        # only keep images that actually have annotations
        self.img_ids = []
        for img_id in sorted(self.coco.getImgIds()):
            ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
            if len(ann_ids) > 0:
                self.img_ids.append(img_id)

        print(f"Loaded {len(self.img_ids)} images from {ann_file}")

        # build category mapping (coco ids aren't always contiguous)
        cat_ids = sorted(self.coco.getCatIds())
        self.cat_id_to_label = {cat_id: i + 1 for i, cat_id in enumerate(cat_ids)}
        self.num_classes = len(cat_ids) + 1  # +1 for background

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs(img_id)[0]
        img_path = os.path.join(self.img_dir, img_info['file_name'])

        img = Image.open(img_path).convert('RGB')
        w, h = img.size

        ann_ids = self.coco.getAnnIds(imgIds=img_id, iscrowd=False)
        anns = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        masks = []
        areas = []

        for ann in anns:
            # skip small stuff — can't segment it reliably anyway
            if ann.get('area', 0) < self.min_area:
                continue

            # bbox: coco gives [x, y, w, h], we need [x1, y1, x2, y2]
            x, y, bw, bh = ann['bbox']
            if bw < 1 or bh < 1:
                continue
            boxes.append([x, y, x + bw, y + bh])
            labels.append(self.cat_id_to_label[ann['category_id']])
            areas.append(ann['area'])

            # decode mask
            if isinstance(ann['segmentation'], list):
                # polygon format
                rle = coco_mask_utils.frPyObjects(ann['segmentation'], h, w)
                rle = coco_mask_utils.merge(rle)
            else:
                rle = ann['segmentation']
            m = coco_mask_utils.decode(rle)
            masks.append(m)

        if len(boxes) == 0:
            # degenerate case — return dummy target
            # this shouldn't happen often since we filtered in __init__
            target = {
                'boxes': torch.zeros((0, 4), dtype=torch.float32),
                'labels': torch.zeros(0, dtype=torch.int64),
                'masks': torch.zeros((0, h, w), dtype=torch.uint8),
                'image_id': torch.tensor([img_id]),
                'area': torch.zeros(0, dtype=torch.float32),
                'iscrowd': torch.zeros(0, dtype=torch.int64),
            }
            img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32)
            img_tensor = img_tensor.permute(2, 0, 1) / 255.0
            return img_tensor, target

        target = {
            'boxes': torch.as_tensor(boxes, dtype=torch.float32),
            'labels': torch.as_tensor(labels, dtype=torch.int64),
            'masks': torch.as_tensor(np.array(masks), dtype=torch.uint8),
            'image_id': torch.tensor([img_id]),
            'area': torch.as_tensor(areas, dtype=torch.float32),
            'iscrowd': torch.zeros(len(boxes), dtype=torch.int64),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            # basic: just convert to tensor
            img = torch.as_tensor(np.array(img), dtype=torch.float32)
            img = img.permute(2, 0, 1) / 255.0

        return img, target

    def get_num_classes(self):
        return self.num_classes


def build_dataset(cfg, split='train', transforms=None):
    if split == 'train':
        root = cfg['data']['train_dir']
    elif split == 'val':
        root = cfg['data']['val_dir']
    elif split == 'test':
        root = cfg['data']['test_dir']
    else:
        raise ValueError(f"Unknown split: {split}")

    return BinDataset(root, transforms=transforms)
