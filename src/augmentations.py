"""
Custom augmentation pipeline for cluttered bin images.

The main one here is occlusion synthesis — pasting random rectangular
patches from elsewhere in the image to simulate overlap you get in a
real warehouse tote. Forces the model to handle partial visibility.
"""

import random
import numpy as np
import torch
from PIL import Image, ImageFilter
import torchvision.transforms.functional as F
# tried albumentations for this but the mask bookkeeping was annoying
# import albumentations as A


class BinAugmentation:
    """Compose augmentations for bin images + targets (boxes, masks)."""

    def __init__(self, cfg):
        aug_cfg = cfg.get('augmentation', {})
        self.h_flip = aug_cfg.get('horizontal_flip', True)
        self.v_flip = aug_cfg.get('vertical_flip', False)

        self.color_jitter = aug_cfg.get('color_jitter', True)
        self.cj_brightness = aug_cfg.get('color_jitter_brightness', 0.3)
        self.cj_contrast = aug_cfg.get('color_jitter_contrast', 0.3)
        self.cj_saturation = aug_cfg.get('color_jitter_saturation', 0.2)
        self.cj_hue = aug_cfg.get('color_jitter_hue', 0.05)

        self.do_crop = aug_cfg.get('random_crop', True)
        self.crop_scale = aug_cfg.get('crop_scale', [0.7, 1.0])

        self.do_occlusion = aug_cfg.get('occlusion_synthesis', True)
        self.occ_max_patches = aug_cfg.get('occlusion_max_patches', 3)
        self.occ_size_range = aug_cfg.get('occlusion_size_range', [0.05, 0.2])

    def __call__(self, img, target):
        # img is a PIL Image, target is a dict with boxes/masks/labels

        if self.h_flip and random.random() < 0.5:
            img, target = self._hflip(img, target)

        if self.v_flip and random.random() < 0.5:
            img, target = self._vflip(img, target)

        if self.do_crop and random.random() < 0.5:
            img, target = self._random_crop(img, target)

        if self.do_occlusion and random.random() < 0.7:
            img = self._occlusion_synthesis(img)

        if self.color_jitter:
            img = self._apply_color_jitter(img)

        # convert to tensor last
        img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0

        return img_tensor, target

    def _hflip(self, img, target):
        img = F.hflip(img)
        w = img.size[0]

        boxes = target['boxes'].clone()
        # flip x coordinates
        boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        target['boxes'] = boxes

        if 'masks' in target:
            target['masks'] = target['masks'].flip(-1)  # flip along width

        return img, target

    def _vflip(self, img, target):
        img = F.vflip(img)
        h = img.size[1]

        boxes = target['boxes'].clone()
        boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        target['boxes'] = boxes

        if 'masks' in target:
            target['masks'] = target['masks'].flip(-2)

        return img, target

    def _random_crop(self, img, target):
        w, h = img.size
        scale = random.uniform(*self.crop_scale)

        new_w = int(w * scale)
        new_h = int(h * scale)

        left = random.randint(0, w - new_w)
        top = random.randint(0, h - new_h)

        img = F.crop(img, top, left, new_h, new_w)

        # adjust boxes
        boxes = target['boxes'].clone()
        boxes[:, 0] -= left
        boxes[:, 1] -= top
        boxes[:, 2] -= left
        boxes[:, 3] -= top

        # clamp to new image boundaries
        boxes[:, 0].clamp_(min=0, max=new_w)
        boxes[:, 1].clamp_(min=0, max=new_h)
        boxes[:, 2].clamp_(min=0, max=new_w)
        boxes[:, 3].clamp_(min=0, max=new_h)

        # drop boxes that became too small after cropping
        keep = (boxes[:, 2] - boxes[:, 0] > 8) & (boxes[:, 3] - boxes[:, 1] > 8)

        if keep.sum() == 0:
            # crop killed everything, just bail and return original
            # this is rare but happens with aggressive crops
            return F.resize(img, (h, w)), target

        target['boxes'] = boxes[keep]
        target['labels'] = target['labels'][keep]
        target['area'] = target['area'][keep]
        target['iscrowd'] = target['iscrowd'][keep]

        if 'masks' in target:
            masks = target['masks'][:, top:top+new_h, left:left+new_w]
            target['masks'] = masks[keep]

        return img, target

    def _occlusion_synthesis(self, img):
        # paste random patches from elsewhere in the image to simulate
        # overlapping objects. blurs edges so it's not a hard seam.
        # TODO: try elliptical patches instead of rectangles
        img_np = np.array(img)
        h, w = img_np.shape[:2]

        n_patches = random.randint(1, self.occ_max_patches)
        for _ in range(n_patches):
            # random patch size
            pw = random.randint(
                int(w * self.occ_size_range[0]),
                int(w * self.occ_size_range[1])
            )
            ph = random.randint(
                int(h * self.occ_size_range[0]),
                int(h * self.occ_size_range[1])
            )

            if pw < 4 or ph < 4:
                continue

            # source location (where we copy from)
            sx = random.randint(0, max(0, w - pw))
            sy = random.randint(0, max(0, h - ph))

            # target location (where we paste)
            tx = random.randint(0, max(0, w - pw))
            ty = random.randint(0, max(0, h - ph))

            patch = img_np[sy:sy+ph, sx:sx+pw].copy()

            # slight gaussian blur on the patch edges to blend
            patch_pil = Image.fromarray(patch)
            patch_pil = patch_pil.filter(ImageFilter.GaussianBlur(radius=1))
            patch = np.array(patch_pil)

            # alpha blend at ~80% opacity so it's not a hard cutout
            alpha = 0.75 + random.random() * 0.2  # 0.75 to 0.95
            region = img_np[ty:ty+ph, tx:tx+pw]
            # handle edge cases where patch doesn't quite fit
            min_h = min(patch.shape[0], region.shape[0])
            min_w = min(patch.shape[1], region.shape[1])
            img_np[ty:ty+min_h, tx:tx+min_w] = (
                alpha * patch[:min_h, :min_w] +
                (1 - alpha) * region[:min_h, :min_w]
            ).astype(np.uint8)

        return Image.fromarray(img_np)

    # def _random_rotate(self, img, target):
    #     """tried 90-deg rotations but masks got misaligned, revisit later"""
    #     angle = random.choice([0, 90, 180, 270])
    #     if angle == 0:
    #         return img, target
    #     img = F.rotate(img, angle)
    #     # ... boxes and masks need to be rotated too, got messy
    #     return img, target

    def _apply_color_jitter(self, img):
        if random.random() < 0.8:  # don't always apply
            img = F.adjust_brightness(
                img, 1.0 + random.uniform(-self.cj_brightness, self.cj_brightness)
            )
        if random.random() < 0.8:
            img = F.adjust_contrast(
                img, 1.0 + random.uniform(-self.cj_contrast, self.cj_contrast)
            )
        if random.random() < 0.5:
            img = F.adjust_saturation(
                img, 1.0 + random.uniform(-self.cj_saturation, self.cj_saturation)
            )
        if random.random() < 0.3:
            img = F.adjust_hue(
                img, random.uniform(-self.cj_hue, self.cj_hue)
            )
        return img


class ValTransform:
    """Minimal transform for validation — just tensor conversion."""
    def __call__(self, img, target):
        img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32)
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        return img_tensor, target
