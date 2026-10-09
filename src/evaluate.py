"""
Evaluation for instance segmentation — computes mAP, mAP@50, and
average mask IoU. Can also dump predictions to JSON for the pick evaluator.

Usage:
    python src/evaluate.py --config configs/resnet101_fpn.yaml \
        --checkpoint results/r101_best.pth \
        --save-preds results/r101_preds.json
"""

import argparse
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

from dataset import build_dataset
from augmentations import ValTransform
from model import build_model, maybe_swap_loss
from utils import load_config, load_checkpoint, collate_fn, get_device


@torch.no_grad()
def evaluate_coco(model, dataloader, device, score_thresh=0.5):
    model.eval()
    coco_gt = dataloader.dataset.coco

    results_det = []   # for box mAP
    results_seg = []   # for segm mAP
    all_mask_ious = []

    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for output, target in zip(outputs, targets):
            img_id = target['image_id'].item()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()
            boxes = output['boxes'].cpu().numpy()
            masks = output['masks'].cpu()  # (N, 1, H, W) float

            # threshold
            keep = scores >= score_thresh
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            masks = masks[keep]

            if len(scores) == 0:
                continue
            # print(f"img {img_id}: {len(scores)} dets above thresh")

            # binary masks at 0.5
            bin_masks = (masks.squeeze(1) > 0.5).numpy().astype(np.uint8)

            # compute pairwise mask IoU against GT
            gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id, iscrowd=False)
            gt_anns = coco_gt.loadAnns(gt_ann_ids)
            if gt_anns:
                ious = _compute_mask_ious(bin_masks, gt_anns, coco_gt, img_id)
                all_mask_ious.extend(ious)

            # format for COCO eval
            for i in range(len(scores)):
                # RLE encode the mask
                rle = mask_utils.encode(
                    np.asfortranarray(bin_masks[i])
                )
                rle['counts'] = rle['counts'].decode('utf-8')

                # reverse the cat_id mapping
                # XXX: rebuilding this every iteration is dumb, should cache it
                cat_mapping = dataloader.dataset.cat_id_to_label
                inv_mapping = {v: k for k, v in cat_mapping.items()}
                cat_id = inv_mapping.get(int(labels[i]), int(labels[i]))

                x1, y1, x2, y2 = boxes[i]
                result = {
                    'image_id': img_id,
                    'category_id': cat_id,
                    'score': float(scores[i]),
                    'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    'segmentation': rle,
                }
                results_det.append(result)
                results_seg.append(result)

    if len(results_seg) == 0:
        print("No predictions above threshold — returning zeros")
        return {'mAP': 0.0, 'mAP_50': 0.0, 'mAP_75': 0.0, 'avg_mask_iou': 0.0}

    # run COCO eval for segmentation
    coco_dt = coco_gt.loadRes(results_seg)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='segm')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    avg_iou = float(np.mean(all_mask_ious)) if all_mask_ious else 0.0

    metrics = {
        'mAP': coco_eval.stats[0],        # mAP @ IoU=0.50:0.95
        'mAP_50': coco_eval.stats[1],      # mAP @ IoU=0.50
        'mAP_75': coco_eval.stats[2],      # mAP @ IoU=0.75
        'avg_mask_iou': avg_iou,
    }
    return metrics


def _compute_mask_ious(pred_masks, gt_anns, coco, img_id):
    # greedy matching, not hungarian — good enough for tracking trends
    # (scipy.optimize.linear_sum_assignment would be more correct)
    img_info = coco.loadImgs(img_id)[0]
    h, w = img_info['height'], img_info['width']

    # decode gt masks
    gt_masks = []
    for ann in gt_anns:
        if isinstance(ann['segmentation'], list):
            rle = mask_utils.frPyObjects(ann['segmentation'], h, w)
            rle = mask_utils.merge(rle)
        else:
            rle = ann['segmentation']
        gt_masks.append(mask_utils.decode(rle))

    ious = []
    used_gt = set()

    for pred_m in pred_masks:
        best_iou = 0.0
        best_idx = -1
        for j, gt_m in enumerate(gt_masks):
            if j in used_gt:
                continue
            # make sure shapes match
            if pred_m.shape != gt_m.shape:
                continue
            intersection = (pred_m & gt_m).sum()
            union = (pred_m | gt_m).sum()
            if union == 0:
                continue
            iou = intersection / union
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_idx >= 0:
            used_gt.add(best_idx)
        ious.append(best_iou)

    return ious


def save_predictions(model, dataloader, device, output_path, score_thresh=0.5):
    model.eval()
    all_preds = []

    with torch.no_grad():
        for images, targets in dataloader:
            images = [img.to(device) for img in images]
            outputs = model(images)

            for output, target in zip(outputs, targets):
                img_id = target['image_id'].item()
                scores = output['scores'].cpu().numpy()
                masks = output['masks'].cpu()
                boxes = output['boxes'].cpu().numpy()

                keep = scores >= score_thresh
                bin_masks = (masks[keep].squeeze(1) > 0.5).numpy().astype(np.uint8)

                for i in range(keep.sum()):
                    rle = mask_utils.encode(np.asfortranarray(bin_masks[i]))
                    rle['counts'] = rle['counts'].decode('utf-8')
                    x1, y1, x2, y2 = boxes[keep][i]

                    all_preds.append({
                        'image_id': img_id,
                        'score': float(scores[keep][i]),
                        'bbox': [float(x1), float(y1), float(x2-x1), float(y2-y1)],
                        'segmentation': rle,
                    })

    with open(output_path, 'w') as f:
        json.dump(all_preds, f)
    print(f"Saved {len(all_preds)} predictions to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--save-preds', type=str, default=None,
                        help='Path to save predictions JSON (for pick_eval.py)')
    parser.add_argument('--score-thresh', type=float, default=0.5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = get_device()

    val_transforms = ValTransform()
    test_dataset = build_dataset(cfg, split='test', transforms=val_transforms)
    num_classes = test_dataset.get_num_classes()

    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=cfg['data'].get('num_workers', 4),
        collate_fn=collate_fn,
    )

    model = build_model(cfg, num_classes)
    model = maybe_swap_loss(model, cfg)
    _, _ = load_checkpoint(args.checkpoint, model)
    model.to(device)

    print("Running evaluation on test set...")
    metrics = evaluate_coco(model, test_loader, device, score_thresh=args.score_thresh)

    print("\n=== Test Results ===")
    print(f"  mAP (0.50:0.95): {metrics['mAP']:.4f}")
    print(f"  mAP @ 0.50:      {metrics['mAP_50']:.4f}")
    print(f"  mAP @ 0.75:      {metrics['mAP_75']:.4f}")
    print(f"  Avg mask IoU:     {metrics['avg_mask_iou']:.4f}")

    if args.save_preds:
        save_predictions(model, test_loader, device, args.save_preds, args.score_thresh)


if __name__ == '__main__':
    main()
