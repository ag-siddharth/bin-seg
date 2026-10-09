"""
Pick-success estimator: maps mask segmentation quality to an estimated
robotic pick success rate.

The idea is that good masks -> good grasp plans -> successful picks.
This is NOT a physics simulation — it's a heuristic proxy that uses:
  1. Mask completeness (how much of the GT object the prediction covers)
  2. Boundary quality (how tight the predicted boundary is)
  3. Isolation score (whether the mask bleeds into neighbors)

These three factors get weighted into a per-object "pickability" score.
An object is considered pickable if the score exceeds a threshold.

The thresholds and weights were eyeballed based on the numbers in
Zeng et al. and the RightHand Robotics blog posts about pick failures.
Could definitely be improved with real pick data.

Usage:
    python src/pick_eval.py \
        --predictions results/r101_preds.json \
        --ground-truth data/test/annotations.json
"""

import argparse
import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils


# weights for the three factors — tuned by hand
COMPLETENESS_WEIGHT = 0.45
BOUNDARY_WEIGHT = 0.30
ISOLATION_WEIGHT = 0.25
# tried 0.5/0.2/0.3 too, didn't change results much

# if the combined score is above this, we count it as a successful pick
PICK_THRESHOLD = 0.55  # 0.5 was too generous, 0.6 too strict


def compute_completeness(pred_mask, gt_mask):
    # basically recall but for mask pixels
    gt_area = gt_mask.sum()
    if gt_area == 0:
        return 0.0
    overlap = (pred_mask & gt_mask).sum()
    return float(overlap) / float(gt_area)


def compute_boundary_quality(pred_mask, gt_mask):
    # simplified boundary IoU — dilate both, compare the boundary ring.
    # 2px tolerance. there's a proper boundary AP metric but this is simpler
    from scipy import ndimage  # lazy import, only pick_eval needs scipy

    # dilate both masks slightly, then look at the boundary ring
    struct = ndimage.generate_binary_structure(2, 1)

    pred_dilated = ndimage.binary_dilation(pred_mask, struct, iterations=2)
    pred_boundary = pred_dilated.astype(np.uint8) - pred_mask.astype(np.uint8)

    gt_dilated = ndimage.binary_dilation(gt_mask, struct, iterations=2)
    gt_boundary = gt_dilated.astype(np.uint8) - gt_mask.astype(np.uint8)

    if gt_boundary.sum() == 0:
        return 1.0  # no boundary to match

    intersection = (pred_boundary & gt_boundary).sum()
    union = (pred_boundary | gt_boundary).sum()

    if union == 0:
        return 0.0
    return float(intersection) / float(union)


def compute_isolation(pred_mask, all_gt_masks, matched_idx):
    pred_area = pred_mask.sum()
    if pred_area == 0:
        return 0.0

    bleed = 0
    for i, gt_m in enumerate(all_gt_masks):
        if i == matched_idx:
            continue
        bleed += (pred_mask & gt_m).sum()

    bleed_ratio = float(bleed) / float(pred_area)
    # invert: 0 bleed -> 1.0 isolation, lots of bleed -> 0.0
    return max(0.0, 1.0 - bleed_ratio * 3)  # scale factor of 3 — harsh on bleed


def match_predictions_to_gt(predictions, coco_gt, iou_thresh=0.5):
    # group preds by image
    preds_by_img = {}
    for pred in predictions:
        img_id = pred['image_id']
        if img_id not in preds_by_img:
            preds_by_img[img_id] = []
        preds_by_img[img_id].append(pred)

    matches = []

    for img_id, preds in preds_by_img.items():
        img_info = coco_gt.loadImgs(img_id)[0]
        h, w = img_info['height'], img_info['width']

        # sort by score descending
        preds = sorted(preds, key=lambda x: x['score'], reverse=True)

        # decode GT masks
        gt_ann_ids = coco_gt.getAnnIds(imgIds=img_id, iscrowd=False)
        gt_anns = coco_gt.loadAnns(gt_ann_ids)

        gt_masks = []
        for ann in gt_anns:
            if isinstance(ann['segmentation'], list):
                rle = mask_utils.frPyObjects(ann['segmentation'], h, w)
                rle = mask_utils.merge(rle)
            else:
                rle = ann['segmentation']
            gt_masks.append(mask_utils.decode(rle))

        used_gt = set()

        for pred in preds:
            pred_rle = pred['segmentation']
            pred_mask = mask_utils.decode(pred_rle)

            best_iou = 0
            best_j = -1
            for j, gt_m in enumerate(gt_masks):
                if j in used_gt:
                    continue
                if pred_mask.shape != gt_m.shape:
                    continue
                inter = (pred_mask & gt_m).sum()
                union = (pred_mask | gt_m).sum()
                iou = inter / union if union > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_j = j

            if best_iou >= iou_thresh and best_j >= 0:
                used_gt.add(best_j)
                matches.append((pred_mask, gt_masks[best_j], gt_masks, best_j))

    return matches


def estimate_pick_success(predictions, coco_gt):
    matches = match_predictions_to_gt(predictions, coco_gt)

    if len(matches) == 0:
        print("No matches found — either no predictions or no GT overlap")
        return 0.0, []

    scores = []
    for pred_mask, gt_mask, all_gt, gt_idx in matches:
        completeness = compute_completeness(pred_mask, gt_mask)
        boundary = compute_boundary_quality(pred_mask, gt_mask)
        isolation = compute_isolation(pred_mask, all_gt, gt_idx)

        combined = (
            COMPLETENESS_WEIGHT * completeness +
            BOUNDARY_WEIGHT * boundary +
            ISOLATION_WEIGHT * isolation
        )
        scores.append({
            'completeness': completeness,
            'boundary_quality': boundary,
            'isolation': isolation,
            'combined': combined,
            'pickable': combined >= PICK_THRESHOLD,
        })

    n_pickable = sum(1 for s in scores if s['pickable'])
    success_rate = n_pickable / len(scores)

    return success_rate, scores


def main():
    parser = argparse.ArgumentParser(description='Estimate pick success from segmentation quality')
    parser.add_argument('--predictions', required=True, help='Predictions JSON from evaluate.py')
    parser.add_argument('--ground-truth', required=True, help='COCO annotations JSON (test set)')
    parser.add_argument('--threshold', type=float, default=PICK_THRESHOLD)
    args = parser.parse_args()

    global PICK_THRESHOLD
    PICK_THRESHOLD = args.threshold

    with open(args.predictions, 'r') as f:
        predictions = json.load(f)

    coco_gt = COCO(args.ground_truth)

    print(f"Loaded {len(predictions)} predictions")
    print(f"Pick threshold: {PICK_THRESHOLD}")
    print()

    success_rate, scores = estimate_pick_success(predictions, coco_gt)

    if scores:
        completeness_avg = np.mean([s['completeness'] for s in scores])
        boundary_avg = np.mean([s['boundary_quality'] for s in scores])
        isolation_avg = np.mean([s['isolation'] for s in scores])
        combined_avg = np.mean([s['combined'] for s in scores])

        print("=== Pick Success Estimation ===")
        print(f"  Matched objects:     {len(scores)}")
        print(f"  Avg completeness:    {completeness_avg:.3f}")
        print(f"  Avg boundary qual:   {boundary_avg:.3f}")
        print(f"  Avg isolation:       {isolation_avg:.3f}")
        print(f"  Avg combined score:  {combined_avg:.3f}")
        print(f"  Estimated pick rate: {success_rate*100:.1f}%")

        # breakdown by score bucket
        buckets = [(0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        print("\nScore distribution:")
        for lo, hi in buckets:
            count = sum(1 for s in scores if lo <= s['combined'] < hi)
            print(f"  [{lo:.1f}, {hi:.1f}): {count} objects ({count/len(scores)*100:.1f}%)")
    else:
        print("No matched objects to evaluate")


if __name__ == '__main__':
    main()
