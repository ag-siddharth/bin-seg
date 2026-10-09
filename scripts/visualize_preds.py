"""
Visualize model predictions on a single image.

Usage:
    python scripts/visualize_preds.py \
        --image data/test/images/scene_042.png \
        --checkpoint results/r101_best.pth \
        --config configs/resnet101_fpn.yaml \
        --threshold 0.6 \
        --output viz_output.png
"""

import argparse
import sys
import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from model import build_model, maybe_swap_loss
from utils import load_config, load_checkpoint


COLORS = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
    '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9',
    '#F8C471', '#82E0AA', '#F1948A', '#AED6F1', '#D2B4DE',
]


def visualize(image_path, checkpoint_path, config_path, threshold=0.5, output_path=None):
    cfg = load_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # we need to know num_classes — hacky but works: just try common values
    # or load from the checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    # try to figure out num_classes from the classifier weight shape
    for key in ckpt['model_state_dict']:
        if 'box_predictor.cls_score.weight' in key:
            num_classes = ckpt['model_state_dict'][key].shape[0]
            break
    else:
        num_classes = 20  # fallback
        print(f"Couldn't determine num_classes from checkpoint, using {num_classes}")

    model = build_model(cfg, num_classes)
    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device)
    model.eval()

    # load and preprocess image
    img = Image.open(image_path).convert('RGB')
    img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32)
    img_tensor = img_tensor.permute(2, 0, 1) / 255.0
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        output = model([img_tensor])[0]

    scores = output['scores'].cpu().numpy()
    boxes = output['boxes'].cpu().numpy()
    masks = output['masks'].cpu().numpy()

    keep = scores >= threshold
    scores = scores[keep]
    boxes = boxes[keep]
    masks = masks[keep]

    print(f"Found {len(scores)} objects above threshold {threshold}")

    # plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # left: original image with boxes
    axes[0].imshow(img)
    axes[0].set_title('Detections')
    for i, (box, score) in enumerate(zip(boxes, scores)):
        color = COLORS[i % len(COLORS)]
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                  linewidth=2, edgecolor=color, facecolor='none')
        axes[0].add_patch(rect)
        axes[0].text(x1, y1 - 5, f'{score:.2f}', color=color, fontsize=9,
                     fontweight='bold', backgroundcolor='black')

    # right: overlay masks
    axes[1].imshow(img)
    axes[1].set_title('Instance Masks')
    overlay = np.zeros((*np.array(img).shape[:2], 4), dtype=np.float32)

    for i, mask in enumerate(masks):
        color = COLORS[i % len(COLORS)]
        # parse hex color
        r = int(color[1:3], 16) / 255
        g = int(color[3:5], 16) / 255
        b = int(color[5:7], 16) / 255

        binary = (mask.squeeze() > 0.5).astype(np.float32)
        overlay[:, :, 0] += binary * r
        overlay[:, :, 1] += binary * g
        overlay[:, :, 2] += binary * b
        overlay[:, :, 3] += binary * 0.4  # alpha

    overlay = np.clip(overlay, 0, 1)
    axes[1].imshow(overlay, alpha=0.6)

    for ax in axes:
        ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    visualize(args.image, args.checkpoint, args.config,
              threshold=args.threshold, output_path=args.output)
