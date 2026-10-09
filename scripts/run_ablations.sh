#!/bin/bash
# Run the four main ablation experiments.
# Takes ~8hrs total on a single 3090 (mostly the r101 runs).
#
# If you only care about one, just run the relevant python command directly.

set -e

echo "========================================="
echo " Ablation 1/4: ResNet-50 FPN baseline"
echo "========================================="
python src/train.py \
    --config configs/resnet50_fpn.yaml \
    --experiment-name r50_baseline \
    --seed 42

echo "========================================="
echo " Ablation 2/4: ResNet-101 FPN baseline"
echo "========================================="
python src/train.py \
    --config configs/resnet101_fpn.yaml \
    --experiment-name r101_baseline \
    --seed 42

echo "========================================="
echo " Ablation 3/4: R101 without occlusion aug"
echo "========================================="
python src/train.py \
    --config configs/ablations/no_occlusion_aug.yaml \
    --experiment-name r101_no_occlusion \
    --seed 42

echo "========================================="
echo " Ablation 4/4: R50 with focal loss"
echo "========================================="
python src/train.py \
    --config configs/ablations/focal_loss.yaml \
    --experiment-name r50_focal \
    --seed 42

echo ""
echo "All ablations complete. Run 'mlflow ui' to compare results."
