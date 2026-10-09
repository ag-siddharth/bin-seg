# Instance Segmentation for Cluttered-Bin Picking

Fine-tuning Mask R-CNN for instance segmentation on cluttered warehouse bin images. The goal is to produce per-item masks good enough to plan robotic grasps — so the evaluation goes beyond standard mAP and also estimates how often a real pick would succeed.

Built as a course/research project at Northeastern University (Fall 2026).

## What this does

1. Trains Mask R-CNN (ResNet-50-FPN or ResNet-101-FPN backbone) on COCO-format bin images
2. Custom augmentation pipeline including occlusion synthesis to handle overlapping items
3. Ablations on backbone, loss function, and augmentation strategies
4. Evaluation that maps mask IoU / mAP to a simulated pick-success metric
5. All runs tracked with MLflow

## Results (best run)

| Backbone | Aug | mAP | mask IoU (avg) | Est. pick success |
|----------|-----|-----|----------------|-------------------|
| R50-FPN | full | 0.441 | 0.689 | 71.2% |
| R101-FPN | full | 0.483 | 0.721 | 74.8% |
| R101-FPN | no-occ | 0.412 | 0.663 | 67.1% |
| R50-FPN | focal | 0.458 | 0.704 | 73.0% |

The occlusion augmentation matters a lot — 5-7% swing in pick success when you turn it off. ResNet-101 helps but the gain is smaller than I expected given the extra compute.

## Setup

```bash
# clone and install
git clone https://github.com/ag-siddharth/cluttered-bin-segmentation.git
cd cluttered-bin-segmentation
pip install -r requirements.txt
```

Python 3.9+ required. Tested on Ubuntu 22.04 with CUDA 11.8.

### Data

The training data is COCO-format. I used the [OCID dataset](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/) converted to COCO annotations, but any COCO-format segmentation dataset works. See `data/README.md` for how to set it up.

```
data/
  train/
    images/
    annotations.json
  val/
    images/
    annotations.json
  test/
    images/
    annotations.json
```

## Training

Basic training run:

```bash
python src/train.py --config configs/resnet101_fpn.yaml --experiment-name r101_baseline
```

The config files control everything — backbone, learning rate schedule, augmentation flags, etc. MLflow UI shows up at `localhost:5000` after you run `mlflow ui` in the project root.

### Running ablations

```bash
bash scripts/run_ablations.sh
```

This kicks off the four main runs (r50 baseline, r101 baseline, no-occlusion, focal loss). Takes ~8 hours total on a single 3090. You can also cherry-pick:

```bash
python src/train.py --config configs/ablations/focal_loss.yaml --experiment-name focal_ablation
```

## Evaluation

```bash
# standard mAP / mask IoU
python src/evaluate.py --config configs/resnet101_fpn.yaml --checkpoint results/r101_best.pth

# pick-success estimation
python src/pick_eval.py --predictions results/r101_preds.json --ground-truth data/test/annotations.json
```

The pick evaluator uses mask completeness + boundary quality as a proxy for whether a suction or parallel-jaw gripper could successfully grasp the item. Not a real physics sim, but correlates decently with the numbers in the RightHand Robotics papers.

## Visualization

```bash
python scripts/visualize_preds.py --image data/test/images/scene_042.png \
    --checkpoint results/r101_best.pth --threshold 0.6
```

Or check out `notebooks/eda_and_analysis.ipynb` for dataset stats and per-category breakdowns.

## Project structure

```
├── configs/             # YAML training configs
│   └── ablations/       # ablation-specific overrides
├── src/
│   ├── train.py         # training loop + MLflow logging
│   ├── evaluate.py      # mAP, mask IoU computation
│   ├── pick_eval.py     # offline-to-pick-success mapping
│   ├── dataset.py       # COCO-format dataset loader
│   ├── augmentations.py # custom augs incl. occlusion synthesis
│   ├── model.py         # model factory (backbone selection, loss swaps)
│   └── utils.py         # config loading, checkpoint helpers, misc
├── scripts/
│   ├── run_ablations.sh
│   ├── visualize_preds.py
│   └── convert_ocid.py  # OCID -> COCO format converter
├── notebooks/
│   └── eda_and_analysis.ipynb
├── data/                # (not tracked — see data/README.md)
└── results/             # checkpoints + prediction JSONs
```

## References

- He et al., "Mask R-CNN", ICCV 2017
- OCID dataset: Suchi et al., "EasyLabel: A Semi-Automatic Pixel-wise Object Annotation Tool for Creating Robotic RGB-D Datasets", ICRA 2019
- Torchvision Mask R-CNN implementation docs
- Zeng et al., "Robotic Pick-and-Place of Novel Objects in Clutter with Multi-Affordance Grasping and Cross-Domain Image Matching" (for pick-success framing)
