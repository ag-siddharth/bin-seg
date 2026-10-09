"""
Training script for Mask R-CNN on cluttered bin images.

Usage:
    python src/train.py --config configs/resnet101_fpn.yaml --experiment-name r101_baseline
    python src/train.py --config configs/ablations/focal_loss.yaml --experiment-name focal_abl
"""

import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
import mlflow

from dataset import build_dataset
from augmentations import BinAugmentation, ValTransform
from model import build_model, maybe_swap_loss
from evaluate import evaluate_coco
from utils import (
    load_config, save_checkpoint, set_seed,
    collate_fn, AverageMeter, get_device
)

# uncomment to profile GPU memory issues
# torch.cuda.memory._record_memory_history()


def train_one_epoch(model, dataloader, optimizer, device, epoch, cfg):
    model.train()
    loss_meter = AverageMeter()
    log_interval = cfg['logging'].get('log_interval', 50)

    for batch_idx, (images, targets) in enumerate(dataloader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # sanity check — sometimes get nan with bad data or lr too high
        if not torch.isfinite(losses):
            print(f"WARNING: non-finite loss at batch {batch_idx}, skipping")
            optimizer.zero_grad()
            continue

        optimizer.zero_grad()
        losses.backward()

        # optional gradient clipping
        grad_clip = cfg['training'].get('grad_clip')
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        optimizer.step()

        loss_meter.update(losses.item(), len(images))

        if (batch_idx + 1) % log_interval == 0:
            print(f"  [{batch_idx+1}/{len(dataloader)}] loss: {loss_meter.avg:.4f} "
                  f"(cls: {loss_dict.get('loss_classifier', 0):.3f} "
                  f"mask: {loss_dict.get('loss_mask', 0):.3f} "
                  f"box: {loss_dict.get('loss_box_reg', 0):.3f})")

    return loss_meter.avg


def build_lr_scheduler(optimizer, cfg):
    sched_type = cfg['training'].get('lr_scheduler', 'step')
    # FIXME: add warmup support here instead of the standalone warmup_lr function
    if sched_type == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg['training'].get('lr_step_size', 10),
            gamma=cfg['training'].get('lr_gamma', 0.1),
        )
    elif sched_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg['training']['epochs']
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")
    return scheduler


def warmup_lr(optimizer, epoch, batch_idx, num_batches, cfg):
    # linear warmup — wrote this but never integrated it into the loop
    # keeping around in case I revisit
    warmup_epochs = cfg['training'].get('warmup_epochs', 1)
    if epoch >= warmup_epochs:
        return

    warmup_factor = cfg['training'].get('warmup_factor', 0.001)
    base_lr = cfg['training']['lr']

    # linear interpolation
    progress = (epoch * num_batches + batch_idx) / (warmup_epochs * num_batches)
    lr = warmup_factor + (1.0 - warmup_factor) * progress
    for pg in optimizer.param_groups:
        pg['lr'] = base_lr * lr


def main():
    parser = argparse.ArgumentParser(description='Train Mask R-CNN for bin segmentation')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--experiment-name', type=str, default='default',
                        help='MLflow experiment name')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(args.seed)
    device = get_device()

    # output setup
    output_dir = cfg['logging'].get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)

    # datasets
    train_transforms = BinAugmentation(cfg)
    val_transforms = ValTransform()

    train_dataset = build_dataset(cfg, split='train', transforms=train_transforms)
    val_dataset = build_dataset(cfg, split='val', transforms=val_transforms)

    num_classes = train_dataset.get_num_classes()
    print(f"Number of classes (incl background): {num_classes}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=cfg['data'].get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=cfg['data'].get('pin_memory', True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # always 1 for eval to avoid padding issues
        shuffle=False,
        num_workers=cfg['data'].get('num_workers', 4),
        collate_fn=collate_fn,
        pin_memory=cfg['data'].get('pin_memory', True),
    )

    # model
    model = build_model(cfg, num_classes)
    model = maybe_swap_loss(model, cfg)
    model.to(device)

    # optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=cfg['training']['lr'],
        momentum=cfg['training'].get('momentum', 0.9),
        weight_decay=cfg['training'].get('weight_decay', 0.0005),
    )

    scheduler = build_lr_scheduler(optimizer, cfg)

    start_epoch = 0
    best_map = 0.0

    if args.resume:
        print(f"Resuming from {args.resume}")
        from utils import load_checkpoint
        start_epoch, metrics = load_checkpoint(args.resume, model, optimizer)
        best_map = metrics.get('mAP', 0.0)
        print(f"Resumed at epoch {start_epoch}, best mAP so far: {best_map:.4f}")

    # MLflow setup
    mlflow.set_tracking_uri(cfg['logging'].get('mlflow_tracking_uri', 'mlruns'))
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run():
        # log config
        mlflow.log_params({
            'backbone': cfg['model']['backbone'],
            'batch_size': cfg['training']['batch_size'],
            'lr': cfg['training']['lr'],
            'epochs': cfg['training']['epochs'],
            'augmentation_occlusion': cfg['augmentation'].get('occlusion_synthesis', True),
            'loss_type': cfg.get('loss', {}).get('type', 'default'),
            'num_classes': num_classes,
            'seed': args.seed,
        })
        mlflow.log_artifact(args.config)

        for epoch in range(start_epoch, cfg['training']['epochs']):
            t0 = time.time()
            print(f"\n--- Epoch {epoch+1}/{cfg['training']['epochs']} "
                  f"(lr: {optimizer.param_groups[0]['lr']:.6f}) ---")

            train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, cfg)
            scheduler.step()

            elapsed = time.time() - t0
            print(f"Epoch {epoch+1} done in {elapsed:.1f}s, avg loss: {train_loss:.4f}")

            mlflow.log_metrics({
                'train_loss': train_loss,
                'lr': optimizer.param_groups[0]['lr'],
            }, step=epoch)

            # evaluate
            metrics = evaluate_coco(model, val_loader, device)
            print(f"Val mAP: {metrics['mAP']:.4f}, "
                  f"mAP@50: {metrics['mAP_50']:.4f}, "
                  f"mask IoU: {metrics['avg_mask_iou']:.4f}")

            mlflow.log_metrics({
                'val_mAP': metrics['mAP'],
                'val_mAP_50': metrics['mAP_50'],
                'val_mask_iou': metrics['avg_mask_iou'],
            }, step=epoch)

            # save best
            if metrics['mAP'] > best_map:
                best_map = metrics['mAP']
                best_path = os.path.join(output_dir, f'{args.experiment_name}_best.pth')
                save_checkpoint(model, optimizer, epoch, metrics, best_path)
                print(f"New best mAP: {best_map:.4f}, saved to {best_path}")

            # periodic saves
            save_every = cfg['logging'].get('save_every', 5)
            if (epoch + 1) % save_every == 0:
                ckpt_path = os.path.join(output_dir, f'{args.experiment_name}_ep{epoch+1}.pth')
                save_checkpoint(model, optimizer, epoch, metrics, ckpt_path)

        print(f"\nTraining complete. Best mAP: {best_map:.4f}")
        mlflow.log_metric('best_mAP', best_map)


if __name__ == '__main__':
    main()
