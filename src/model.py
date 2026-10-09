"""
Model setup — wraps torchvision Mask R-CNN with backbone selection
and optional focal loss. Most of this is boilerplate.
"""

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator  # not used rn but might need for custom anchors


def build_model(cfg, num_classes):
    model_cfg = cfg['model']
    backbone_name = model_cfg.get('backbone', 'resnet50_fpn')
    pretrained = model_cfg.get('pretrained', True)

    if backbone_name == 'resnet50_fpn':
        if pretrained:
            weights = torchvision.models.detection.MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        else:
            weights = None
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=weights)

    elif backbone_name == 'resnet101_fpn':
        # torchvision doesn't have a convenience function for r101 mask rcnn
        # so we build it from the backbone
        if pretrained:
            backbone_weights = torchvision.models.ResNet101_Weights.DEFAULT
        else:
            backbone_weights = None
        backbone = torchvision.models.resnet101(weights=backbone_weights)
        backbone = _make_fpn_backbone(backbone)

        model = MaskRCNN(
            backbone,
            num_classes=num_classes,
            min_size=model_cfg.get('min_size', 640),
            max_size=model_cfg.get('max_size', 1024),
            box_score_thresh=model_cfg.get('box_score_thresh', 0.5),
            box_nms_thresh=model_cfg.get('box_nms_thresh', 0.4),
        )
        return model  # already has right num_classes

    # elif backbone_name == 'mobilenet_v3':
    #     # tried this for faster inference, accuracy was garbage
    #     model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)
    #     # would need to swap out the backbone properly, gave up for now

    else:
        raise ValueError(f"Unknown backbone: {backbone_name}. Use resnet50_fpn or resnet101_fpn")

    # replace the classifier head for our num_classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # replace the mask head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # adjust thresholds from config
    model.roi_heads.score_thresh = model_cfg.get('box_score_thresh', 0.5)
    model.roi_heads.nms_thresh = model_cfg.get('box_nms_thresh', 0.4)

    return model


def _make_fpn_backbone(resnet_model):
    """Convert a ResNet model into an FPN backbone for Mask R-CNN."""
    from torchvision.models.detection.backbone_utils import BackboneWithFPN
    from torchvision.ops.feature_pyramid_network import LastLevelMaxPool

    # grab the layers we need
    return_layers = {
        'layer1': '0',
        'layer2': '1',
        'layer3': '2',
        'layer4': '3',
    }

    # figure out channel sizes — r101 same as r50 for these
    in_channels_list = [256, 512, 1024, 2048]

    backbone = BackboneWithFPN(
        resnet_model,
        return_layers=return_layers,
        in_channels_list=in_channels_list,
        out_channels=256,
        extra_blocks=LastLevelMaxPool(),
    )
    backbone.out_channels = 256
    return backbone


class FocalLoss(nn.Module):
    """Focal loss — downweights easy examples, from the RetinaNet paper."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probability of correct class
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * ce_loss).mean()


def maybe_swap_loss(model, cfg):
    # monkey-patch the loss if config says focal. hacky but it works
    loss_cfg = cfg.get('loss', {})
    loss_type = loss_cfg.get('type', 'default')

    if loss_type == 'focal':
        alpha = loss_cfg.get('focal_alpha', 0.25)
        gamma = loss_cfg.get('focal_gamma', 2.0)
        # store it so we can use it in the custom forward
        model._focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        print(f"Using focal loss (alpha={alpha}, gamma={gamma})")
    else:
        print("Using default cross-entropy loss")

    return model
