from __future__ import annotations

import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


class DropoutFastRCNNPredictor(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, p: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        self.cls_score = nn.Linear(in_channels, num_classes)
        self.bbox_pred = nn.Linear(in_channels, num_classes * 4)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        scores = self.cls_score(x)
        bbox_deltas = self.bbox_pred(x)
        return scores, bbox_deltas


class DropoutMaskRCNNPredictor(nn.Module):
    def __init__(self, in_channels: int, dim_reduced: int, num_classes: int, p: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=p)
        self.conv5_mask = nn.ConvTranspose2d(in_channels, dim_reduced, 2, 2, 0)
        self.relu = nn.ReLU(inplace=True)
        self.mask_fcn_logits = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)

        for name, param in self.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param, mode="fan_out", nonlinearity="relu")
            elif "bias" in name:
                nn.init.constant_(param, 0)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)
        x = self.conv5_mask(x)
        x = self.relu(x)
        return self.mask_fcn_logits(x)


def build_maskrcnn(*, num_classes: int, mc_dropout_p: float = 0.0) -> nn.Module:
    model = maskrcnn_resnet50_fpn(weights=None, weights_backbone=None)

    # Replace predictors for the requested number of classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    if mc_dropout_p and mc_dropout_p > 0:
        model.roi_heads.box_predictor = DropoutFastRCNNPredictor(in_features, num_classes, p=mc_dropout_p)
    else:
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    dim_reduced = model.roi_heads.mask_predictor.conv5_mask.out_channels
    if mc_dropout_p and mc_dropout_p > 0:
        model.roi_heads.mask_predictor = DropoutMaskRCNNPredictor(
            in_features_mask, dim_reduced, num_classes, p=mc_dropout_p
        )
    else:
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, dim_reduced, num_classes)

    return model


def set_dropout_train_only(model: nn.Module, enabled: bool) -> None:
    # For MC dropout at inference: keep dropout modules in train mode while the rest stays eval.
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train(enabled)

