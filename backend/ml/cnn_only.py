import torch
import torch.nn as nn
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
)


class CNNOnlyQualityNet(
    nn.Module
):

    def __init__(
        self,
        num_defects,
    ):

        super().__init__()

        self.backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        feature_dim = (
            self.backbone.fc.in_features
        )

        self.backbone.fc = nn.Identity()

        self.quality_head = nn.Sequential(
            nn.Linear(
                feature_dim,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                256,
                1,
            ),
            nn.Sigmoid(),
        )

        self.defect_head = nn.Sequential(
            nn.Linear(
                feature_dim,
                256,
            ),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(
                256,
                num_defects,
            ),
            nn.Sigmoid(),
        )

    def forward(
        self,
        images,
    ):

        embedding = self.backbone(
            images
        )

        quality = self.quality_head(
            embedding
        ).squeeze(1)

        defects = self.defect_head(
            embedding
        )

        return quality, defects