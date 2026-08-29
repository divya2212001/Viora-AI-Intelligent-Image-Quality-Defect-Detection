import torch
import torch.nn as nn

from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


class ImageQualityNet(
    nn.Module
):

    def __init__(
        self,
        num_features=8,
        num_defects=6,
    ):

        super().__init__()

        # PRETRAINED CNN

        backbone = resnet18(
            weights=ResNet18_Weights.DEFAULT
        )

        # Original ResNet classifier.
        cnn_features = (
            backbone.fc.in_features
        )

        backbone.fc = nn.Identity()

        self.backbone = backbone

        # OpenCV FEATURE BRANCH

        self.feature_branch = nn.Sequential(

            nn.Linear(
                num_features,
                32,
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                32
            ),

            nn.Dropout(
                0.15
            ),
        )

        # FUSION

        self.fusion = nn.Sequential(

            nn.Linear(
                cnn_features + 32,
                128,
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                128
            ),

            nn.Dropout(
                0.30
            ),
        )

        # QUALITY HEAD

        self.quality_head = nn.Sequential(

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                1,
            ),

            nn.Sigmoid(),
        )

        # DEFECT HEAD

        self.defect_head = nn.Sequential(

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                64,
                num_defects,
            ),

            nn.Sigmoid(),
        )

    def forward(
        self,
        image,
        features,
    ):

        image_embedding = (
            self.backbone(
                image
            )
        )

        feature_embedding = (
            self.feature_branch(
                features
            )
        )

        combined = torch.cat(
            [
                image_embedding,
                feature_embedding,
            ],
            dim=1,
        )

        fused = self.fusion(
            combined
        )

        quality = (
            self.quality_head(
                fused
            )
            .squeeze(1)
        )

        defects = (
            self.defect_head(
                fused
            )
        )

        return (
            quality,
            defects,
        )
