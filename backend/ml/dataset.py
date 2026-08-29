import cv2
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from ml.config import (
    DEFECT_NAMES,
    IMAGE_DIR,
    IMAGE_SIZE,
)

from ml.features import extract_features
from ml.defect_targets import apply_synthetic_defect


class KonIQDataset(
    Dataset
):

    def __init__(
        self,
        csv_path,
        feature_mean=None,
        feature_std=None,
        training=False,
        synthetic_defects=False,
    ):

        self.df = pd.read_csv(
            csv_path
        )

        self.feature_mean = (
            feature_mean
        )

        self.feature_std = (
            feature_std
        )

        self.training = training
        self.synthetic_defects = synthetic_defects

    def __len__(self):

        return len(
            self.df
        )

    def __getitem__(
        self,
        index,
    ):

        row = self.df.iloc[
            index
        ]

        image_path = (
            IMAGE_DIR
            / row["filename"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            raise RuntimeError(
                f"Could not read image: "
                f"{image_path}"
            )

        if self.synthetic_defects:
            image, defects = apply_synthetic_defect(
                image,
                str(row["filename"]),
            )
        else:
            defects = None

        features = extract_features(
            image
        )

        if (
            self.feature_mean
            is not None
            and self.feature_std
            is not None
        ):

            features = (
                features
                - self.feature_mean
            ) / self.feature_std

        # Image preprocessing
        rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        rgb = cv2.resize(
            rgb,
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),
            interpolation=cv2.INTER_AREA,
        )

        # Basic augmentation only during training.
        if self.training:

            if np.random.rand() < 0.5:

                rgb = np.fliplr(
                    rgb
                ).copy()

        image_tensor = (
            torch.from_numpy(
                rgb
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        feature_tensor = torch.tensor(
            features,
            dtype=torch.float32,
        )

        # qmos is approximately 1-5.
        # Normalize to 0-1 for training.
        quality = (
            float(row["qmos"])
            / 5.0
        )

        quality_tensor = torch.tensor(
            quality,
            dtype=torch.float32,
        )

        # Five human-annotation distortion frequencies.
        if defects is None:
            # The raw KonIQ++ columns are deliberately not mapped to the six
            # application classes.  Clean quality evaluation does not use
            # defect labels; synthetic_defects=True supplies valid targets.
            defects = np.zeros(len(DEFECT_NAMES), dtype=np.float32)

        defect_tensor = torch.tensor(
            defects,
            dtype=torch.float32,
        )

        return (
            image_tensor,
            feature_tensor,
            quality_tensor,
            defect_tensor,
        )
