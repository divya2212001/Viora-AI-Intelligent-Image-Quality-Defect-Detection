from pathlib import Path

import cv2
import numpy as np
import torch

from ml.config import (
    ARTIFACTS_DIR,
    DEFECT_NAMES,
    FEATURE_NAMES,
    IMAGE_SIZE,
)

from ml.features import extract_features

from ml.model import ImageQualityNet


class QualityPredictor:

    def __init__(
        self,
    ):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        model_path = (
            ARTIFACTS_DIR
            / "image_quality_model.pt"
        )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.model = ImageQualityNet(
            num_features=len(
                FEATURE_NAMES
            ),
            num_defects=len(
                DEFECT_NAMES
            ),
        )

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.to(
            self.device
        )

        self.model.eval()

        self.feature_mean = np.load(
            ARTIFACTS_DIR
            / "feature_mean.npy"
        )

        self.feature_std = np.load(
            ARTIFACTS_DIR
            / "feature_std.npy"
        )

    def predict(
        self,
        image_bytes,
    ):


        # Decode image bytes to OpenCV BGR image

        array = np.frombuffer(
            image_bytes,
            dtype=np.uint8,
        )

        image = cv2.imdecode(
            array,
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise ValueError(
                "Could not decode image."
            )


        # OpenCV features

        features = extract_features(
            image
        )

        features = (
            features
            - self.feature_mean
        ) / self.feature_std


        # Image


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

        image_tensor = (
            torch.from_numpy(
                rgb
            )
            .permute(2, 0, 1)
            .float()
            / 255.0
        )

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )

        feature_tensor = (
            torch.tensor(
                features,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(self.device)
        )

        # Prediction
        with torch.no_grad():

            quality, defects = (
                self.model(
                    image_tensor,
                    feature_tensor,
                )
            )

        quality = float(
            quality.cpu().item()
        )

        defects = (
            defects
            .cpu()
            .numpy()[0]
        )

        # qmos: 0-1 → 0-5
        qmos = (
            quality * 5.0
        )

        # qmos: 0-5 → 0-100
        quality_score = (
            quality * 100.0
        )

        defect_scores = {
            name: float(
                value
            )
            for name, value
            in zip(
                DEFECT_NAMES,
                defects,
            )
        }

        return {
            "quality_score":
                round(
                    quality_score,
                    2,
                ),

            "qmos":
                round(
                    qmos,
                    4,
                ),

            "defects":
                defect_scores,

            "statistics": {
                name: float(
                    value
                )
                for name, value
                in zip(
                    FEATURE_NAMES,
                    features,
                )
            },
        }