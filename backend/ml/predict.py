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

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        self.defect_names = checkpoint.get("defect_names", DEFECT_NAMES)
        if not isinstance(self.defect_names, list) or not self.defect_names:
            raise ValueError("Checkpoint does not declare defect-label metadata.")

        self.model = ImageQualityNet(
            num_features=len(
                FEATURE_NAMES
            ),
            num_defects=len(
                self.defect_names
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


        # Avoid division by zero.

        self.feature_std = np.where(
            np.abs(self.feature_std) < 1e-8,
            1.0,
            self.feature_std,
        )

        if self.feature_mean.shape != (len(FEATURE_NAMES),) or self.feature_std.shape != (len(FEATURE_NAMES),):
            raise ValueError("Feature normalization artifacts do not match the model feature count.")


        print(
            "QualityPredictor loaded successfully."
        )

    def decode_image(
        self,
        image_bytes,
    ):

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

        return image
    def prepare_inputs(
        self,
        image_bytes,
    ):
        """
        Prepare the two inputs required by the
        Hybrid CNN + Computer Vision model.

        Returns:

            image_tensor
            feature_tensor
        """

        image = self.decode_image(
            image_bytes
        )


        features = extract_features(
            image
        )

        features = np.asarray(
            features,
            dtype=np.float32,
        )


        if len(features) != len(
            FEATURE_NAMES
        ):

            raise ValueError(
                "Feature count mismatch. "
                f"Expected {len(FEATURE_NAMES)}, "
                f"got {len(features)}."
            )

        normalized_features = (
            features
            - self.feature_mean
        ) / self.feature_std

        normalized_features = (
            normalized_features.astype(
                np.float32
            )
        )

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


        # NumPy → Tensor

        image_tensor = (
            torch.from_numpy(
                rgb
            )
            .permute(
                2,
                0,
                1,
            )
            .float()
            / 255.0
        )


        # Add batch dimension

        image_tensor = (
            image_tensor
            .unsqueeze(0)
            .to(self.device)
        )


        feature_tensor = (
            torch.tensor(
                normalized_features,
                dtype=torch.float32,
            )
            .unsqueeze(0)
            .to(self.device)
        )


        return (
            image_tensor,
            feature_tensor,
        )

    def predict(
        self,
        image_bytes,
    ):


        (
            image_tensor,
            feature_tensor,
        ) = self.prepare_inputs(
            image_bytes
        )


        with torch.no_grad():

            quality, defects = (
                self.model(
                    image_tensor,
                    feature_tensor,
                )
            )


        raw_features = extract_features(self.decode_image(image_bytes))

        quality = float(
            quality.cpu().item()
        )


        # Keep quality between 0 and 1.

        quality = max(
            0.0,
            min(
                1.0,
                quality,
            ),
        )
        defects = (
            defects
            .cpu()
            .numpy()[0]
        )

        qmos = (
            quality * 5.0
        )

        if qmos >= 4.0:
            quality_label = "Excellent"
        elif qmos >= 3.5:
            quality_label = "Good"
        elif qmos >= 2.5:
            quality_label = "Fair"
        elif qmos >= 1.5:
            quality_label = "Poor"
        else:
            quality_label = "Very Poor"

        quality_score = (
            quality * 100.0
        )

        defect_scores = {

            name: round(
                float(value),
                6,
            )

            for name, value in zip(
                self.defect_names,
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

            "quality_label": quality_label,

            "statistics": {

                name: round(
                    float(value),
                    6,
                )

                for name, value in zip(
                    FEATURE_NAMES,
                    raw_features,
                )
            },
        }
