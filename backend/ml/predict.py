import cv2
import numpy as np
import torch

from pathlib import Path

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

        # =====================================================
        # DEVICE
        # =====================================================

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )


        # =====================================================
        # MODEL PATH
        # =====================================================

        model_path = (
            ARTIFACTS_DIR
            / "image_quality_model.pt"
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )


        # =====================================================
        # LOAD MODEL CHECKPOINT
        # =====================================================

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )


        # =====================================================
        # CREATE HYBRID MODEL
        # =====================================================

        self.model = ImageQualityNet(
            num_features=len(
                FEATURE_NAMES
            ),
            num_defects=len(
                DEFECT_NAMES
            ),
        )


        # =====================================================
        # LOAD TRAINED WEIGHTS
        # =====================================================

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.to(
            self.device
        )

        self.model.eval()


        # =====================================================
        # LOAD FEATURE NORMALIZATION
        # =====================================================

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
            self.feature_std == 0,
            1.0,
            self.feature_std,
        )


        print(
            "QualityPredictor loaded successfully."
        )


    # =========================================================
    # DECODE IMAGE
    # =========================================================

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


    # =========================================================
    # PREPARE MODEL INPUTS
    # =========================================================

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

        # -----------------------------------------------------
        # 1. Decode image
        # -----------------------------------------------------

        image = self.decode_image(
            image_bytes
        )


        # -----------------------------------------------------
        # 2. Extract OpenCV features
        # -----------------------------------------------------

        features = extract_features(
            image
        )

        features = np.asarray(
            features,
            dtype=np.float32,
        )


        # -----------------------------------------------------
        # 3. Validate feature count
        # -----------------------------------------------------

        if len(features) != len(
            FEATURE_NAMES
        ):

            raise ValueError(
                "Feature count mismatch. "
                f"Expected {len(FEATURE_NAMES)}, "
                f"got {len(features)}."
            )


        # -----------------------------------------------------
        # 4. Normalize CV features
        # -----------------------------------------------------

        normalized_features = (
            features
            - self.feature_mean
        ) / self.feature_std

        normalized_features = (
            normalized_features.astype(
                np.float32
            )
        )


        # -----------------------------------------------------
        # 5. Prepare image for ResNet18
        # -----------------------------------------------------

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


        # -----------------------------------------------------
        # 6. Prepare CV feature tensor
        # -----------------------------------------------------

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


    # =========================================================
    # PREDICTION
    # =========================================================

    def predict(
        self,
        image_bytes,
    ):

        # -----------------------------------------------------
        # Prepare inputs
        # -----------------------------------------------------

        (
            image_tensor,
            feature_tensor,
        ) = self.prepare_inputs(
            image_bytes
        )


        # -----------------------------------------------------
        # Model inference
        # -----------------------------------------------------

        with torch.no_grad():

            quality, defects = (
                self.model(
                    image_tensor,
                    feature_tensor,
                )
            )


        # -----------------------------------------------------
        # Quality
        # -----------------------------------------------------

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


        # -----------------------------------------------------
        # Defects
        # -----------------------------------------------------

        defects = (
            defects
            .cpu()
            .numpy()[0]
        )


        # -----------------------------------------------------
        # qMOS
        # -----------------------------------------------------

        qmos = (
            quality * 5.0
        )


        # -----------------------------------------------------
        # Quality score
        # -----------------------------------------------------

        quality_score = (
            quality * 100.0
        )


        # -----------------------------------------------------
        # Defect dictionary
        # -----------------------------------------------------

        defect_scores = {

            name: round(
                float(value),
                6,
            )

            for name, value in zip(
                DEFECT_NAMES,
                defects,
            )
        }


        # -----------------------------------------------------
        # Normalized CV features
        # -----------------------------------------------------

        features = (
            feature_tensor
            .detach()
            .cpu()
            .numpy()[0]
        )


        # -----------------------------------------------------
        # Final result
        # -----------------------------------------------------

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

                name: round(
                    float(value),
                    6,
                )

                for name, value in zip(
                    FEATURE_NAMES,
                    features,
                )
            },
        }


    # =========================================================
    # GRAD-CAM
    # =========================================================

    def generate_gradcam(
        self,
        image_bytes,
    ):
        """
        Generate a Grad-CAM explanation for the
        predicted image quality.

        The Grad-CAM is generated from the final
        convolutional layer of ResNet18.
        """

        # -----------------------------------------------------
        # Import here to avoid circular imports
        # -----------------------------------------------------

        from app.services.explainability_service import (
            generate_gradcam,
        )


        # -----------------------------------------------------
        # Prepare inputs
        # -----------------------------------------------------

        (
            image_tensor,
            feature_tensor,
        ) = self.prepare_inputs(
            image_bytes
        )


        # -----------------------------------------------------
        # Get original image
        # -----------------------------------------------------

        original_image = (
            self.decode_image(
                image_bytes
            )
        )


        # -----------------------------------------------------
        # Grad-CAM output directory
        # -----------------------------------------------------

        from app.config import settings

        output_dir = (
            Path(
                settings.UPLOAD_DIR
            )
            / "gradcam"
        )


        # -----------------------------------------------------
        # Generate heatmap
        # -----------------------------------------------------

        filename = generate_gradcam(

            model=self.model,

            image_tensor=image_tensor,

            feature_tensor=feature_tensor,

            original_image=original_image,

            output_dir=output_dir,
        )


        return filename