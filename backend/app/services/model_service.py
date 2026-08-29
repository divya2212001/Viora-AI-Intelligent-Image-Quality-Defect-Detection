from typing import Any

from ml.predict import QualityPredictor

from ..config import settings


class ModelService:
    """
    Production inference service for the trained
    Hybrid CNN + Computer Vision model.
    """

    def __init__(self) -> None:

        self.name = settings.MODEL_NAME

        self.version = settings.MODEL_VERSION

        # Load the trained model once.
        self.predictor = QualityPredictor()

    def predict(
        self,
        image_bytes: bytes,
    ) -> dict[str, Any]:
        """
        Run the trained Hybrid CNN + CV model.
        """

        return self.predictor.predict(
            image_bytes
        )

    def get_model_info(
        self,
    ) -> dict[str, str]:

        return {
            "name": self.name,
            "version": self.version,
            "architecture": (
                "Hybrid CNN + Computer Vision"
            ),
        }