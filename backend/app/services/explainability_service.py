from typing import Any


class ExplainabilityService:
    """
    Explainability service.

    The final implementation will generate Grad-CAM
    heatmaps from the trained PyTorch model.
    """

    def generate_explanation(
        self,
        image_bytes: bytes,
        predictions: dict[str, float],
    ) -> dict[str, Any]:
        """
        Generate explainability information.

        Currently returns an empty result until the
        trained deep-learning model is integrated.
        """

        return {
            "heatmap": None,
        }