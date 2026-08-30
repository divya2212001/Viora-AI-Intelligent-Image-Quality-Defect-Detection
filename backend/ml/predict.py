import threading

import cv2
import numpy as np
import torch

from ml.config import ARTIFACTS_DIR, DEFECT_NAMES, FEATURE_NAMES, IMAGE_SIZE
from ml.features import extract_features
from ml.model import ImageQualityNet


class QualityPredictor:
    """One process-wide model instance used by both inference and Grad-CAM."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_lock = threading.RLock()

        model_path = ARTIFACTS_DIR / "image_quality_model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.defect_names = checkpoint.get("defect_names", DEFECT_NAMES)
        if not isinstance(self.defect_names, list) or not self.defect_names:
            raise ValueError("Checkpoint does not declare defect-label metadata.")

        self.model = ImageQualityNet(
            num_features=len(FEATURE_NAMES),
            num_defects=len(self.defect_names),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        del checkpoint

        self.model.to(self.device)
        self.model.eval()
        # Grad-CAM differentiates with respect to its input tensor, not weights.
        # Keeping parameter gradients disabled avoids retaining parameter .grad buffers.
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.feature_mean = np.load(ARTIFACTS_DIR / "feature_mean.npy")
        self.feature_std = np.load(ARTIFACTS_DIR / "feature_std.npy")
        self.feature_std = np.where(np.abs(self.feature_std) < 1e-8, 1.0, self.feature_std)
        if (
            self.feature_mean.shape != (len(FEATURE_NAMES),)
            or self.feature_std.shape != (len(FEATURE_NAMES),)
        ):
            raise ValueError("Feature normalization artifacts do not match the model feature count.")

        print("QualityPredictor loaded successfully.")

    def decode_image(self, image_bytes: bytes) -> np.ndarray:
        encoded = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image.")
        return image

    def resize_for_inference(
        self,
        image: np.ndarray,
        max_dimension: int = 1536,
    ) -> np.ndarray:
        """Downscale only oversized inputs while preserving aspect ratio."""
        height, width = image.shape[:2]
        largest_dimension = max(height, width)
        if largest_dimension <= max_dimension:
            return image

        scale = max_dimension / largest_dimension
        resized_width = max(1, round(width * scale))
        resized_height = max(1, round(height * scale))
        return cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

    def prepare_inputs_from_image(
        self,
        image: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build model tensors from an already-decoded, bounded BGR image."""
        features = np.asarray(extract_features(image), dtype=np.float32)
        if len(features) != len(FEATURE_NAMES):
            raise ValueError(
                f"Feature count mismatch. Expected {len(FEATURE_NAMES)}, got {len(features)}."
            )

        normalized_features = ((features - self.feature_mean) / self.feature_std).astype(
            np.float32,
            copy=False,
        )
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(
            rgb,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_AREA,
        )

        image_tensor = torch.from_numpy(rgb).permute(2, 0, 1).to(dtype=torch.float32)
        image_tensor.div_(255.0)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        feature_tensor = torch.from_numpy(normalized_features).unsqueeze(0).to(self.device)
        return image_tensor, feature_tensor

    def prepare_inputs(self, image_bytes: bytes) -> tuple[torch.Tensor, torch.Tensor]:
        """Compatibility helper for callers that supply encoded image bytes."""
        image = self.decode_image(image_bytes)
        try:
            return self.prepare_inputs_from_image(image)
        finally:
            del image

    def predict_from_image(self, image: np.ndarray) -> dict:
        """Run normal inference with no autograd graph or retained tensor outputs."""
        image_tensor = feature_tensor = quality_tensor = defect_tensor = None
        try:
            image_tensor, feature_tensor = self.prepare_inputs_from_image(image)
            # Inference mode is deliberately limited to the non-explainability path.
            with self.model_lock, torch.inference_mode():
                self.model.eval()
                quality_tensor, defect_tensor = self.model(image_tensor, feature_tensor)

            quality = float(quality_tensor.item())
            defects = defect_tensor.squeeze(0).cpu().tolist()
            raw_features = extract_features(image)
        finally:
            # Python releases CPU tensors promptly; CUDA allocations are handled by PyTorch.
            del image_tensor, feature_tensor, quality_tensor, defect_tensor

        quality = max(0.0, min(1.0, quality))
        qmos = quality * 5.0
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

        return {
            "quality_score": round(quality * 100.0, 2),
            "qmos": round(qmos, 4),
            "defects": {
                name: round(float(value), 6)
                for name, value in zip(self.defect_names, defects)
            },
            "quality_label": quality_label,
            "statistics": {
                name: round(float(value), 6)
                for name, value in zip(FEATURE_NAMES, raw_features)
            },
        }

    def predict(self, image_bytes: bytes) -> dict:
        image = self.decode_image(image_bytes)
        try:
            inference_image = self.resize_for_inference(image)
            try:
                return self.predict_from_image(inference_image)
            finally:
                if inference_image is not image:
                    del inference_image
        finally:
            del image
