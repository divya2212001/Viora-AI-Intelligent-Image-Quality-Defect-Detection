"""Backward-compatible Grad-CAM import without a second hook implementation."""

from app.services.explainability_service import generate_gradcam

__all__ = ["generate_gradcam"]
