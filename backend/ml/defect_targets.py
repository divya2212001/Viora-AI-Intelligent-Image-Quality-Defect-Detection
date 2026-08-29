"""Deterministic, labelled synthetic degradations for defect-head training.

KonIQ++ provides perceptual quality scores and five broad annotation
frequencies, but it does not contain labels for the application's six required
defect categories.  This module creates known degradations only after the
train/validation/test split, so the held-out image identities are never used
for fitting.  These labels are therefore provenance-backed synthetic targets,
not renamed KonIQ++ annotations.
"""

from hashlib import sha256

import cv2
import numpy as np

from ml.config import DEFECT_NAMES


def _seed(filename: str) -> int:
    return int.from_bytes(sha256(filename.encode("utf-8")).digest()[:8], "little")


def target_for_filename(filename: str) -> tuple[str | None, np.ndarray]:
    """Assign one reproducible class, with one seventh of samples kept clean."""
    choice = _seed(filename) % (len(DEFECT_NAMES) + 1)
    labels = np.zeros(len(DEFECT_NAMES), dtype=np.float32)
    if choice == len(DEFECT_NAMES):
        return None, labels
    labels[choice] = 1.0
    return DEFECT_NAMES[choice], labels


def apply_synthetic_defect(image: np.ndarray, filename: str) -> tuple[np.ndarray, np.ndarray]:
    """Apply the labelled degradation selected for *filename* to a BGR image."""
    defect, labels = target_for_filename(filename)
    if defect is None:
        return image, labels

    rng = np.random.default_rng(_seed(filename))
    if defect == "blur":
        return cv2.GaussianBlur(image, (0, 0), sigmaX=float(rng.uniform(2.0, 4.0))), labels
    if defect == "underexposure":
        return cv2.convertScaleAbs(image, alpha=float(rng.uniform(0.22, 0.45)), beta=0), labels
    if defect == "overexposure":
        return cv2.convertScaleAbs(image, alpha=float(rng.uniform(1.4, 1.9)), beta=int(rng.integers(35, 80))), labels
    if defect == "noise":
        noise = rng.normal(0, float(rng.uniform(20, 38)), image.shape).astype(np.float32)
        return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8), labels
    if defect == "corruption":
        # A severe JPEG encode/decode and pixelation preserve a decodable image
        # while supplying a reproducible corruption target.
        height, width = image.shape[:2]
        small = cv2.resize(image, (max(8, width // 6), max(8, height // 6)), interpolation=cv2.INTER_AREA)
        degraded = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
        ok, encoded = cv2.imencode(".jpg", degraded, [cv2.IMWRITE_JPEG_QUALITY, int(rng.integers(8, 20))])
        return (cv2.imdecode(encoded, cv2.IMREAD_COLOR) if ok else degraded), labels

    # Potential visual defect: an opaque scratch/occlusion, not an unrelated
    # source-dataset category.
    result = image.copy()
    height, width = result.shape[:2]
    x1, y1 = int(rng.integers(0, max(1, width // 2))), int(rng.integers(0, max(1, height // 2)))
    x2, y2 = min(width - 1, x1 + max(8, width // 5)), min(height - 1, y1 + max(8, height // 5))
    cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 0), thickness=-1)
    cv2.line(result, (0, int(rng.integers(height))), (width - 1, int(rng.integers(height))), (255, 255, 255), thickness=max(2, width // 120))
    return result, labels
