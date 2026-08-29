from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image


def decode_image(image_bytes: bytes) -> np.ndarray:
    """
    Decode image bytes into a BGR OpenCV image.
    """

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
            "Unable to decode image."
        )

    return image


def calculate_sharpness(
    gray: np.ndarray,
) -> float:
    """
    Variance of Laplacian.

    Higher values generally indicate stronger
    high-frequency structure / sharpness.
    """

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    return float(laplacian.var())


def calculate_brightness(
    gray: np.ndarray,
) -> float:
    """
    Mean grayscale intensity in [0, 255].
    """

    return float(np.mean(gray))


def calculate_contrast(
    gray: np.ndarray,
) -> float:
    """
    Standard deviation of grayscale intensity.
    """

    return float(np.std(gray))


def calculate_noise_level(
    gray: np.ndarray,
) -> float:
    """
    Estimate high-frequency noise by subtracting
    a Gaussian-smoothed image from the original.

    Normalized to approximately [0, 1].
    """

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    residual = (
        gray.astype(np.float32)
        - blurred.astype(np.float32)
    )

    noise_std = float(
        np.std(residual)
    )

    normalized = noise_std / 255.0

    return float(
        np.clip(normalized, 0.0, 1.0)
    )


def calculate_entropy(
    gray: np.ndarray,
) -> float:
    """
    Shannon entropy of grayscale intensity
    distribution.
    """

    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256],
    )

    probabilities = histogram.flatten()

    total = probabilities.sum()

    if total == 0:
        return 0.0

    probabilities /= total

    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log2(probabilities)
    )

    return float(entropy)


def calculate_saturation(
    image: np.ndarray,
) -> float:
    """
    Mean HSV saturation normalized to [0, 1].
    """

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    saturation = hsv[:, :, 1]

    return float(
        np.mean(saturation) / 255.0
    )


def calculate_exposure_ratios(
    gray: np.ndarray,
) -> tuple[float, float]:
    """
    Calculate proportions of very dark and
    very bright pixels.
    """

    total_pixels = gray.size

    if total_pixels == 0:
        return 0.0, 0.0

    dark_pixels = np.sum(
        gray <= 20
    )

    bright_pixels = np.sum(
        gray >= 235
    )

    dark_ratio = (
        dark_pixels / total_pixels
    )

    bright_ratio = (
        bright_pixels / total_pixels
    )

    return (
        float(dark_ratio),
        float(bright_ratio),
    )


def extract_features(
    image_bytes: bytes,
) -> dict[str, Any]:
    """
    Extract all image-quality features.
    """

    image = decode_image(image_bytes)

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    sharpness = calculate_sharpness(
        gray
    )

    brightness = calculate_brightness(
        gray
    )

    contrast = calculate_contrast(
        gray
    )

    noise_level = calculate_noise_level(
        gray
    )

    entropy = calculate_entropy(
        gray
    )

    saturation = calculate_saturation(
        image
    )

    (
        dark_pixel_ratio,
        bright_pixel_ratio,
    ) = calculate_exposure_ratios(
        gray
    )

    height, width = image.shape[:2]

    return {
        "brightness": round(
            brightness,
            4,
        ),

        "contrast": round(
            contrast,
            4,
        ),

        "sharpness": round(
            sharpness,
            4,
        ),

        "noise_level": round(
            noise_level,
            6,
        ),

        "entropy": round(
            entropy,
            4,
        ),

        "saturation": round(
            saturation,
            6,
        ),

        "dark_pixel_ratio": round(
            dark_pixel_ratio,
            6,
        ),

        "bright_pixel_ratio": round(
            bright_pixel_ratio,
            6,
        ),

        "width": width,
        "height": height,
    }