import cv2
import numpy as np


FEATURE_NAMES = [
    "brightness",
    "contrast",
    "sharpness",
    "noise_level",
    "entropy",
    "saturation",
    "dark_pixel_ratio",
    "bright_pixel_ratio",
]


def extract_features(
    image,
):
    """
    Extract interpretable image-quality
    features using OpenCV.

    Input:
        BGR uint8 image

    Output:
        numpy array of shape (8,)
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Brightness
    brightness = (
        float(
            np.mean(gray)
        )
        / 255.0
    )

    # Contrast
    contrast = (
        float(
            np.std(gray)
        )
        / 255.0
    )

    # Sharpness

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_64F,
    )

    sharpness_raw = float(
        laplacian.var()
    )

    sharpness = np.log1p(
        max(
            sharpness_raw,
            0.0,
        )
    )

    # Noise estimate

    smooth = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    residual = (
        gray.astype(
            np.float32
        )
        - smooth.astype(
            np.float32
        )
    )

    noise_level = float(
        np.std(residual)
        / 255.0
    )


    # Entropy
    histogram = cv2.calcHist(
        [gray],
        [0],
        None,
        [256],
        [0, 256],
    ).flatten()

    histogram = (
        histogram
        / (
            histogram.sum()
            + 1e-8
        )
    )

    valid = histogram > 0

    entropy = float(
        -np.sum(
            histogram[valid]
            * np.log2(
                histogram[valid]
            )
        )
        / 8.0
    )

    # Saturation
    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV,
    )

    saturation = float(
        np.mean(
            hsv[:, :, 1]
        )
        / 255.0
    )

    # Dark pixel ratio
    dark_pixel_ratio = float(
        np.mean(
            gray <= 20
        )
    )

    # Bright pixel ratio

    bright_pixel_ratio = float(
        np.mean(
            gray >= 235
        )
    )

    return np.array(
        [
            brightness,
            contrast,
            sharpness,
            noise_level,
            entropy,
            saturation,
            dark_pixel_ratio,
            bright_pixel_ratio,
        ],
        dtype=np.float32,
    )