from io import BytesIO

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from ..config import settings


async def validate_image_upload(
    file: UploadFile,
) -> bytes:
    """
    Validate an uploaded image.

    Checks:
    - filename exists
    - MIME type
    - file size
    - image can actually be decoded
    """

    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename was provided.",
        )

    if (
        file.content_type
        not in settings.ALLOWED_IMAGE_TYPES
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Unsupported image type. "
                "Allowed formats: JPG, PNG, WEBP, BMP."
            ),
        )

    content = await file.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if len(content) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"Image exceeds the maximum allowed "
                f"size of {settings.MAX_FILE_SIZE_MB} MB."
            ),
        )

    # Validate using Pillow.
    try:
        image = Image.open(BytesIO(content))

        # Forces Pillow to actually decode the image.
        image.verify()

    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "The uploaded file is not a valid "
                "or readable image."
            ),
        )

    # Additional OpenCV validation.
    array = np.frombuffer(
        content,
        dtype=np.uint8,
    )

    decoded = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "The image could not be decoded "
                "by the image processing engine."
            ),
        )

    if (
        decoded.shape[0] < 16
        or decoded.shape[1] < 16
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Image dimensions are too small "
                "for reliable quality analysis."
            ),
        )

    return content