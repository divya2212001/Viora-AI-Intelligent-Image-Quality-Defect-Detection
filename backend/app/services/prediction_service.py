from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from ml.predict import QualityPredictor

from app.config import settings

from app.database import (
    predictions_collection,
)

from app.models.prediction import (
    create_prediction_document,
)

from app.services.explainability_service import (
    generate_gradcam,
)


# ---------------------------------------------------------
# Load model ONCE
# ---------------------------------------------------------

predictor = QualityPredictor()


# ---------------------------------------------------------
# Quality labels
# ---------------------------------------------------------

def get_quality_label(
    qmos: float,
) -> str:

    if qmos >= 4.0:
        return "Excellent"

    if qmos >= 3.5:
        return "Good"

    if qmos >= 2.5:
        return "Fair"

    if qmos >= 1.5:
        return "Poor"

    return "Very Poor"


# ---------------------------------------------------------
# Recommendation
# ---------------------------------------------------------

def get_recommendation(
    qmos: float,
    defects: dict,
) -> str:

    if defects:

        highest_defect = max(
            defects,
            key=defects.get,
        )

        highest_score = defects[
            highest_defect
        ]

    else:

        highest_defect = "quality issue"
        highest_score = 0.0


    if qmos >= 4.0:

        return (
            "Image quality is excellent "
            "with minimal detected defects."
        )

    if qmos >= 3.5:

        return (
            "Image quality is good. "
            "Minor quality issues may be present."
        )

    if qmos >= 2.5:

        return (
            f"Image quality is moderate. "
            f"The main detected issue is "
            f"{highest_defect}."
        )

    return (
        f"Image quality is low. "
        f"The most prominent detected issue "
        f"is {highest_defect} "
        f"({highest_score:.2f})."
    )


# ---------------------------------------------------------
# Main prediction
# ---------------------------------------------------------

def predict_image(
    image_bytes: bytes,
    filename: str,
):

    # -----------------------------------------------------
    # 1. Run ML prediction
    # -----------------------------------------------------

    prediction = predictor.predict(
        image_bytes
    )

    qmos = prediction["qmos"]

    defects = prediction["defects"]


    # -----------------------------------------------------
    # 2. Quality information
    # -----------------------------------------------------

    quality_label = get_quality_label(
        qmos
    )

    recommendation = get_recommendation(
        qmos,
        defects,
    )


    # -----------------------------------------------------
    # 3. Generate prediction ID
    # -----------------------------------------------------

    prediction_id = str(
        uuid4()
    )


    # -----------------------------------------------------
    # 4. Save original image
    # -----------------------------------------------------

    upload_dir = Path(
        settings.UPLOAD_DIR
    )

    upload_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_filename = (
        f"{prediction_id}.jpg"
    )

    image_path = (
        upload_dir
        / safe_filename
    )


    # Decode image

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
            "Could not decode uploaded image."
        )


    # Save original image

    cv2.imwrite(
        str(image_path),
        image,
    )


    # -----------------------------------------------------
    # 5. Generate Grad-CAM
    # -----------------------------------------------------

    gradcam_url = None

    try:

        gradcam_filename = (
            predictor.generate_gradcam(
                image_bytes
            )
        )

        if gradcam_filename:

            gradcam_url = (
                f"/uploads/gradcam/"
                f"{gradcam_filename}"
            )

    except Exception as error:

        print(
            "WARNING: Grad-CAM generation failed:"
        )

        print(error)


    # -----------------------------------------------------
    # 6. Image URL
    # -----------------------------------------------------

    image_url = (
        f"/uploads/{safe_filename}"
    )


    # -----------------------------------------------------
    # 7. Final result
    # -----------------------------------------------------

    result = {

        "prediction_id":
            prediction_id,

        "filename":
            filename,

        "image_url":
            image_url,

        "quality_score":
            prediction[
                "quality_score"
            ],

        "qmos":
            qmos,

        "quality_label":
            quality_label,

        "defects":
            defects,

        "statistics":
            prediction[
                "statistics"
            ],

        "recommendation":
            recommendation,

        "gradcam_url":
            gradcam_url,
    }


    # -----------------------------------------------------
    # 8. Save to MongoDB
    # -----------------------------------------------------

    document = (
        create_prediction_document(
            filename,
            result,
        )
    )

    document["_id"] = prediction_id

    predictions_collection.insert_one(
        document
    )


    return result