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

predictor = QualityPredictor()

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

def get_recommendation(
    qmos: float,
    defects: dict,
) -> str:

    if defects:

        highest_defect = max(
            defects,
            key=defects.get,
        )

        highest_score = float(
            defects[
                highest_defect
            ]
        )

    else:

        highest_defect = (
            "quality issue"
        )

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



def predict_image(
    image_bytes: bytes,
    filename: str,
):

    prediction = predictor.predict(
        image_bytes
    )

    qmos = prediction[
        "qmos"
    ]

    defects = prediction[
        "defects"
    ]
    quality_label = (
        get_quality_label(
            qmos
        )
    )

    recommendation = (
        get_recommendation(
            qmos,
            defects,
        )
    )

    prediction_id = str(
        uuid4()
    )

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


    success = cv2.imwrite(
        str(image_path),
        image,
    )


    if not success:

        raise RuntimeError(
            "Failed to save uploaded image."
        )


    image_url = (
        f"/uploads/{safe_filename}"
    )


    gradcam_url = None


    try:

        (
            image_tensor,
            feature_tensor,
        ) = predictor.prepare_inputs(
            image_bytes
        )

        gradcam_dir = (
            upload_dir
            / "gradcam"
        )


        gradcam_filename = (
            generate_gradcam(
                model=predictor.model,

                image_tensor=image_tensor,

                feature_tensor=feature_tensor,

                original_image=image,

                output_dir=gradcam_dir,

                output_filename=f"{prediction_id}.jpg",
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

        print(
            repr(error)
        )

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


    document = (
        create_prediction_document(
            filename,
            result,
        )
    )


    # Use prediction ID as MongoDB _id
    document["_id"] = (
        prediction_id
    )


    try:
        predictions_collection.insert_one(document)
        result["persistence_status"] = "stored"
    except Exception as exc:
        # An unavailable database must not discard an otherwise valid model
        # result or make Grad-CAM unusable. History becomes available again
        # once MongoDB is restored and future results are stored.
        print(f"WARNING: prediction was not persisted: {exc}")
        result["persistence_status"] = "unavailable"

    return result
