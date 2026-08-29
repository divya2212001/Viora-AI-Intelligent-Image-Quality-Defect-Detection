from typing import Any
from uuid import uuid4

from ..database import (
    predictions_collection,
)

from ..models.prediction import (
    create_prediction_document,
)

from .model_service import ModelService



# MODEL SERVICE
model_service = ModelService()



# QUALITY LABEL
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


# RECOMMENDATION
def get_recommendation(
    qmos: float,
    defects: dict[str, float],
) -> str:

    if not defects:

        return (
            "Image quality has been evaluated."
        )

    highest_defect = max(
        defects,
        key=defects.get,
    )

    highest_score = defects[
        highest_defect
    ]

    if qmos >= 4.0:

        return (
            "Image quality is excellent "
            "with minimal detected quality issues."
        )

    if qmos >= 3.5:

        return (
            "Image quality is good. "
            "Minor quality issues may be present."
        )

    if qmos >= 2.5:

        return (
            "Image quality is moderate. "
            f"The highest detected issue score "
            f"is associated with {highest_defect}."
        )

    return (
        "Image quality is low. "
        f"The highest detected issue score "
        f"is associated with {highest_defect} "
        f"({highest_score:.2f})."
    )



# PREDICTION

def predict_image(
    image_bytes: bytes,
    filename: str,
) -> dict[str, Any]:


    # Run trained Hybrid CNN + CV model
    prediction = model_service.predict(
        image_bytes
    )

    qmos = float(
        prediction["qmos"]
    )

    quality_score = float(
        prediction["quality_score"]
    )

    defects = {
        name: float(value)
        for name, value in prediction[
            "defects"
        ].items()
    }

    statistics = {
        name: float(value)
        for name, value in prediction[
            "statistics"
        ].items()
    }


    # Business-level interpretation
    quality_label = (
        get_quality_label(qmos)
    )

    recommendation = (
        get_recommendation(
            qmos,
            defects,
        )
    )


    # Generate prediction ID
    prediction_id = str(
        uuid4()
    )


    # API result

    result = {

        "prediction_id":
            prediction_id,

        "filename":
            filename,

        "quality_score":
            round(
                quality_score,
                2,
            ),

        "qmos":
            round(
                qmos,
                4,
            ),

        "quality_label":
            quality_label,

        "defects":
            defects,

        "statistics":
            statistics,

        "recommendation":
            recommendation,
    }

    # Save prediction to MongoDB

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



# MODEL INFORMATION


def get_model_information():

    return (
        model_service.get_model_info()
    )