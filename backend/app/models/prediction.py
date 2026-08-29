from datetime import datetime, timezone


def create_prediction_document(
    prediction_id: str,
    filename: str,
    session_id: str,
    prediction: dict,
):
    """
    Create the MongoDB document for an image analysis.
    """

    return {

        "prediction_id":
            prediction_id,

        "session_id":
            session_id,

        "filename":
            filename,

        "image_url":
            prediction.get(
                "image_url"
            ),

        "gradcam_url":
            prediction.get(
                "gradcam_url"
            ),

        "quality_score":
            prediction[
                "quality_score"
            ],

        "qmos":
            prediction[
                "qmos"
            ],

        "quality_label":
            prediction[
                "quality_label"
            ],

        "defects":
            prediction[
                "defects"
            ],

        "statistics":
            prediction[
                "statistics"
            ],

        "recommendation":
            prediction[
                "recommendation"
            ],

        "created_at":
            datetime.now(
                timezone.utc
            ),
    }