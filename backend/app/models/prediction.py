from datetime import datetime, timezone


def create_prediction_document(
    filename: str,
    prediction: dict,
):

    return {
        "prediction_id":
            prediction[
                "prediction_id"
            ],

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