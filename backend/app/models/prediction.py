from datetime import datetime, timezone


def create_prediction_document(
    filename: str,
    prediction: dict,
):

    return {

        "filename":
            filename,

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