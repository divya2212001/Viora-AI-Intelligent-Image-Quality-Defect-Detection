from typing import Any

from bson import ObjectId


def serialize_analysis(
    document: dict[str, Any],
    base_url: str = "",
) -> dict[str, Any]:
    """
    Convert MongoDB document into API-friendly JSON.
    """

    analysis_id = str(
        document["_id"]
    )

    stored_filename = document.get(
        "stored_filename"
    )

    image_url = None

    if stored_filename:
        image_url = (
            f"{base_url}/uploads/"
            f"{stored_filename}"
        )

    return {
        "id": analysis_id,

        "filename": document.get(
            "filename",
            "unknown",
        ),

        "image_url": image_url,

        "quality_score": document.get(
            "quality_score",
            0,
        ),

        "quality_label": document.get(
            "quality_label",
            "UNKNOWN",
        ),

        "issues": document.get(
            "issues",
            [],
        ),

        "statistics": document.get(
            "statistics",
            {},
        ),

        "model": document.get(
            "model",
            {},
        ),

        "processing_time_ms": document.get(
            "processing_time_ms",
            0,
        ),

        "explainability": document.get(
            "explainability",
            {},
        ),

        "created_at": document.get(
            "created_at"
        ),
    }


def serialize_history_item(
    document: dict[str, Any],
    base_url: str = "",
) -> dict[str, Any]:

    stored_filename = document.get(
        "stored_filename"
    )

    image_url = None

    if stored_filename:
        image_url = (
            f"{base_url}/uploads/"
            f"{stored_filename}"
        )

    return {
        "id": str(
            document["_id"]
        ),

        "filename": document.get(
            "filename",
            "unknown",
        ),

        "image_url": image_url,

        "quality_score": document.get(
            "quality_score",
            0,
        ),

        "quality_label": document.get(
            "quality_label",
            "UNKNOWN",
        ),

        "created_at": document.get(
            "created_at"
        ),
    }


def is_valid_object_id(
    value: str,
) -> bool:
    return ObjectId.is_valid(value)