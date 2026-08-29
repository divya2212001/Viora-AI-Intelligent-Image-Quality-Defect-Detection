from pathlib import Path

from fastapi import (
    APIRouter,
    Query,
    HTTPException,
)

from app.config import settings
from app.database import (
    predictions_collection,
)


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


@router.get("/history")
def get_history(
    limit: int = 20,
    session_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
):

    limit = max(
        1,
        min(limit, 100)
    )

    cursor = (
        predictions_collection
        .find({"session_id": session_id})
        .sort(
            "created_at",
            -1,
        )
        .limit(limit)
    )

    results = []

    for document in cursor:
        document.pop(
            "_id",
            None,
        )
        results.append(
            document
        )
    return results

@router.get(
    "/analyses/{prediction_id}"
)
def get_analysis(
    prediction_id: str,
    session_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
):

    document = (
        predictions_collection
        .find_one(
            {
                "prediction_id":
                    prediction_id,
                "session_id": session_id,
            }
        )
    )

    if document is None:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    document.pop(
        "_id",
        None,
    )

    return document

@router.delete(
    "/analyses/{prediction_id}"
)
def delete_analysis(
    prediction_id: str,
    session_id: str = Query(
        ...,
        min_length=1,
        max_length=128,
    ),
):

    document = (
        predictions_collection
        .find_one(
            {
                "prediction_id":
                    prediction_id,
                "session_id": session_id,
            }
        )
    )

    if document is None:

        raise HTTPException(
            status_code=404,
            detail="Analysis not found.",
        )

    image_path = (
        settings.UPLOAD_DIR /
        f"{prediction_id}.jpg"
    )

    if image_path.exists():

        image_path.unlink()


    gradcam_url = (
        document.get("gradcam_url")
    )

    if gradcam_url:

        gradcam_path = (
            settings.UPLOAD_DIR /
            "gradcam" /
            Path(gradcam_url).name
        )

        if gradcam_path.exists():

            gradcam_path.unlink()

    predictions_collection.delete_one(
        {
            "prediction_id":
                prediction_id,
            "session_id": session_id,
        }
    )

    return {
        "message":
            "Analysis deleted successfully."
    }
