from pathlib import Path

from fastapi import (
    APIRouter,
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
):

    limit = max(
        1,
        min(limit, 100)
    )

    cursor = (
        predictions_collection
        .find({})
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
):

    document = (
        predictions_collection
        .find_one(
            {
                "prediction_id":
                    prediction_id
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
):

    document = (
        predictions_collection
        .find_one(
            {
                "prediction_id":
                    prediction_id
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
                prediction_id
        }
    )

    return {
        "message":
            "Analysis deleted successfully."
    }