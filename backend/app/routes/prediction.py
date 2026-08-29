from fastapi import (
    APIRouter,
    File,
    HTTPException,
    UploadFile,
)

from ..database import (
    predictions_collection,
)

from ..schemas.prediction import (
    PredictionResponse,
)

from ..services.prediction_service import (
    get_model_information,
    predict_image,
)


router = APIRouter(
    prefix="/api",
    tags=["Prediction"],
)



# CONFIGURATION
ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}

MAX_FILE_SIZE = (
    10 * 1024 * 1024
)



# PREDICT
@router.post(
    "/predict",
    response_model=PredictionResponse,
)
async def predict(
    file: UploadFile = File(...),
):


    # Validate content type
    if (
        file.content_type
        not in ALLOWED_CONTENT_TYPES
    ):

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported image format. "
                "Please upload JPEG, PNG, or WebP."
            ),
        )


    # Read image
    image_bytes = await file.read()


    # Empty file
    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty.",
        )


    # File size
    if len(image_bytes) > MAX_FILE_SIZE:

        raise HTTPException(
            status_code=413,
            detail=(
                "Image is too large. "
                "Maximum allowed size is 10 MB."
            ),
        )


    # Prediction
    try:

        result = predict_image(
            image_bytes=image_bytes,
            filename=(
                file.filename
                or "uploaded_image"
            ),
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        print(
            "Prediction error:",
            repr(error),
        )

        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to process the image."
            ),
        )



# HISTORY
@router.get(
    "/history",
)
def get_prediction_history(
    limit: int = 20,
):

    limit = max(
        1,
        min(
            limit,
            100,
        ),
    )

    predictions = list(
        predictions_collection
        .find(
            {},
            {
                "_id": 0,
            },
        )
        .sort(
            "created_at",
            -1,
        )
        .limit(limit)
    )

    return {
        "count":
            len(predictions),

        "predictions":
            predictions,
    }



# MODEL INFORMATION
@router.get(
    "/model-info",
)
def model_info():

    return get_model_information()