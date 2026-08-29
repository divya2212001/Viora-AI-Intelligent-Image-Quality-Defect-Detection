from fastapi import (
    APIRouter,
    File,
    UploadFile,
    HTTPException,
)

from app.services.prediction_service import (
    predict_image,
)
from app.config import settings


router = APIRouter(
    prefix="/api",
    tags=["Prediction"],
)


@router.post("/predict")
async def predict(
    file: UploadFile = File(...),
):

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail=(
                "File type could not be determined."
            ),
        )


    if file.content_type not in settings.ALLOWED_IMAGE_TYPES:

        raise HTTPException(
            status_code=400,
            detail=(
                "Please upload an image file."
            ),
        )


    image_bytes = await file.read()


    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail=(
                "Uploaded image is empty."
            ),
        )

    if len(image_bytes) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded image exceeds the size limit.")


    try:

        result = predict_image(
            image_bytes,
            file.filename or "image.jpg",
        )


        return result


    except HTTPException:

        raise


    except ValueError as exc:

        raise HTTPException(status_code=422, detail="The uploaded file is not a decodable image.") from exc

    except Exception as exc:

        print(
            "Prediction error:",
            repr(exc)
        )

        raise HTTPException(
            status_code=500,
            detail="Image analysis could not be completed. Please try another image.",
        )
