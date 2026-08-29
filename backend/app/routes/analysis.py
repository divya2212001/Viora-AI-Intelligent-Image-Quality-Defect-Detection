import os
import time
import uuid
from pathlib import Path

from bson import ObjectId
from fastapi import (
    APIRouter,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)
from pymongo import DESCENDING

from ..config import settings
from ..database import analyses_collection
from ..models.analysis import (
    create_analysis_document,
)
from ..schemas.analysis import (
    AnalysisHistoryResponse,
    AnalysisResponse,
    DeleteResponse,
)
from ..services.explainability_service import (
    ExplainabilityService,
)
from ..services.image_analyzer import (
    extract_features,
)
from ..services.image_validator import (
    validate_image_upload,
)
from ..services.model_service import (
    ModelService,
)
from ..services.quality_service import (
    build_quality_result,
)
from ..utils.helpers import (
    is_valid_object_id,
    serialize_analysis,
    serialize_history_item,
)


router = APIRouter(
    prefix="/api",
    tags=["Analysis"],
)


model_service = ModelService()

explainability_service = (
    ExplainabilityService()
)


@router.post(
    "/analyze",
    response_model=AnalysisResponse,
    status_code=status.HTTP_201_CREATED,
)
async def analyze_image(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Upload and analyze a single image.
    """

    start_time = time.perf_counter()

    # -----------------------------------------
    # 1. Validate upload
    # -----------------------------------------

    image_bytes = await validate_image_upload(
        file
    )

    # -----------------------------------------
    # 2. Extract CV features
    # -----------------------------------------

    try:
        statistics = extract_features(
            image_bytes
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                "Unable to analyze the image. "
                f"Processing error: {exc}"
            ),
        )

    # -----------------------------------------
    # 3. ML inference
    # -----------------------------------------

    try:
        predictions = model_service.predict(
            statistics
        )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Model inference failed."
            ),
        )

    # -----------------------------------------
    # 4. Quality decision
    # -----------------------------------------

    quality_result = (
        build_quality_result(
            predictions
        )
    )

    # -----------------------------------------
    # 5. Explainability
    # -----------------------------------------

    try:
        explainability = (
            explainability_service
            .generate_explanation(
                image_bytes,
                predictions,
            )
        )

    except Exception:
        explainability = {
            "heatmap": None,
        }

    # -----------------------------------------
    # 6. Save image
    # -----------------------------------------

    extension = (
        Path(
            file.filename
        ).suffix.lower()
    )

    if extension not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
        ".bmp",
    }:
        extension = ".jpg"

    stored_filename = (
        f"{uuid.uuid4().hex}"
        f"{extension}"
    )

    image_path = (
        settings.UPLOAD_DIR
        / stored_filename
    )

    try:
        with open(
            image_path,
            "wb",
        ) as output_file:
            output_file.write(
                image_bytes
            )

    except OSError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to save the uploaded image."
            ),
        )

    # -----------------------------------------
    # 7. Processing time
    # -----------------------------------------

    processing_time_ms = (
        time.perf_counter()
        - start_time
    ) * 1000

    processing_time_ms = round(
        processing_time_ms,
        2,
    )

    # -----------------------------------------
    # 8. MongoDB document
    # -----------------------------------------

    document = create_analysis_document(
        filename=file.filename,
        stored_filename=stored_filename,
        content_type=file.content_type
        or "application/octet-stream",
        file_size=len(image_bytes),

        quality_score=quality_result[
            "quality_score"
        ],

        quality_label=quality_result[
            "quality_label"
        ],

        issues=quality_result[
            "issues"
        ],

        statistics=statistics,

        model=(
            model_service
            .get_model_info()
        ),

        processing_time_ms=(
            processing_time_ms
        ),

        explainability=(
            explainability
        ),
    )

    try:
        insert_result = (
            analyses_collection.insert_one(
                document
            )
        )

        document["_id"] = (
            insert_result.inserted_id
        )

    except Exception:
        # Remove saved image if DB persistence fails.
        try:
            image_path.unlink(
                missing_ok=True
            )
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Analysis was completed, "
                "but the result could not be persisted."
            ),
        )

    # -----------------------------------------
    # 9. Serialize response
    # -----------------------------------------

    base_url = str(
        request.base_url
    ).rstrip("/")

    response = serialize_analysis(
        document,
        base_url,
    )

    return response


@router.get(
    "/analyses",
    response_model=AnalysisHistoryResponse,
)
def get_analysis_history(
    request: Request,
    limit: int = 20,
    skip: int = 0,
):
    """
    Retrieve previous analyses.
    """

    if limit < 1 or limit > 100:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "limit must be between 1 and 100."
            ),
        )

    if skip < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="skip cannot be negative.",
        )

    try:
        total = (
            analyses_collection.count_documents(
                {}
            )
        )

        cursor = (
            analyses_collection
            .find({})
            .sort(
                "created_at",
                DESCENDING,
            )
            .skip(skip)
            .limit(limit)
        )

        base_url = str(
            request.base_url
        ).rstrip("/")

        analyses = [
            serialize_history_item(
                document,
                base_url,
            )
            for document in cursor
        ]

        return {
            "analyses": analyses,
            "total": total,
        }

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to retrieve analysis history."
            ),
        )


@router.get(
    "/analyses/{analysis_id}",
    response_model=AnalysisResponse,
)
def get_analysis(
    analysis_id: str,
    request: Request,
):
    """
    Retrieve one previous analysis.
    """

    if not is_valid_object_id(
        analysis_id
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid analysis ID.",
        )

    try:
        document = (
            analyses_collection.find_one(
                {
                    "_id": ObjectId(
                        analysis_id
                    )
                }
            )
        )

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to retrieve analysis."
            ),
        )

    if document is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis not found.",
        )

    base_url = str(
        request.base_url
    ).rstrip("/")

    return serialize_analysis(
        document,
        base_url,
    )


@router.delete(
    "/analyses/{analysis_id}",
    response_model=DeleteResponse,
)
def delete_analysis(
    analysis_id: str,
):
    """
    Delete an analysis and its stored image.
    """

    if not is_valid_object_id(
        analysis_id
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid analysis ID.",
        )

    try:
        document = (
            analyses_collection.find_one(
                {
                    "_id": ObjectId(
                        analysis_id
                    )
                }
            )
        )

        if document is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Analysis not found.",
            )

        result = (
            analyses_collection.delete_one(
                {
                    "_id": ObjectId(
                        analysis_id
                    )
                }
            )
        )

        if result.deleted_count != 1:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=(
                    "Unable to delete analysis."
                ),
            )

        stored_filename = document.get(
            "stored_filename"
        )

        if stored_filename:
            image_path = (
                settings.UPLOAD_DIR
                / stored_filename
            )

            try:
                image_path.unlink(
                    missing_ok=True
                )
            except OSError:
                pass

        return {
            "message": (
                "Analysis deleted successfully."
            )
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "Unable to delete analysis."
            ),
        )