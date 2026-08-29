from fastapi import APIRouter

from app.database import (
    check_database_connection,
)
from app.services.prediction_service import predictor


router = APIRouter(
    tags=["Health"]
)

@router.get(
    "/health"
)
def health():

    database_status = (
        check_database_connection()
    )

    return {

        "status":
            "ok",

        "database":
            "connected"
            if database_status
            else "disconnected",

        "model": "loaded" if predictor is not None else "unavailable",
    }
