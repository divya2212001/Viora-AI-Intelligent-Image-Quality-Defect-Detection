from typing import Dict

from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):

    prediction_id: str

    filename: str

    quality_score: float = Field(
        description=(
            "Image quality score from 0 to 100."
        )
    )

    qmos: float = Field(
        description=(
            "Predicted quality score from 0 to 5."
        )
    )

    quality_label: str

    defects: Dict[str, float]

    statistics: Dict[str, float]

    recommendation: str