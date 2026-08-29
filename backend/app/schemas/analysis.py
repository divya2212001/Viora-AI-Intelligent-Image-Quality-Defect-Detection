from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Issue(BaseModel):
    type: str

    severity: str

    confidence: float = Field(
        ge=0.0,
        le=1.0,
    )


class ImageStatistics(BaseModel):
    brightness: float | None = None
    contrast: float | None = None
    sharpness: float | None = None
    noise_level: float | None = None
    entropy: float | None = None
    saturation: float | None = None

    dark_pixel_ratio: float | None = None
    bright_pixel_ratio: float | None = None


class ModelInfo(BaseModel):
    name: str
    version: str


class Explainability(BaseModel):
    heatmap: str | None = None


class AnalysisResponse(BaseModel):
    id: str

    filename: str

    image_url: str | None = None

    quality_score: float

    quality_label: str

    issues: list[Issue]

    statistics: ImageStatistics

    model: ModelInfo

    processing_time_ms: float

    explainability: Explainability | None = None

    created_at: datetime


class AnalysisHistoryItem(BaseModel):
    id: str

    filename: str

    image_url: str | None = None

    quality_score: float

    quality_label: str

    created_at: datetime


class AnalysisHistoryResponse(BaseModel):
    analyses: list[AnalysisHistoryItem]
    total: int


class DeleteResponse(BaseModel):
    message: str


class HealthResponse(BaseModel):
    status: str
    database: str
    version: str