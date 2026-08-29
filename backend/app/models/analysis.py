from datetime import datetime, timezone
from typing import Any


def create_analysis_document(
    *,
    filename: str,
    stored_filename: str,
    content_type: str,
    file_size: int,
    quality_score: float,
    quality_label: str,
    issues: list[dict[str, Any]],
    statistics: dict[str, Any],
    model: dict[str, Any],
    processing_time_ms: float,
    explainability: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Create a MongoDB analysis document.
    """

    return {
        "filename": filename,
        "stored_filename": stored_filename,
        "content_type": content_type,
        "file_size": file_size,

        "quality_score": quality_score,
        "quality_label": quality_label,

        "issues": issues,
        "statistics": statistics,

        "model": model,

        "processing_time_ms": processing_time_ms,

        "explainability": explainability or {},

        "created_at": datetime.now(timezone.utc),
    }