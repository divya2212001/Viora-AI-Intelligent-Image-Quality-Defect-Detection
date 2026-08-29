import json

from fastapi import APIRouter

from ml.config import ARTIFACTS_DIR


router = APIRouter(prefix="/api", tags=["Model"])


@router.get("/model-info")
def get_model_info():
    """Expose the metadata saved with the checkpoint, never placeholder values."""
    metadata_path = ARTIFACTS_DIR / "model_metadata.json"
    if not metadata_path.exists():
        return {"name": "Unknown", "version": "Unknown", "architecture": "Unavailable"}
    with metadata_path.open() as metadata_file:
        metadata = json.load(metadata_file)
    return {
        "name": metadata.get("model_name", "Unknown"),
        "version": metadata.get("model_version", "Unknown"),
        "architecture": metadata.get("architecture", "Unknown"),
        "defect_targets": metadata.get("defect_targets", []),
        "quality_target": metadata.get("quality_target"),
    }
