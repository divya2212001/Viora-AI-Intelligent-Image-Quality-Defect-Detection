"""Promote a validated six-label candidate checkpoint to the active model."""

import json
import shutil

import torch

from ml.config import ARTIFACTS_DIR, DEFECT_NAMES
from ml.model import ImageQualityNet


def main():
    candidate = ARTIFACTS_DIR / "image_quality_model_candidate.pt"
    candidate_metadata = ARTIFACTS_DIR / "model_metadata_candidate.json"
    validation_report = ARTIFACTS_DIR.parent / "reports" / "six_label_defect_evaluation.json"
    if not candidate.exists() or not candidate_metadata.exists() or not validation_report.exists():
        raise RuntimeError("Candidate checkpoint, candidate metadata, and held-out defect report are all required before promotion.")

    checkpoint = torch.load(candidate, map_location="cpu")
    if checkpoint.get("defect_names") != DEFECT_NAMES:
        raise RuntimeError("Candidate checkpoint does not declare the exact required six-label order.")
    model = ImageQualityNet(num_defects=len(DEFECT_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with candidate_metadata.open() as file:
        metadata = json.load(file)
    if metadata.get("defect_targets") != DEFECT_NAMES:
        raise RuntimeError("Candidate metadata does not declare the exact required six labels.")
    shutil.copy2(candidate, ARTIFACTS_DIR / "image_quality_model.pt")
    shutil.copy2(candidate_metadata, ARTIFACTS_DIR / "model_metadata.json")
    print("Promoted validated six-label candidate checkpoint.")


if __name__ == "__main__":
    main()
