from typing import Any


ISSUE_NAMES = {
    "blur": "Blur",
    "underexposure": "Underexposure",
    "overexposure": "Overexposure",
    "noise": "Noise",
    "corruption": "Corruption",
    "defect": "Potential visual defect",
}


def get_severity(
    confidence: float,
) -> str:
    if confidence >= 0.85:
        return "high"

    if confidence >= 0.60:
        return "medium"

    return "low"


def predictions_to_issues(
    predictions: dict[str, float],
    threshold: float = 0.50,
) -> list[dict[str, Any]]:
    issues = []

    for issue_type, confidence in predictions.items():

        confidence = float(confidence)

        if confidence < threshold:
            continue

        issues.append(
            {
                "type": issue_type,
                "severity": get_severity(
                    confidence
                ),
                "confidence": round(
                    confidence,
                    4,
                ),
            }
        )

    issues.sort(
        key=lambda issue: issue["confidence"],
        reverse=True,
    )

    return issues


def calculate_quality_score(
    predictions: dict[str, float],
) -> float:
    """
    Convert issue probabilities into an overall
    quality score.

    The final weighting should be calibrated using
    validation data after model training.
    """

    weights = {
        "blur": 0.20,
        "underexposure": 0.15,
        "overexposure": 0.15,
        "noise": 0.15,
        "corruption": 0.20,
        "defect": 0.15,
    }

    weighted_penalty = 0.0

    for issue_type, weight in weights.items():
        probability = float(
            predictions.get(
                issue_type,
                0.0,
            )
        )

        weighted_penalty += (
            probability * weight
        )

    score = (
        100.0
        * (1.0 - weighted_penalty)
    )

    return round(
        max(
            0.0,
            min(100.0, score),
        ),
        2,
    )


def get_quality_label(
    score: float,
) -> str:
    if score >= 90:
        return "EXCELLENT"

    if score >= 75:
        return "ACCEPTABLE"

    if score >= 50:
        return "DEGRADED"

    return "DEFECTIVE"


def build_quality_result(
    predictions: dict[str, float],
) -> dict[str, Any]:

    issues = predictions_to_issues(
        predictions
    )

    score = calculate_quality_score(
        predictions
    )

    label = get_quality_label(
        score
    )

    return {
        "quality_score": score,
        "quality_label": label,
        "issues": issues,
    }