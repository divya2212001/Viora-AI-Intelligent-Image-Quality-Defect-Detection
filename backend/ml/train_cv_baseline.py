import json

import joblib
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader

from ml.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    FEATURE_NAMES,
    NUM_WORKERS,
    PROCESSED_DIR,
    REPORTS_DIR,
)

from dataset import KonIQDataset


def correlation(y_true, y_pred):

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return 0.0

    return float(
        pearsonr(
            y_true,
            y_pred,
        )[0]
    )


def rank_correlation(
    y_true,
    y_pred,
):

    if (
        np.std(y_true) == 0
        or np.std(y_pred) == 0
    ):
        return 0.0

    return float(
        spearmanr(
            y_true,
            y_pred,
        )[0]
    )


def collect_features(csv_file):

    feature_mean = np.load(
        ARTIFACTS_DIR
        / "feature_mean.npy"
    )

    feature_std = np.load(
        ARTIFACTS_DIR
        / "feature_std.npy"
    )

    dataset = KonIQDataset(
        csv_file,
        feature_mean,
        feature_std,
        training=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    features = []
    qualities = []

    for (
        images,
        cv_features,
        quality,
        defects,
    ) in loader:

        features.append(
            cv_features.numpy()
        )

        qualities.append(
            quality.numpy()
        )

    features = np.concatenate(
        features,
        axis=0,
    )

    qualities = np.concatenate(
        qualities,
        axis=0,
    )

    return features, qualities


def evaluate(
    model,
    x,
    y,
):

    prediction = model.predict(x)

    # Dataset quality is normalized 0-1.
    y5 = y * 5.0
    prediction5 = prediction * 5.0

    mae = mean_absolute_error(
        y5,
        prediction5,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y5,
            prediction5,
        )
    )

    plcc = correlation(
        y5,
        prediction5,
    )

    srcc = rank_correlation(
        y5,
        prediction5,
    )

    return {
        "MAE_0_to_5": float(mae),
        "RMSE_0_to_5": float(rmse),
        "PLCC": float(plcc),
        "SRCC": float(srcc),
    }


def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "CV BASELINE"
    )


    train_x, train_y = collect_features(
        PROCESSED_DIR / "train.csv"
    )

    val_x, val_y = collect_features(
        PROCESSED_DIR / "validation.csv"
    )

    test_x, test_y = collect_features(
        PROCESSED_DIR / "test.csv"
    )

    print(
        f"\nTrain: {len(train_x)}"
    )

    print(
        f"Validation: {len(val_x)}"
    )

    print(
        f"Test: {len(test_x)}"
    )

    print(
        f"Features: {train_x.shape[1]}"
    )

    # RANDOM FOREST

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )

    print(
        "\nTraining Random Forest..."
    )

    model.fit(
        train_x,
        train_y,
    )

    # VALIDATION

    validation_metrics = evaluate(
        model,
        val_x,
        val_y,
    )

    print(
        "\nValidation:"
    )

    for key, value in (
        validation_metrics.items()
    ):
        print(
            f"{key}: {value:.4f}"
        )

    # TEST

    test_metrics = evaluate(
        model,
        test_x,
        test_y,
    )

    print(
        "\nTEST:"
    )

    for key, value in (
        test_metrics.items()
    ):
        print(
            f"{key}: {value:.4f}"
        )

    # SAVE MODEL
    model_file = (
        ARTIFACTS_DIR
        / "cv_baseline_model.joblib"
    )

    joblib.dump(
        model,
        model_file,
    )

    # SAVE METRICS

    results = {
        "model": "Random Forest CV Baseline",
        "features": FEATURE_NAMES,
        "validation": validation_metrics,
        "test": test_metrics,
    }

    metrics_file = (
        REPORTS_DIR
        / "cv_baseline_metrics.json"
    )

    with open(
        metrics_file,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        f"\n✓ Model saved: {model_file}"
    )

    print(
        f"✓ Metrics saved: {metrics_file}"
    )


if __name__ == "__main__":
    main()