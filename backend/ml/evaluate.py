import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from scipy.stats import (
    pearsonr,
    spearmanr,
)

from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)

from torch.utils.data import DataLoader

from ml.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DEFECT_NAMES,
    FEATURE_NAMES,
    NUM_WORKERS,
    PROCESSED_DIR,
    REPORTS_DIR,
)

from ml.dataset import KonIQDataset
from ml.model import ImageQualityNet


# CORRELATION FUNCTIONS


def correlation(
    true,
    pred,
):

    if (
        len(true) < 2
        or np.std(true) == 0
        or np.std(pred) == 0
    ):

        return 0.0

    return float(
        pearsonr(
            true,
            pred,
        )[0]
    )


def spearman(
    true,
    pred,
):

    if (
        len(true) < 2
        or np.std(true) == 0
        or np.std(pred) == 0
    ):

        return 0.0

    return float(
        spearmanr(
            true,
            pred,
        )[0]
    )


# ============================================================
# PLOT: ACTUAL VS PREDICTED
# ============================================================

def save_actual_vs_predicted(
    true_quality,
    predicted_quality,
):

    output_file = (
        REPORTS_DIR
        / "quality_actual_vs_predicted.png"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.scatter(
        true_quality,
        predicted_quality,
        alpha=0.4,
        s=15,
    )

    minimum = min(
        true_quality.min(),
        predicted_quality.min(),
    )

    maximum = max(
        true_quality.max(),
        predicted_quality.max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
    )

    plt.xlabel(
        "Actual qMOS"
    )

    plt.ylabel(
        "Predicted qMOS"
    )

    plt.title(
        "Actual vs Predicted Image Quality"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
    )

    plt.close()

    return output_file


# ============================================================
# PLOT: ERROR DISTRIBUTION
# ============================================================

def save_error_distribution(
    errors,
):

    output_file = (
        REPORTS_DIR
        / "quality_error_distribution.png"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.hist(
        errors,
        bins=40,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.xlabel(
        "Prediction Error"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Quality Prediction Error Distribution"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        output_file,
        dpi=200,
    )

    plt.close()

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)

    print(
        "STEP 7 - FINAL TEST EVALUATION"
    )

    print("=" * 70)

    # DEVICE

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # LOAD FEATURE NORMALIZATION

    mean_file = (
        ARTIFACTS_DIR
        / "feature_mean.npy"
    )

    std_file = (
        ARTIFACTS_DIR
        / "feature_std.npy"
    )

    if not mean_file.exists():

        raise FileNotFoundError(
            f"Missing:\n{mean_file}"
        )

    if not std_file.exists():

        raise FileNotFoundError(
            f"Missing:\n{std_file}"
        )

    feature_mean = np.load(
        mean_file
    )

    feature_std = np.load(
        std_file
    )

    # Prevent division by zero.
    feature_std = np.maximum(
        feature_std,
        1e-8,
    )

    # LOAD TEST DATASET

    test_csv = (
        PROCESSED_DIR
        / "test.csv"
    )

    if not test_csv.exists():

        raise FileNotFoundError(
            f"Test CSV not found:\n{test_csv}"
        )

    test_dataset = KonIQDataset(
        test_csv,
        feature_mean,
        feature_std,
        training=False,
        synthetic_defects=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    defect_test_dataset = KonIQDataset(
        test_csv,
        feature_mean,
        feature_std,
        training=False,
        synthetic_defects=True,
    )
    defect_test_loader = DataLoader(
        defect_test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(
        f"Test samples: "
        f"{len(test_dataset)}"
    )

    # LOAD TRAINED MODEL

    model_file = Path(os.environ.get(
        "MODEL_CHECKPOINT_PATH",
        str(ARTIFACTS_DIR / "image_quality_model.pt"),
    ))

    if not model_file.exists():

        raise FileNotFoundError(
            f"Model not found:\n{model_file}"
        )

    checkpoint = torch.load(
        model_file,
        map_location=device,
    )

    model = ImageQualityNet(
        num_features=len(
            FEATURE_NAMES
        ),
        num_defects=len(
            DEFECT_NAMES
        ),
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "✓ Model loaded"
    )

    # PREDICTIONS

    true_quality = []

    predicted_quality = []

    true_defects = []

    predicted_defects = []

    filenames = []

    print(
        "\nGenerating test predictions..."
    )

    with torch.no_grad():

        for batch in test_loader:

            # ------------------------------------------------
            # Your current Dataset returns 4 values:
            #
            # images
            # features
            # qualities
            # defects
            #
            # ------------------------------------------------

            images, features, qualities, defects = batch

            images = images.to(
                device
            )

            features = features.to(
                device
            )

            quality_pred, defect_pred = (
                model(
                    images,
                    features,
                )
            )

            true_quality.extend(
                qualities.cpu().numpy()
            )

            predicted_quality.extend(
                quality_pred
                .cpu()
                .numpy()
            )

            true_defects.extend(
                defects.cpu().numpy()
            )

            predicted_defects.extend(
                defect_pred
                .cpu()
                .numpy()
            )

    # The qMOS metrics above use clean, untouched held-out images. Defect
    # metrics use separate, known degradations of those same held-out images.
    true_defects = []
    predicted_defects = []
    with torch.no_grad():
        for images, features, _, defects in defect_test_loader:
            _, defect_pred = model(images.to(device), features.to(device))
            true_defects.extend(defects.numpy())
            predicted_defects.extend(defect_pred.cpu().numpy())

    # CONVERT TO NUMPY

    true_quality = np.asarray(
        true_quality,
        dtype=np.float32,
    )

    predicted_quality = np.asarray(
        predicted_quality,
        dtype=np.float32,
    )

    true_defects = np.asarray(
        true_defects,
        dtype=np.float32,
    )

    predicted_defects = np.asarray(
        predicted_defects,
        dtype=np.float32,
    )

    # GET FILENAMES DIRECTLY FROM TEST CSV
    #
    # We don't depend on Dataset returning filenames.
    # This keeps compatibility with your current dataset.py.
    #

    test_df = pd.read_csv(
        test_csv
    )

    if len(test_df) != len(
        true_quality
    ):

        raise RuntimeError(
            "Number of predictions does not "
            "match number of test rows."
        )

    filenames = (
        test_df["filename"]
        .astype(str)
        .tolist()
    )

    # QUALITY: NORMALIZED 0-1 -> qMOS 0-5

    true_quality_5 = (
        true_quality * 5.0
    )

    predicted_quality_5 = (
        predicted_quality * 5.0
    )

    # QUALITY METRICS

    quality_errors = (
        predicted_quality_5
        - true_quality_5
    )

    quality_absolute_errors = (
        np.abs(
            quality_errors
        )
    )

    quality_mae = (
        mean_absolute_error(
            true_quality_5,
            predicted_quality_5,
        )
    )

    quality_rmse = np.sqrt(
        mean_squared_error(
            true_quality_5,
            predicted_quality_5,
        )
    )

    quality_pearson = correlation(
        true_quality_5,
        predicted_quality_5,
    )

    quality_spearman = spearman(
        true_quality_5,
        predicted_quality_5,
    )

    # DEFECT METRICS. These are binary metrics against the deterministic
    # synthetic degradations applied to held-out image identities.

    defect_results = {}

    for index, name in enumerate(
        DEFECT_NAMES
    ):

        y_true = (
            true_defects[:, index]
        )

        y_pred = (
            predicted_defects[:, index]
        )

        defect_results[name] = {

            "mae": float(
                mean_absolute_error(
                    y_true,
                    y_pred,
                )
            ),

            "rmse": float(
                np.sqrt(
                    mean_squared_error(
                        y_true,
                        y_pred,
                    )
                )
            ),

            "pearson": correlation(
                y_true,
                y_pred,
            ),

            "spearman": spearman(
                y_true,
                y_pred,
            ),
            "precision_at_0_5": float(precision_score(y_true, y_pred >= 0.5, zero_division=0)),
            "recall_at_0_5": float(recall_score(y_true, y_pred >= 0.5, zero_division=0)),
            "f1_at_0_5": float(f1_score(y_true, y_pred >= 0.5, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
            "average_precision": float(average_precision_score(y_true, y_pred)) if len(np.unique(y_true)) > 1 else None,
        }

    # CREATE TEST PREDICTION DATAFRAME

    prediction_data = {

        "filename":
            filenames,

        "actual_qmos":
            true_quality_5,

        "predicted_qmos":
            predicted_quality_5,

        "quality_error":
            quality_errors,

        "absolute_quality_error":
            quality_absolute_errors,
    }

    # Add defect predictions.

    for index, name in enumerate(
        DEFECT_NAMES
    ):

        prediction_data[
            f"actual_{name}"
        ] = true_defects[
            :, index
        ]

        prediction_data[
            f"predicted_{name}"
        ] = predicted_defects[
            :, index
        ]

    prediction_df = pd.DataFrame(
        prediction_data
    )

    # SAVE ALL TEST PREDICTIONS

    predictions_file = (
        REPORTS_DIR
        / "test_predictions.csv"
    )

    prediction_df.to_csv(
        predictions_file,
        index=False,
    )

    # WORST CASES

    worst_cases = (
        prediction_df
        .sort_values(
            "absolute_quality_error",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    worst_cases_file = (
        REPORTS_DIR
        / "worst_cases.csv"
    )

    worst_cases.to_csv(
        worst_cases_file,
        index=False,
    )

    # BEST CASES

    best_cases = (
        prediction_df
        .sort_values(
            "absolute_quality_error",
            ascending=True,
        )
        .head(20)
        .copy()
    )

    best_cases_file = (
        REPORTS_DIR
        / "best_cases.csv"
    )

    best_cases.to_csv(
        best_cases_file,
        index=False,
    )

    # OVER-PREDICTION CASES

    over_prediction = (
        prediction_df
        .sort_values(
            "quality_error",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    over_prediction.to_csv(
        REPORTS_DIR
        / "over_prediction_cases.csv",
        index=False,
    )

    # UNDER-PREDICTION CASES

    under_prediction = (
        prediction_df
        .sort_values(
            "quality_error",
            ascending=True,
        )
        .head(20)
        .copy()
    )

    under_prediction.to_csv(
        REPORTS_DIR
        / "under_prediction_cases.csv",
        index=False,
    )

    # PLOTS

    scatter_file = (
        save_actual_vs_predicted(
            true_quality_5,
            predicted_quality_5,
        )
    )

    error_file = (
        save_error_distribution(
            quality_errors
        )
    )

    # FINAL METRICS JSON

    results = {

        "test_samples":
            int(len(test_dataset)),

        "quality": {

            "MAE_0_to_5":
                float(quality_mae),

            "RMSE_0_to_5":
                float(quality_rmse),

            "PLCC":
                float(quality_pearson),

            "SRCC":
                float(quality_spearman),
        },

        "defects":
            defect_results,

        "model": {

            "architecture":
                "Pretrained ResNet18 + OpenCV features",

            "checkpoint":
                str(model_file),
        },
    }

    metrics_file = REPORTS_DIR / (
        "six_label_defect_evaluation.json"
        if model_file.name == "image_quality_model_candidate.pt"
        else "metrics.json"
    )

    with open(
        metrics_file,
        "w",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # PRINT RESULTS

    print(
        "\n" + "=" * 70
    )

    print(
        "FINAL TEST RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nTest samples: "
        f"{len(test_dataset)}"
    )

    print(
        "\nQUALITY"
    )

    print(
        f"MAE:     "
        f"{quality_mae:.4f}"
    )

    print(
        f"RMSE:    "
        f"{quality_rmse:.4f}"
    )

    print(
        f"PLCC:    "
        f"{quality_pearson:.4f}"
    )

    print(
        f"SRCC:    "
        f"{quality_spearman:.4f}"
    )

    print(
        "\nDEFECTS"
    )

    for name, values in (
        defect_results.items()
    ):

        print(
            f"\n{name}"
        )

        print(
            f"  MAE:     "
            f"{values['mae']:.4f}"
        )

        print(
            f"  RMSE:    "
            f"{values['rmse']:.4f}"
        )

        print(
            f"  Pearson: "
            f"{values['pearson']:.4f}"
        )

        print(
            f"  Spearman: "
            f"{values['spearman']:.4f}"
        )

    # FILES

    print(
        "\n" + "=" * 70
    )

    print(
        "FILES CREATED"
    )

    print(
        "=" * 70
    )

    print(
        f"\n✓ {metrics_file}"
    )

    print(
        f"✓ {predictions_file}"
    )

    print(
        f"✓ {worst_cases_file}"
    )

    print(
        f"✓ {best_cases_file}"
    )

    print(
        f"✓ {scatter_file}"
    )

    print(
        f"✓ {error_file}"
    )

    print(
        "\n✓ STEP 7 COMPLETE"
    )


if __name__ == "__main__":
    main()
