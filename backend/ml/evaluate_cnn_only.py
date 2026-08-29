import json

import numpy as np
import torch

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from torch.utils.data import DataLoader

from ml.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DEFECT_NAMES,
    NUM_WORKERS,
    PROCESSED_DIR,
    REPORTS_DIR,
)

from dataset import KonIQDataset
from cnn_only import CNNOnlyQualityNet


def corr(
    y_true,
    y_pred,
):

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


def spear(
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


def main():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    mean = np.load(
        ARTIFACTS_DIR
        / "feature_mean.npy"
    )

    std = np.load(
        ARTIFACTS_DIR
        / "feature_std.npy"
    )

    dataset = KonIQDataset(
        PROCESSED_DIR / "test.csv",
        mean,
        std,
        training=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    checkpoint = torch.load(
        ARTIFACTS_DIR
        / "cnn_only_model.pt",
        map_location=device,
    )

    # The archived CNN-only baseline remains a five-output KonIQ++ model.
    # It is evaluated only for qMOS comparison and is not used by the API.
    model = CNNOnlyQualityNet(
        num_defects=len(checkpoint.get("defect_names", DEFECT_NAMES))
    ).to(device)

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for (
            images,
            features,
            quality,
            defects,
        ) in loader:

            images = images.to(
                device
            )

            prediction, _ = model(
                images
            )

            y_true.extend(
                quality.numpy()
            )

            y_pred.extend(
                prediction.cpu().numpy()
            )

    y_true = np.asarray(
        y_true
    )

    y_pred = np.asarray(
        y_pred
    )

    y_true_5 = y_true * 5.0
    y_pred_5 = y_pred * 5.0

    mae = mean_absolute_error(
        y_true_5,
        y_pred_5,
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true_5,
            y_pred_5,
        )
    )

    plcc = corr(
        y_true_5,
        y_pred_5,
    )

    srcc = spear(
        y_true_5,
        y_pred_5,
    )

    results = {

        "model":
            "CNN-only ResNet18",

        "test_samples":
            len(dataset),

        "MAE_0_to_5":
            float(mae),

        "RMSE_0_to_5":
            float(rmse),

        "PLCC":
            float(plcc),

        "SRCC":
            float(srcc),
    }

    output = (
        REPORTS_DIR
        / "cnn_only_metrics.json"
    )

    with open(
        output,
        "w",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print(
        "\nCNN-ONLY TEST RESULTS"
    )

    print(
        f"MAE:  {mae:.4f}"
    )

    print(
        f"RMSE: {rmse:.4f}"
    )

    print(
        f"PLCC: {plcc:.4f}"
    )

    print(
        f"SRCC: {srcc:.4f}"
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()
