import json

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from scipy.stats import pearsonr, spearmanr

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)

from ml.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DEFECT_NAMES,
    FEATURE_NAMES,
    NUM_WORKERS,
    PROCESSED_DIR,
    REPORTS_DIR,
)

from dataset import KonIQDataset
from cnn_only import CNNOnlyQualityNet


EPOCHS = 20
LEARNING_RATE = 1e-4


def correlation(
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


def spearman(
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


def load_dataset(
    csv_file,
):

    feature_mean = np.load(
        ARTIFACTS_DIR
        / "feature_mean.npy"
    )

    feature_std = np.load(
        ARTIFACTS_DIR
        / "feature_std.npy"
    )

    return KonIQDataset(
        csv_file,
        feature_mean,
        feature_std,
        training=True,
    )


def evaluate(
    model,
    loader,
    device,
):

    model.eval()

    quality_true = []
    quality_pred = []

    total_loss = 0.0

    quality_loss = nn.MSELoss()

    with torch.no_grad():

        for (
            images,
            features,
            qualities,
            defects,
        ) in loader:

            images = images.to(
                device
            )

            qualities = qualities.to(
                device
            )

            defects = defects.to(
                device
            )

            q_pred, d_pred = model(
                images
            )

            loss_q = quality_loss(
                q_pred,
                qualities,
            )

            loss_d = quality_loss(
                d_pred,
                defects,
            )

            loss = (
                loss_q
                + 0.5 * loss_d
            )

            total_loss += (
                loss.item()
            )

            quality_true.extend(
                qualities.cpu().numpy()
            )

            quality_pred.extend(
                q_pred.cpu().numpy()
            )

    quality_true = np.asarray(
        quality_true
    )

    quality_pred = np.asarray(
        quality_pred
    )

    y_true_5 = (
        quality_true * 5.0
    )

    y_pred_5 = (
        quality_pred * 5.0
    )

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

    plcc = correlation(
        y_true_5,
        y_pred_5,
    )

    srcc = spearman(
        y_true_5,
        y_pred_5,
    )

    return (
        total_loss / len(loader),
        {
            "MAE_0_to_5": float(mae),
            "RMSE_0_to_5": float(rmse),
            "PLCC": float(plcc),
            "SRCC": float(srcc),
        },
    )


def main():

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(
        "=" * 70
    )

    print(
        "STEP 4 - CNN ONLY MODEL"
    )

    print(
        "=" * 70
    )

    print(
        f"\nDevice: {device}"
    )

    # ========================================================
    # DATA
    # ========================================================

    train_dataset = load_dataset(
        PROCESSED_DIR / "train.csv"
    )

    val_dataset = load_dataset(
        PROCESSED_DIR / "validation.csv"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # MODEL

    model = CNNOnlyQualityNet(
        num_defects=len(
            DEFECT_NAMES
        )
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4,
    )

    mse = nn.MSELoss()

    best_val_loss = float("inf")

    best_epoch = 0

    model_file = (
        ARTIFACTS_DIR
        / "cnn_only_model.pt"
    )


    # TRAIN

    for epoch in range(
        1,
        EPOCHS + 1,
    ):

        model.train()

        running_loss = 0.0

        for (
            images,
            features,
            qualities,
            defects,
        ) in train_loader:

            images = images.to(
                device
            )

            qualities = qualities.to(
                device
            )

            defects = defects.to(
                device
            )

            optimizer.zero_grad()

            quality_pred, defect_pred = (
                model(images)
            )

            quality_loss = mse(
                quality_pred,
                qualities,
            )

            defect_loss = mse(
                defect_pred,
                defects,
            )

            loss = (
                quality_loss
                + 0.5 * defect_loss
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
            )

        train_loss = (
            running_loss
            / len(train_loader)
        )

        val_loss, val_metrics = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train loss: {train_loss:.6f}"
        )

        print(
            f"Validation loss: "
            f"{val_loss:.6f}"
        )

        print(
            f"Validation PLCC: "
            f"{val_metrics['PLCC']:.4f}"
        )

        print(
            f"Validation SRCC: "
            f"{val_metrics['SRCC']:.4f}"
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch,

                    "validation_loss":
                        val_loss,
                },
                model_file,
            )

            print(
                "✓ Best CNN model saved"
            )

    # SAVE METADATA

    metadata = {

        "model":
            "CNN-only ResNet18",

        "architecture":
            "Pretrained ResNet18",

        "uses_cv_features":
            False,

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_val_loss,
    }

    with open(
        ARTIFACTS_DIR
        / "cnn_only_metadata.json",
        "w",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        "\n✓ CNN-only training complete"
    )

    print(
        f"✓ Saved: {model_file}"
    )


if __name__ == "__main__":
    main()