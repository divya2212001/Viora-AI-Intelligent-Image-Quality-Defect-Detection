import json
import random

import numpy as np
import torch

from torch.utils.data import (
    DataLoader,
)

from ml.config import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DEFECT_NAMES,
    FEATURE_NAMES,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    PROCESSED_DIR,
    RANDOM_SEED,
    WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
)

from ml.dataset import KonIQDataset

from ml.model import ImageQualityNet


CANDIDATE_MODEL_PATH = ARTIFACTS_DIR / "image_quality_model_candidate.pt"
CANDIDATE_METADATA_PATH = ARTIFACTS_DIR / "model_metadata_candidate.json"


def seed_everything():

    random.seed(
        RANDOM_SEED
    )

    np.random.seed(
        RANDOM_SEED
    )

    torch.manual_seed(
        RANDOM_SEED
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            RANDOM_SEED
        )


def calculate_feature_statistics(
    csv_path,
):

    import cv2

    from ml.config import IMAGE_DIR

    from ml.features import extract_features

    import pandas as pd

    df = pd.read_csv(
        csv_path
    )

    all_features = []

    print(
        "Calculating feature statistics..."
    )

    for index, row in df.iterrows():

        image_path = (
            IMAGE_DIR
            / row["filename"]
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            continue

        features = extract_features(
            image
        )

        all_features.append(
            features
        )

    all_features = np.asarray(
        all_features,
        dtype=np.float32,
    )

    mean = np.mean(
        all_features,
        axis=0,
    )

    std = np.std(
        all_features,
        axis=0,
    )

    std = np.maximum(
        std,
        1e-6,
    )

    return mean, std


def evaluate_loss(
    model,
    loader,
    quality_loss_fn,
    defect_loss_fn,
    device,
):

    model.eval()

    total_loss = 0.0

    total_quality_loss = 0.0

    total_defect_loss = 0.0

    total_samples = 0

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

            features = features.to(
                device
            )

            qualities = qualities.to(
                device
            )

            defects = defects.to(
                device
            )

            quality_pred, defect_pred = (
                model(
                    images,
                    features,
                )
            )

            quality_loss = (
                quality_loss_fn(
                    quality_pred,
                    qualities,
                )
            )

            defect_loss = (
                defect_loss_fn(
                    defect_pred,
                    defects,
                )
            )

            loss = (
                quality_loss
                + defect_loss
            )

            batch_size = (
                images.size(0)
            )

            total_loss += (
                loss.item()
                * batch_size
            )

            total_quality_loss += (
                quality_loss.item()
                * batch_size
            )

            total_defect_loss += (
                defect_loss.item()
                * batch_size
            )

            total_samples += (
                batch_size
            )

    return (
        total_loss
        / total_samples,

        total_quality_loss
        / total_samples,

        total_defect_loss
        / total_samples,
    )


def main():

    seed_everything()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(
        f"Device: {device}"
    )

    train_csv = (
        PROCESSED_DIR
        / "train.csv"
    )

    validation_csv = (
        PROCESSED_DIR
        / "validation.csv"
    )

    if not train_csv.exists():

        raise RuntimeError(
            "Run prepare_koniq.py first."
        )

    # FEATURE NORMALIZATION
    mean_path = ARTIFACTS_DIR / "feature_mean.npy"
    std_path = ARTIFACTS_DIR / "feature_std.npy"
    if mean_path.exists() and std_path.exists():
        # These artifacts were fitted on train.csv by the existing quality
        # training pipeline. Reusing them preserves its preprocessing exactly.
        feature_mean = np.load(mean_path)
        feature_std = np.load(std_path)
    else:
        feature_mean, feature_std = calculate_feature_statistics(train_csv)

    np.save(
        ARTIFACTS_DIR
        / "feature_mean.npy",
        feature_mean,
    )

    np.save(
        ARTIFACTS_DIR
        / "feature_std.npy",
        feature_std,
    )

    with open(
        ARTIFACTS_DIR
        / "feature_stats.json",
        "w",
    ) as file:

        json.dump(
            {
                "features": FEATURE_NAMES,
                "mean":
                    feature_mean.tolist(),
                "std":
                    feature_std.tolist(),
            },
            file,
            indent=2,
        )

    # DATASETS

    train_dataset = KonIQDataset(
        train_csv,
        feature_mean,
        feature_std,
        training=True,
        synthetic_defects=True,
    )

    validation_dataset = KonIQDataset(
        validation_csv,
        feature_mean,
        feature_std,
        training=False,
        synthetic_defects=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # MODEL

    model = ImageQualityNet(
        num_features=len(
            FEATURE_NAMES
        ),
        num_defects=len(
            DEFECT_NAMES
        ),
    ).to(device)

    # Preserve the previously validated qMOS regressor and retrain only the
    # expanded defect head.  The legacy checkpoint has five unrelated KonIQ++
    # annotation outputs, so its defect-head weights are intentionally skipped.
    legacy_path = ARTIFACTS_DIR / "image_quality_model.pt"
    legacy_checkpoint = torch.load(legacy_path, map_location=device)
    legacy_state = legacy_checkpoint["model_state_dict"]
    compatible_state = {
        name: value
        for name, value in legacy_state.items()
        if name in model.state_dict()
        and model.state_dict()[name].shape == value.shape
        and not name.startswith("defect_head")
    }
    model.load_state_dict(compatible_state, strict=False)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("defect_head")

    quality_loss_fn = torch.nn.SmoothL1Loss()
    defect_loss_fn = torch.nn.BCELoss()

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )
    )

    # TRAINING
    best_validation_loss = float(
        "inf"
    )

    best_epoch = 0

    patience_counter = 0

    for epoch in range(
        NUM_EPOCHS
    ):

        model.train()

        running_loss = 0.0

        running_samples = 0

        for (
            images,
            features,
            qualities,
            defects,
        ) in train_loader:

            images = images.to(
                device
            )

            features = features.to(
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
                model(
                    images,
                    features,
                )
            )

            quality_loss = (
                quality_loss_fn(
                    quality_pred,
                    qualities,
                )
            )

            defect_loss = (
                defect_loss_fn(
                    defect_pred,
                    defects,
                )
            )

            # Quality features and backbone are frozen, so this is a true
            # defect-head fine-tune and cannot alter held-out qMOS metrics.
            loss = defect_loss

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

            optimizer.step()

            batch_size = (
                images.size(0)
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            running_samples += (
                batch_size
            )

        train_loss = (
            running_loss
            / running_samples
        )

        (
            validation_loss,
            validation_quality_loss,
            validation_defect_loss,
        ) = evaluate_loss(
            model,
            validation_loader,
            quality_loss_fn,
            defect_loss_fn,
            device,
        )

        scheduler.step(
            validation_loss
        )

        print(
            f"\nEpoch "
            f"{epoch + 1}/{NUM_EPOCHS}"
        )

        print(
            f"Train loss: "
            f"{train_loss:.5f}"
        )

        print(
            f"Validation loss: "
            f"{validation_loss:.5f}"
        )

        print(
            f"Validation quality: "
            f"{validation_quality_loss:.5f}"
        )

        print(
            f"Validation defects: "
            f"{validation_defect_loss:.5f}"
        )

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            best_epoch = (
                epoch + 1
            )

            patience_counter = 0

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "feature_names":
                        FEATURE_NAMES,

                    "defect_names":
                        DEFECT_NAMES,

                    "defect_target_source": (
                        "deterministic synthetic degradations applied after "
                        "the train/validation/test image split"
                    ),

                    "image_size":
                        224,

                    "best_validation_loss":
                        best_validation_loss,

                    "best_epoch":
                        best_epoch,
                },
                CANDIDATE_MODEL_PATH,
            )

            print(
                "✓ Best candidate model saved"
            )

        else:

            patience_counter += 1

            print(
                f"No improvement "
                f"({patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE})"
            )

            if (
                patience_counter
                >= EARLY_STOPPING_PATIENCE
            ):

                print(
                    "Early stopping."
                )

                break

    # METADATA

    metadata = {

        "model_name":
            "KonIQ++ Hybrid Image Quality Model",

        "model_version":
            "1.1.0-six-label-candidate",

        "architecture":
            "Pretrained ResNet18 + OpenCV features",

        "quality_target":
            "qmos normalized to 0-1",

        "defect_targets":
            DEFECT_NAMES,

        "defect_target_source": (
            "Deterministic synthetic blur, underexposure, overexposure, "
            "noise, corruption, and visual-defect transformations. KonIQ++ "
            "does not natively label these six categories."
        ),

        "cv_features":
            FEATURE_NAMES,

        "best_epoch":
            best_epoch,

        "best_validation_loss":
            best_validation_loss,

        "random_seed":
            RANDOM_SEED,
    }

    with CANDIDATE_METADATA_PATH.open("w") as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    print(
        "\nTraining complete. Candidate checkpoint awaits held-out validation."
    )


if __name__ == "__main__":
    main()
