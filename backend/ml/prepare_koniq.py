from pathlib import Path

import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent

RAW_DIR = BASE_DIR / "data" / "raw" / "koniq"

IMAGE_DIR = RAW_DIR / "images"

CSV_FILE = RAW_DIR / "koniqplusplus.csv"

PROCESSED_DIR = (
    BASE_DIR / "data" / "processed"
)

RANDOM_SEED = 42


def main():
    print("KonIQ++ DATASET PREPARATION")
    if not CSV_FILE.exists():

        raise FileNotFoundError(
            f"CSV not found:\n{CSV_FILE}"
        )

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found:\n{IMAGE_DIR}"
        )

    df = pd.read_csv(
        CSV_FILE
    )

    print(
        f"\nCSV rows: {len(df)}"
    )

    print(
        f"CSV columns: {len(df.columns)}"
    )


    required_columns = [
        "filename",
        "qmos",
        "sd",
        "votes",
        "artifacts",
        "blur",
        "contrast",
        "colors",
        "other",
        "degraded.amount",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns:\n"
            + "\n".join(missing_columns)
        )

    print(
        "✓ Required columns found"
    )

    duplicates = (
        df["filename"]
        .duplicated()
        .sum()
    )

    print(
        f"Duplicate filenames: {duplicates}"
    )

    if duplicates:

        raise ValueError(
            "Duplicate filenames found."
        )

    numeric_columns = [
        "qmos",
        "sd",
        "votes",
        "artifacts",
        "blur",
        "contrast",
        "colors",
        "other",
        "degraded.amount",
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    null_counts = (
        df[numeric_columns]
        .isna()
        .sum()
    )

    if null_counts.any():

        print(
            "\nMissing numeric values:"
        )

        print(
            null_counts[
                null_counts > 0
            ]
        )

        raise ValueError(
            "Missing numeric values found."
        )

    print(
        "✓ Numeric values valid"
    )

    defect_columns = [
        "artifacts",
        "blur",
        "contrast",
        "colors",
        "other",
    ]

    for column in defect_columns:

        minimum = df[column].min()

        maximum = df[column].max()

        print(
            f"{column:12s}: "
            f"{minimum:.4f} → "
            f"{maximum:.4f}"
        )

        if minimum < 0 or maximum > 1:

            raise ValueError(
                f"{column} is outside [0, 1]"
            )

    print(
        "✓ Defect frequencies are in [0, 1]"
    )

    print("\nQuality score statistics:")

    print(
        df["qmos"].describe()
    )

    print(
        "\nScanning image directory..."
    )

    image_files = [
        path
        for path in IMAGE_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in {
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        }
    ]

    image_names = {
        path.name
        for path in image_files
    }

    csv_names = set(
        df["filename"]
    )

    missing_images = (
        csv_names - image_names
    )

    extra_images = (
        image_names - csv_names
    )

    print(
        f"Images found: {len(image_names)}"
    )

    print(
        f"Missing images: "
        f"{len(missing_images)}"
    )

    print(
        f"Extra images: "
        f"{len(extra_images)}"
    )

    if missing_images:

        print(
            "\nFirst missing images:"
        )

        for filename in sorted(
            missing_images
        )[:20]:

            print(filename)

        raise ValueError(
            "Some CSV images are missing."
        )

    print(
        "✓ Every CSV image exists"
    )


    df = df.sample(
        frac=1.0,
        random_state=RANDOM_SEED,
    ).reset_index(
        drop=True
    )

    n = len(df)

    train_end = int(
        n * 0.70
    )

    validation_end = int(
        n * 0.85
    )

    train_df = df.iloc[
        :train_end
    ].copy()

    validation_df = df.iloc[
        train_end:validation_end
    ].copy()

    test_df = df.iloc[
        validation_end:
    ].copy()

    print("\nDataset split:")

    print(
        f"Training:   {len(train_df)}"
    )

    print(
        f"Validation: {len(validation_df)}"
    )

    print(
        f"Test:       {len(test_df)}"
    )

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False,
    )

    validation_df.to_csv(
        PROCESSED_DIR / "validation.csv",
        index=False,
    )

    test_df.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False,
    )

    print(
        "\n✓ train.csv created"
    )

    print(
        "✓ validation.csv created"
    )

    print(
        "✓ test.csv created"
    )

    print(
        "DATASET PREPARATION COMPLETE"
    )

if __name__ == "__main__":
    main()