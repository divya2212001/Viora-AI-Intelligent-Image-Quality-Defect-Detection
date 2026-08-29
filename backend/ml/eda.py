import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ml.config import (
    PROCESSED_DIR,
    REPORTS_DIR,
)


def describe_csv(path, name):
    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    df = pd.read_csv(path)

    print(f"Rows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum())

    return df


def find_column(df, candidates):
    for column in candidates:
        if column in df.columns:
            return column

    return None


def main():

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_file = PROCESSED_DIR / "train.csv"
    val_file = PROCESSED_DIR / "validation.csv"
    test_file = PROCESSED_DIR / "test.csv"

    train = describe_csv(
        train_file,
        "TRAIN DATA",
    )

    val = describe_csv(
        val_file,
        "VALIDATION DATA",
    )

    test = describe_csv(
        test_file,
        "TEST DATA",
    )

    # ========================================================
    # DATASET SPLIT SUMMARY
    # ========================================================

    split_summary = {
        "train": len(train),
        "validation": len(val),
        "test": len(test),
        "total": len(train) + len(val) + len(test),
    }

    print("\nDataset split:")
    print(json.dumps(
        split_summary,
        indent=2,
    ))

    # ========================================================
    # FIND QUALITY COLUMN
    # ========================================================

    quality_candidates = [
        "qmos",
        "mos",
        "quality",
        "score",
        "mean_opinion_score",
    ]

    quality_column = find_column(
        train,
        quality_candidates,
    )

    if quality_column is None:

        print(
            "\nWARNING: Could not automatically "
            "find quality column."
        )

    else:

        print(
            f"\nQuality column: {quality_column}"
        )

        print(
            train[quality_column].describe()
        )

        # Quality distribution

        plt.figure(
            figsize=(8, 6)
        )

        plt.hist(
            train[quality_column],
            bins=40,
        )

        plt.xlabel(
            quality_column
        )

        plt.ylabel(
            "Number of images"
        )

        plt.title(
            "Training Quality Distribution"
        )

        plt.grid(
            alpha=0.25
        )

        plt.tight_layout()

        plt.savefig(
            REPORTS_DIR
            / "eda_quality_distribution.png",
            dpi=200,
        )

        plt.close()

    # ========================================================
    # NUMERIC FEATURE CORRELATIONS
    # ========================================================

    numeric_columns = (
        train
        .select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    if quality_column in numeric_columns:

        correlations = (
            train[
                numeric_columns
            ]
            .corr()[quality_column]
            .sort_values(
                ascending=False
            )
        )

        print(
            "\nFeature correlations with quality:"
        )

        print(correlations)

        correlations.to_csv(
            REPORTS_DIR
            / "eda_quality_correlations.csv"
        )

    # ========================================================
    # SAVE EDA SUMMARY
    # ========================================================

    summary = {
        "train_samples": len(train),
        "validation_samples": len(val),
        "test_samples": len(test),
        "total_samples": (
            len(train)
            + len(val)
            + len(test)
        ),
        "train_columns": train.columns.tolist(),
        "missing_values": {
            column: int(value)
            for column, value
            in train.isnull().sum().items()
        },
    }

    with open(
        REPORTS_DIR
        / "eda_summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        "\n✓ EDA complete"
    )


if __name__ == "__main__":
    main()