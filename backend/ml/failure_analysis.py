from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import cv2

from ml.config import (
    IMAGE_DIR,
    REPORTS_DIR,
)


def create_contact_sheet(
    df,
    output_path,
    title,
    columns=4,
):
    """
    Create a visual contact sheet for manual
    failure-case inspection.
    """

    rows = (
        len(df) + columns - 1
    ) // columns

    plt.figure(
        figsize=(
            columns * 4,
            rows * 4,
        )
    )

    for i, (_, row) in enumerate(
        df.iterrows()
    ):

        image_path = (
            IMAGE_DIR
            / str(row["filename"])
        )

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"WARNING: Cannot read "
                f"{image_path}"
            )

            continue

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        ax = plt.subplot(
            rows,
            columns,
            i + 1,
        )

        ax.imshow(image)

        ax.axis("off")

        actual = row[
            "actual_qmos"
        ]

        predicted = row[
            "predicted_qmos"
        ]

        error = row[
            "absolute_error"
        ]

        ax.set_title(
            f"Actual: {actual:.2f}\n"
            f"Pred: {predicted:.2f}\n"
            f"Error: {error:.2f}",
            fontsize=9,
        )

    plt.suptitle(
        title,
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


def main():

    print("=" * 70)

    print(
        "STEP 8 - FAILURE CASE ANALYSIS"
    )

    print("=" * 70)

    prediction_file = (
        REPORTS_DIR
        / "test_predictions.csv"
    )

    if not prediction_file.exists():

        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{prediction_file}\n\n"
            "Run evaluate.py first."
        )

    df = pd.read_csv(
        prediction_file
    )

    # ========================================================
    # Calculate error
    # ========================================================

    df["error"] = (
        df["predicted_qmos"]
        - df["actual_qmos"]
    )

    df["absolute_error"] = (
        df["error"].abs()
    )

    # ========================================================
    # WORST CASES
    # ========================================================

    worst = (
        df.sort_values(
            "absolute_error",
            ascending=False,
        )
        .head(20)
        .copy()
    )

    worst.to_csv(
        REPORTS_DIR
        / "worst_cases.csv",
        index=False,
    )

    # ========================================================
    # BEST CASES
    # ========================================================

    best = (
        df.sort_values(
            "absolute_error",
            ascending=True,
        )
        .head(20)
        .copy()
    )

    best.to_csv(
        REPORTS_DIR
        / "best_cases.csv",
        index=False,
    )

    # ========================================================
    # OVER-PREDICTION CASES
    # ========================================================

    over_prediction = (
        df.sort_values(
            "error",
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

    # ========================================================
    # UNDER-PREDICTION CASES
    # ========================================================

    under_prediction = (
        df.sort_values(
            "error",
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

    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        f"\nTotal test images: {len(df)}"
    )

    print(
        "\nMean absolute error:"
    )

    print(
        f"{df['absolute_error'].mean():.4f}"
    )

    print(
        "\nMaximum error:"
    )

    print(
        f"{df['absolute_error'].max():.4f}"
    )

    print(
        "\nMedian absolute error:"
    )

    print(
        f"{df['absolute_error'].median():.4f}"
    )

    # ========================================================
    # LARGE ERROR COUNTS
    # ========================================================

    print(
        "\nLarge-error cases:"
    )

    for threshold in [
        0.25,
        0.50,
        0.75,
        1.00,
    ]:

        count = (
            df["absolute_error"]
            >= threshold
        ).sum()

        percentage = (
            count
            / len(df)
            * 100
        )

        print(
            f"Error >= {threshold:.2f}: "
            f"{count} "
            f"({percentage:.2f}%)"
        )

    # ========================================================
    # CONTACT SHEETS
    # ========================================================

    print(
        "\nCreating failure-case images..."
    )

    create_contact_sheet(
        worst,
        REPORTS_DIR
        / "worst_cases_contact_sheet.png",
        "Top 20 Worst Quality Predictions",
    )

    create_contact_sheet(
        best,
        REPORTS_DIR
        / "best_cases_contact_sheet.png",
        "Top 20 Best Quality Predictions",
    )

    create_contact_sheet(
        over_prediction,
        REPORTS_DIR
        / "over_prediction_contact_sheet.png",
        "Largest Quality Over-Predictions",
    )

    create_contact_sheet(
        under_prediction,
        REPORTS_DIR
        / "under_prediction_contact_sheet.png",
        "Largest Quality Under-Predictions",
    )

    # ========================================================
    # ERROR DISTRIBUTION
    # ========================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.hist(
        df["absolute_error"],
        bins=40,
    )

    plt.xlabel(
        "Absolute qMOS Error"
    )

    plt.ylabel(
        "Number of Images"
    )

    plt.title(
        "Distribution of Quality Prediction Errors"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    plt.savefig(
        REPORTS_DIR
        / "failure_error_distribution.png",
        dpi=200,
    )

    plt.close()

    print(
        "\n✓ Failure analysis complete"
    )

    print(
        "\nGenerated:"
    )

    print(
        "  worst_cases.csv"
    )

    print(
        "  best_cases.csv"
    )

    print(
        "  over_prediction_cases.csv"
    )

    print(
        "  under_prediction_cases.csv"
    )

    print(
        "  worst_cases_contact_sheet.png"
    )

    print(
        "  best_cases_contact_sheet.png"
    )

    print(
        "  over_prediction_contact_sheet.png"
    )

    print(
        "  under_prediction_contact_sheet.png"
    )

    print(
        "  failure_error_distribution.png"
    )


if __name__ == "__main__":
    main()