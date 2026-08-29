import json

import pandas as pd

from ml.config import REPORTS_DIR


def main():

    cv_file = (
        REPORTS_DIR
        / "cv_baseline_metrics.json"
    )

    cnn_file = (
        REPORTS_DIR
        / "cnn_only_metrics.json"
    )

    hybrid_file = (
        REPORTS_DIR
        / "metrics.json"
    )

    with open(cv_file) as f:
        cv = json.load(f)

    with open(cnn_file) as f:
        cnn = json.load(f)

    with open(hybrid_file) as f:
        hybrid = json.load(f)

    rows = [

        {
            "Model":
                "CV Baseline",

            "MAE":
                cv["test"]["MAE_0_to_5"],

            "RMSE":
                cv["test"]["RMSE_0_to_5"],

            "PLCC":
                cv["test"]["PLCC"],

            "SRCC":
                cv["test"]["SRCC"],
        },

        {
            "Model":
                "CNN-only ResNet18",

            "MAE":
                cnn["MAE_0_to_5"],

            "RMSE":
                cnn["RMSE_0_to_5"],

            "PLCC":
                cnn["PLCC"],

            "SRCC":
                cnn["SRCC"],
        },

        {
            "Model":
                "Hybrid CNN + CV",

            "MAE":
                hybrid["quality"]["MAE_0_to_5"],

            "RMSE":
                hybrid["quality"]["RMSE_0_to_5"],

            "PLCC":
                hybrid["quality"]["PLCC"],

            "SRCC":
                hybrid["quality"]["SRCC"],
        },
    ]

    df = pd.DataFrame(
        rows
    )

    output = (
        REPORTS_DIR
        / "model_comparison.csv"
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        "\nMODEL COMPARISON"
    )

    print(
        df.to_string(
            index=False
        )
    )

    print(
        f"\nSaved: {output}"
    )


if __name__ == "__main__":
    main()