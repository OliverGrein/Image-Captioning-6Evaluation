import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

import csv
import os

RESULTS_PATH = Path(__file__).resolve().parent.parent / "results"

def load_data(model, metric):
    filename = RESULTS_PATH / f"{model}" / f"{model}_{metric}_expert_distances.csv"
    return pd.read_csv(filename)


def create_pivot_table(df):
    return df[df["avg_rating"].isin([1.0, 2.0, 3.0, 4.0])].pivot_table(
        values="avg_distance", index="avg_rating", aggfunc="mean"
    )


def plot_distances(pivot_table, model, metric):
    pivot_table.plot(kind="bar")
    plt.title(
        f"Average {metric.capitalize()} Distance by Rating for {model.capitalize()}"
    )
    plt.xlabel("Average Rating")
    plt.ylabel("Average Distance")
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f"{model}" / f"{metric}_plot.png")
    plt.close()


def get_breakpoints(pivot_table):
    return pivot_table["avg_distance"].tolist()


def classify_distance(distance, breakpoints, metric):
    if metric in ["chebyshev", "euclidian", "minkowski"]:
        if distance <= breakpoints[3]:
            return 4
        elif distance <= breakpoints[2]:
            return 3
        elif distance <= breakpoints[1]:
            return 2
        else:
            return 1
    elif metric == "cosine":
        if distance <= breakpoints[0]:
            return 1
        elif distance <= breakpoints[1]:
            return 2
        elif distance <= breakpoints[2]:
            return 3
        else:
            return 4


def calculate_correlation(df, breakpoints, metric):
    df["predicted_rating"] = df["avg_distance"].apply(
        lambda x: classify_distance(x, breakpoints, metric)
    )
    correlation = df["avg_rating"].corr(df["predicted_rating"], method="kendall")
    return correlation


def main(model, metric):
    df = load_data(model, metric)
    pivot_table = create_pivot_table(df)
    plot_distances(pivot_table, model, metric)

    breakpoints = get_breakpoints(pivot_table)
    correlation = calculate_correlation(df, breakpoints, metric)

    correlations_file = RESULTS_PATH / "correlations.csv"
    file_exists = os.path.isfile(correlations_file)
    with open(correlations_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Model", "Metric", "Correlation"])
        writer.writerow([model, metric, correlation])

    breakpoints_file = RESULTS_PATH / "breakpoints.csv"
    file_exists = os.path.isfile(breakpoints_file)
    with open(breakpoints_file, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(
                [
                    "Model",
                    "Metric",
                    "Breakpoint1",
                    "Breakpoint2",
                    "Breakpoint3",
                    "Breakpoint4",
                ]
            )
        writer.writerow([model, metric] + breakpoints)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate metrics for image captioning"
    )
    parser.add_argument("model", help="Model name (e.g., cohere)")
    parser.add_argument(
        "metric", help="Metric name (e.g., chebyshev, cosine, euclidean, minkowski)"
    )
    args = parser.parse_args()

    main(args.model, args.metric)
