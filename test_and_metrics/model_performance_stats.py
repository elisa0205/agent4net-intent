from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean, pvariance, pstdev

import matplotlib.pyplot as plt


REQUIRED_FIELDS = ("attempts", "token_usage", "elapsed_time")
NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_stats_file(file_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}

    with file_path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip()

    return data


def parse_number(value: str) -> float:
    match = NUMBER_RE.search(value)
    if not match:
        raise ValueError(f"Unable to parse numeric value from: {value}")
    return float(match.group(0))


def extract_model_name(stats_data: dict[str, str], stats_file: Path, results_dir: Path) -> str:
    model_name = stats_data.get("model_name", "").strip()
    if model_name:
        return model_name

    # Fallback to directory structure: test1-results/<model>/...
    try:
        rel = stats_file.relative_to(results_dir)
        if len(rel.parts) >= 1:
            return rel.parts[0]
    except ValueError:
        pass

    return "unknown"


def collect_metrics(results_dir: Path) -> dict[str, dict[str, list[float]]]:
    per_model: dict[str, dict[str, list[float]]] = {}

    for stats_file in results_dir.rglob("*.stats"):
        stats_data = parse_stats_file(stats_file)
        model_name = extract_model_name(stats_data, stats_file, results_dir)

        if model_name not in per_model:
            per_model[model_name] = {field: [] for field in REQUIRED_FIELDS}

        try:
            if stats_data.get("consistency") == "VALID":
                per_model[model_name]["attempts"].append(parse_number(stats_data["attempts"]))
                per_model[model_name]["token_usage"].append(parse_number(stats_data["token_usage"]))
                per_model[model_name]["elapsed_time"].append(parse_number(stats_data["elapsed_time"]))
        except KeyError as missing:
            raise KeyError(f"Missing field {missing} in file: {stats_file}") from missing
        except ValueError as err:
            raise ValueError(f"Invalid numeric value in file {stats_file}: {err}") from err

    return per_model


def summarize(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("Cannot summarize an empty list of values")

    return {
        "mean": mean(values),
        "min": min(values),
        "max": max(values),
        "variance": pvariance(values),
        "stddev": pstdev(values),
    }


def count_outliers(values: list[float]) -> int:
    if len(values) < 4:
        return 0

    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2

    def median(data: list[float]) -> float:
        size = len(data)
        center = size // 2
        if size % 2 == 0:
            return (data[center - 1] + data[center]) / 2
        return data[center]

    if len(sorted_values) % 2 == 0:
        lower_half = sorted_values[:midpoint]
        upper_half = sorted_values[midpoint:]
    else:
        lower_half = sorted_values[:midpoint]
        upper_half = sorted_values[midpoint + 1 :]

    q1 = median(lower_half)
    q3 = median(upper_half)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return sum(1 for value in values if value < lower_bound or value > upper_bound)


def build_summary(per_model: dict[str, dict[str, list[float]]]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []

    for model_name in sorted(per_model.keys()):
        metrics = per_model[model_name]
        attempts_stats = summarize(metrics["attempts"])
        token_stats = summarize(metrics["token_usage"])
        elapsed_stats = summarize(metrics["elapsed_time"])

        rows.append(
            {
                "model_name": model_name,
                "num_samples": len(metrics["attempts"]),
                "attempts_mean": attempts_stats["mean"],
                "attempts_min": attempts_stats["min"],
                "attempts_max": attempts_stats["max"],
                "attempts_variance": attempts_stats["variance"],
                "attempts_stddev": attempts_stats["stddev"],
                "token_usage_mean": token_stats["mean"],
                "token_usage_min": token_stats["min"],
                "token_usage_max": token_stats["max"],
                "token_usage_variance": token_stats["variance"],
                "token_usage_stddev": token_stats["stddev"],
                "elapsed_time_mean_s": elapsed_stats["mean"],
                "elapsed_time_min_s": elapsed_stats["min"],
                "elapsed_time_max_s": elapsed_stats["max"],
                "elapsed_time_variance_s": elapsed_stats["variance"],
                "elapsed_time_stddev_s": elapsed_stats["stddev"],
            }
        )

    return rows


def print_summary(rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        print("No .stats files found.")
        return

    for row in rows:
        print(f"\nModel: {row['model_name']}")
        print(f"Samples: {row['num_samples']}")
        print(
            "Attempts -> "
            f"mean={row['attempts_mean']:.4f}, "
            f"min={row['attempts_min']:.4f}, "
            f"max={row['attempts_max']:.4f}, "
            f"variance={row['attempts_variance']:.4f}, "
            f"stddev={row['attempts_stddev']:.4f}"
        )
        print(
            "Token usage -> "
            f"mean={row['token_usage_mean']:.4f}, "
            f"min={row['token_usage_min']:.4f}, "
            f"max={row['token_usage_max']:.4f}, "
            f"variance={row['token_usage_variance']:.4f}, "
            f"stddev={row['token_usage_stddev']:.4f}"
        )
        print(
            "Elapsed time (s) -> "
            f"mean={row['elapsed_time_mean_s']:.4f}, "
            f"min={row['elapsed_time_min_s']:.4f}, "
            f"max={row['elapsed_time_max_s']:.4f}, "
            f"variance={row['elapsed_time_variance_s']:.4f}, "
            f"stddev={row['elapsed_time_stddev_s']:.4f}"
        )


def generate_box_plot(per_model: dict[str, dict[str, list[float]]], output_plot: Path) -> list[Path]:
    if not per_model:
        return []

    models = sorted(per_model.keys())

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    def render_single_plot(
        title: str,
        metric_values: list[list[float]],
        suffix: str,
    ) -> None:
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)

        flierprops = {
            "marker": "o",
            "markerfacecolor": "none",
            "markeredgecolor": "black",
            "markersize": 6,
            "linestyle": "none",
            "alpha": 1.0,
        }

        meanprops = {
            "marker": "x",
            "markeredgecolor": "black",
            "markersize": 6,
        }

        box_color_map = {
            "Attempts": "#64c4f8",
            "Token Usage": "#faa357",
            "Elapsed Time (seconds)": "#f478ca",
        }

        boxprops = {"facecolor": box_color_map.get(title, "#7db6f8"), "edgecolor": "black"}
        medianprops = {"color": "#111111", "linewidth": 2}
        whiskerprops = {"color": "#4a4a4a", "linewidth": 1.4}
        capprops = {"color": "#4a4a4a", "linewidth": 1.4}

        ax.boxplot(
            metric_values,
            widths=0.30,
            patch_artist=True,
            showfliers=True,
            showmeans=True,
            meanprops=meanprops,
            flierprops=flierprops,
            boxprops=boxprops,
            medianprops=medianprops,
            whiskerprops=whiskerprops,
            capprops=capprops,
        )

        ax.set_xticks(range(1, len(models) + 1))
        ax.set_xticklabels(models, rotation=15, ha="right")

        ax.set_title(f"{title}")
        ax.set_xlabel("Model")
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_ylabel(title)

        output_file = output_plot.with_name(f"{output_plot.stem}_{suffix}_boxplot{output_plot.suffix}")
        fig.savefig(output_file, dpi=150)
        output_paths.append(output_file)
        plt.close(fig)

    render_single_plot("Attempts", [per_model[model]["attempts"] for model in models], "attempts")
    render_single_plot("Token Usage", [per_model[model]["token_usage"] for model in models], "token_usage")
    render_single_plot("Elapsed Time (seconds)", [per_model[model]["elapsed_time"] for model in models], "elapsed_time")

    return output_paths

def generate_errorbar_plots(rows: list[dict[str, float | str | int]], output_plot: Path) -> list[Path]:
    if not rows:
        return []

    output_plot.parent.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    metrics = [
        (
            "Attempts",
            "attempts_mean",
            "attempts_stddev",
            "#64c4f8",
            "attempts",
        ),
        (
            "Token Usage",
            "token_usage_mean",
            "token_usage_stddev",
            "#faa357",
            "token_usage",
        ),
        (
            "Elapsed Time (seconds)",
            "elapsed_time_mean_s",
            "elapsed_time_stddev_s",
            "#f478ca",
            "elapsed_time",
        ),
    ]

    models = [row["model_name"] for row in rows]

    for title, mean_key, std_key, color, suffix in metrics:
        means = [float(row[mean_key]) for row in rows]
        errors = [float(row[std_key]) for row in rows]
        print(f"Generating error bar plot for {title}: means={means}, errors={errors}")

        fig, ax = plt.subplots(figsize=(10, 12), constrained_layout=True)
        
        ax.bar(
            models,
            means,
            yerr=errors,
            color=color,
            edgecolor="black",
            capsize=6,
            alpha=0.9,
            width = 0.38
        )

        ax.set_title(title)
        ax.set_xlabel("Model")
        ax.set_ylabel(title)
        ax.tick_params(axis="x", labelrotation=15)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        output_file = output_plot.with_name(f"{output_plot.stem}_{suffix}_errorbar{output_plot.suffix}")
        fig.savefig(output_file, dpi=150)
        output_paths.append(output_file)
        plt.close(fig)

    return output_paths

def main() -> None:

    results_dir = Path("test1-results")
    output_plot = Path("test1-results/model_performance_summary.png")

    per_model = collect_metrics(results_dir)
    rows = build_summary(per_model)
    print_summary(rows)

    boxplot_paths = generate_box_plot(per_model, output_plot)
    for path in boxplot_paths:
        print(f"Box plot written to: {path}")

    errorbar_paths = generate_errorbar_plots(rows, output_plot)
    for path in errorbar_paths:
        print(f"Error bar plot written to: {path}")


if __name__ == "__main__":
    main()
