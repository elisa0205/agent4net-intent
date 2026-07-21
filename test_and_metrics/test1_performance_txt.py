from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean, pvariance, pstdev

NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

METRIC_LABELS = [
    ("BLEU", "bleu"),
    ("CodeBLEU", "codebleu"),
    ("Edit_Distance", "edit_distance"),
    ("Exact_Match", "exact_match"),
    ("Key_Match", "key_match"),
    ("KV_Match", "kv_match"),
    ("KV_Wildcard", "kv_wildcard"),
    ("Matched kinds", "matched_kinds"),
    ("Kind_Precision", "kind_precision"),
    ("Kind_Recall", "kind_recall"),
]


def parse_report_file(file_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}

    with file_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
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


def extract_model_name(report_data: dict[str, str], report_file: Path, results_dir: Path) -> str:
    generated_file = report_data.get("Generated file", "").strip()
    if generated_file:
        try:
            generated_path = Path(generated_file)
            rel = generated_path.relative_to(results_dir)
            if rel.parts:
                return rel.parts[0]
        except ValueError:
            pass

    try:
        rel = report_file.relative_to(results_dir)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass

    return "unknown"


def collect_metrics(results_dir: Path) -> dict[str, dict[str, list[float]]]:
    per_model: dict[str, dict[str, list[float]]] = {}

    for report_file in sorted(results_dir.rglob("*.txt")):
        report_data = parse_report_file(report_file)

        if "Generated file" not in report_data:
            continue

        if "No common kinds found." in report_data:
            model_name = extract_model_name(report_data, report_file, results_dir)
            per_model.setdefault(model_name, {label: [] for _, label in METRIC_LABELS})
            continue

        model_name = extract_model_name(report_data, report_file, results_dir)
        per_model.setdefault(model_name, {label: [] for _, label in METRIC_LABELS})

        try:
            for source_label, metric_key in METRIC_LABELS:
                if source_label in report_data:
                    per_model[model_name][metric_key].append(parse_number(report_data[source_label]))
        except ValueError as err:
            raise ValueError(f"Invalid numeric value in file {report_file}: {err}") from err

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


def build_summary(per_model: dict[str, dict[str, list[float]]]) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []

    for model_name in sorted(per_model.keys()):
        metrics = per_model[model_name]
        row: dict[str, float | str | int] = {
            "model_name": model_name,
            "num_samples": 0,
        }

        first_metric_values = next((values for values in metrics.values() if values), [])
        row["num_samples"] = len(first_metric_values)

        for metric_key, values in metrics.items():
            if not values:
                continue
            stats = summarize(values)
            row[f"{metric_key}_mean"] = stats["mean"]
            row[f"{metric_key}_min"] = stats["min"]
            row[f"{metric_key}_max"] = stats["max"]
            row[f"{metric_key}_variance"] = stats["variance"]
            row[f"{metric_key}_stddev"] = stats["stddev"]

        rows.append(row)

    return rows


def print_summary(rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        print("No .txt report files found.")
        return

    for row in rows:
        print(f"\nModel: {row['model_name']}")
        print(f"Samples: {row['num_samples']}")

        for _, metric_key in METRIC_LABELS:
            mean_key = f"{metric_key}_mean"
            min_key = f"{metric_key}_min"
            max_key = f"{metric_key}_max"
            variance_key = f"{metric_key}_variance"
            stddev_key = f"{metric_key}_stddev"

            if mean_key not in row:
                continue

            label = metric_key.replace("_", " ").title()
            print(
                f"{label} -> "
                f"mean={float(row[mean_key]):.4f}, "
                f"min={float(row[min_key]):.4f}, "
                f"max={float(row[max_key]):.4f}, "
                f"variance={float(row[variance_key]):.4f}, "
                f"stddev={float(row[stddev_key]):.4f}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize manifest_similarity.py text reports."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("test1-structure-results"),
        help="Directory containing the .txt reports generated by manifest_similarity.py",
    )
    args = parser.parse_args()

    per_model = collect_metrics(args.results_dir)
    rows = build_summary(per_model)
    print_summary(rows)


if __name__ == "__main__":
    main()