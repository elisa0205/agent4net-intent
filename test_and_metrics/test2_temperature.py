from __future__ import annotations

import argparse
import re
from pathlib import Path
from statistics import mean, pstdev

import matplotlib.pyplot as plt

# GENERA UN GRAFICO (TEMPERATURA IN ASCISSA, UNA LINEA PER MODELLO) PER OGNI METRICA
# Basato sulla stessa logica di parsing di test1_performance_txt.py

NUMBER_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")

METRIC_LABELS = [
    ("BLEU", "bleu"),
    ("CodeBLEU", "codebleu"),
    ("Edit_Distance", "edit_distance"),
    ("Key_Match", "key_match"),
    ("KV_Match", "kv_match"),
    ("KV_Wildcard", "kv_wildcard"),
    ("Kind_Precision", "kind_precision"),
    ("Kind_Recall", "kind_recall"),
]

# Temperature fisse per tutti i modelli
TEMPERATURES = [0.1, 0.7, 1.4]

# Ordine fisso dei modelli in legenda
MODEL_ORDER = [
    "granite-4-h-small",
    "llama-3-3-70b-instruct",
    "llama-4-maverick-17b-128e-instruct-fp8",
    "gpt-oss-120b",
    "mistral-small-3-1-24b-instruct-2503",
    "mistral-large-2512",
]


def ordered_models(models: list[str]) -> list[str]:
    """Ordina i modelli secondo MODEL_ORDER; eventuali modelli non elencati
    vengono aggiunti in coda, in ordine alfabetico."""
    known = [m for m in MODEL_ORDER if m in models]
    unknown = sorted(m for m in models if m not in MODEL_ORDER)
    return known + unknown

# Palette categorica fissa (stesso ordine per ogni modello, ripetibile tra run)
COLOR_CYCLE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X"]
DASH_CYCLE = [(), (6, 3), (2, 2), (4, 1, 1, 1), (1, 1), (5, 2, 1, 2), (3, 3)]


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


def collect_metrics_for_temperature(
    results_dir: Path, temperature: float
) -> dict[str, dict[str, list[float]]]:
    """Uguale a collect_metrics dello script originale, filtrando per una singola temperatura."""
    per_model: dict[str, dict[str, list[float]]] = {}

    temp_tag = f"temp_{temperature}".replace(",", ".")

    for report_file in sorted(results_dir.rglob("*.txt")):
        if temp_tag not in str(report_file):
            continue

        report_data = parse_report_file(report_file)

        if "Generated file" not in report_data:
            continue

        model_name = extract_model_name(report_data, report_file, results_dir)
        per_model.setdefault(model_name, {label: [] for _, label in METRIC_LABELS})

        if "No common kinds found." in report_data:
            continue

        try:
            for source_label, metric_key in METRIC_LABELS:
                if source_label in report_data:
                    per_model[model_name][metric_key].append(parse_number(report_data[source_label]))
        except ValueError as err:
            raise ValueError(f"Invalid numeric value in file {report_file}: {err}") from err

    return per_model


def collect_all_temperatures(
    results_dir: Path, temperatures: list[float]
) -> dict[float, dict[str, dict[str, list[float]]]]:
    """{ temperatura: { modello: { metric_key: [valori] } } }"""
    return {temp: collect_metrics_for_temperature(results_dir, temp) for temp in temperatures}


def build_metric_series(
    data_by_temp: dict[float, dict[str, dict[str, list[float]]]], metric_key: str
) -> dict[str, dict[str, list[float]]]:
    """
    Riorganizza i dati per una singola metrica in:
    { modello: {"means": [...], "stddevs": [...], "temps": [...]} }
    allineati con le temperature che hanno almeno un valore per quel modello.
    """
    models = sorted({model for per_model in data_by_temp.values() for model in per_model})
    series: dict[str, dict[str, list[float]]] = {
        model: {"temps": [], "means": [], "stddevs": []} for model in models
    }

    for temp in TEMPERATURES:
        per_model = data_by_temp.get(temp, {})
        for model in models:
            values = per_model.get(model, {}).get(metric_key, [])
            if not values:
                continue
            series[model]["temps"].append(temp)
            series[model]["means"].append(mean(values))
            series[model]["stddevs"].append(pstdev(values) if len(values) > 1 else 0.0)

    # rimuove modelli senza alcun dato per questa metrica
    return {model: s for model, s in series.items() if s["temps"]}


def plot_metric(metric_label: str, metric_key: str, series: dict[str, dict[str, list[float]]], out_dir: Path) -> None:
    if not series:
        return

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for idx, model_name in enumerate(ordered_models(list(series.keys()))):
        s = series[model_name]
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]
        marker = MARKER_CYCLE[idx % len(MARKER_CYCLE)]
        dash = DASH_CYCLE[idx % len(DASH_CYCLE)]

        temps = s["temps"]
        means = s["means"]

        ax.plot(
            temps, means,
            label=model_name, color=color, marker=marker,
            markersize=6, linewidth=2,
            dashes=dash if dash else (None, None),
        )

    ax.set_xlabel("Temperature")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs temperature")
    ax.set_xticks(TEMPERATURES)
    ax.grid(True, axis="y", linestyle="-", linewidth=0.5, alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))

    fig.tight_layout()
    out_path = out_dir / f"{metric_key}_by_temperature.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Salvato: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Genera un grafico (temperatura in ascissa, una linea per modello) per ogni metrica."
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("test1-results"),
        help="Directory contenente i report .txt generati da manifest_similarity.py",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("test1-results/plots_temperature"),
        help="Directory di output per i grafici PNG",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default=None,
        help="Chiave della metrica da plottare (es. bleu, exact_match). Se omesso, le plotta tutte.",
    )

    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    data_by_temp = collect_all_temperatures(args.results_dir, TEMPERATURES)

    metrics_to_plot = METRIC_LABELS
    if args.metric:
        metrics_to_plot = [(lbl, key) for lbl, key in METRIC_LABELS if key == args.metric]
        if not metrics_to_plot:
            raise SystemExit(f"Metrica sconosciuta: {args.metric}")

    for metric_label, metric_key in metrics_to_plot:
        series = build_metric_series(data_by_temp, metric_key)
        plot_metric(metric_label, metric_key, series, args.out_dir)


if __name__ == "__main__":
    main()