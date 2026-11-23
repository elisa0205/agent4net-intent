import argparse
import csv
import os
import requests
from pathlib import Path
from collections import defaultdict



def write_yaml_to_file(model_name: str, temperature: float, example_name: str, intent_model: str, iteration: int, result: dict):

    if temperature is None:
        temperature = "default"

    model_name = model_name.rsplit("/", 1)[-1]
    intent_model = intent_model.rsplit("/", 1)[-1]

    RESULTS_DIR = Path("test1-results") / model_name / f"temp_{temperature}" / example_name
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    filename = os.path.join(RESULTS_DIR, f"{intent_model}_{iteration}.yaml")

    yaml_content = result.get("generated_yaml", "")

    #correct from EOL to LF
    yaml_code = yaml_content.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"

    # Write the YAML file
    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write(yaml_code)

    write_stats(model_name, temperature, example_name, intent_model, iteration, result)


def write_stats(model_name: str, temperature: float, example_name: str, intent_model: str, iteration: int, result: dict):

    if temperature is None:
        temperature = "default"

    model_name = model_name.rsplit("/", 1)[-1]
    intent_model = intent_model.rsplit("/", 1)[-1]

    RESULTS_DIR = Path("test1-results") / model_name / f"temp_{temperature}" / example_name
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    stats_file = os.path.join(RESULTS_DIR, f"{intent_model}_{iteration}.stats")

    consistency = result.get("consistency", "")
    attempts = result.get("attempts", 0)
    token_usage = result.get("token_usage", 0)
    elapsed_time = result.get("elapsed_time", 0.0)
    consistency_fails = result.get("consistency_fails", 0)
    syntax_fails = result.get("syntax_fails", 0)
    k8s_fails = result.get("k8s_fails", 0)
    k8s_validator_time = result.get("k8s_validator_time", 0.0)

    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"temperature: {temperature}\n")
        f.write(f"consistency: {consistency}\n")
        f.write(f"attempts: {attempts}\n")
        f.write(f"consistency_fails: {consistency_fails}\n")
        f.write(f"syntax_fails: {syntax_fails}\n")
        f.write(f"k8s_fails: {k8s_fails}\n")
        f.write(f"token_usage: {token_usage}\n")
        f.write(f"elapsed_time: {elapsed_time} s\n")
        f.write(f"k8s_validator_time: {k8s_validator_time} s\n")

#Load intents from the CSV file and group them by example name and model
def load_intents(csv_path: Path) -> dict[str, list[str]]:
    grouped = defaultdict(list)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            example_name = row["example"].strip()
            intent_text = row["text"].strip()
            model_name = row["model"].strip()
            grouped[(example_name, model_name)].append(intent_text)
    return dict(grouped)

def generate_manifest(task, model_name, temperature, url):
    payload = {
        "task": task,
        "model_name": model_name,
        "temperature": temperature
    }

    try:
        response = requests.post(url, json=payload)

        if response.status_code == 200:
            content = response.json()
            return content
        else:
            print('Error:', response.status_code, response.text)
            return
    except requests.exceptions.RequestException as e:
        print('Error:', e)
        return


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run the manifest-generation agent over a dataset of intents. "
                    "Requires the FastAPI service (src/main.py) to be running.")
    parser.add_argument("--csv", type=Path, default=Path("test1-dataset.csv"),
                        help="CSV of intents with columns: example, text, model")
    parser.add_argument("--model", default="watsonx/ibm/granite-4-h-small",
                        help="model name in LiteLLM format (e.g. watsonx/... or ollama/...)")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000/manifest")
    args = parser.parse_args()

    intents = load_intents(args.csv)

    for (example, model), tasks in intents.items():
        iteration = 0
        for task in tasks:
            iteration += 1
            result = generate_manifest(task, args.model, args.temperature, args.api_url)
            if result is not None:
                if (result.get("consistency", "") == "INVALID" or
                    result.get("attempts", 0) == result.get("consistency_fails", 0) + result.get("syntax_fails", 0) + result.get("k8s_fails", 0)):
                    write_stats(args.model, args.temperature, example, model, iteration, result)
                else:
                    write_yaml_to_file(args.model, args.temperature, example, model, iteration, result)



    