import requests
import csv
import os
from pathlib import Path
from collections import defaultdict

#CSV_PATH = Path("expanded-dataset(in).csv")
CSV_PATH = Path("expanded-test.csv")



def write_yaml_to_file(model_name: str, temperature: float, example_name: str, intent_model: str, iteration: int, attempts: int, yaml_content: str, token_usage: int):

    if temperature is None:
        temperature = "default"

    model_name = model_name.rsplit("/", 1)[-1]
    intent_model = intent_model.rsplit("/", 1)[-1]

    RESULTS_DIR = Path("results") / model_name / f"temp_{temperature}" / example_name
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    filename = os.path.join(RESULTS_DIR, f"{intent_model}_{iteration}.yaml")

    #correct from EOL to LF
    yaml_code = yaml_content.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"

    # Write the YAML file
    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write(yaml_code)

    write_stats(model_name, temperature, example_name, intent_model, iteration, attempts, token_usage)


def write_stats(model_name: str, temperature: float, example_name: str, intent_model: str, iteration: int, attempts: int, token_usage: int):

    if temperature is None:
        temperature = "default"

    model_name = model_name.rsplit("/", 1)[-1]
    intent_model = intent_model.rsplit("/", 1)[-1]

    RESULTS_DIR = Path("results") / model_name / f"temp_{temperature}" / example_name
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    stats_file = os.path.join(RESULTS_DIR, f"{intent_model}_{iteration}.stats")

    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(f"model_name: {model_name}\n")
        f.write(f"temperature: {temperature}\n")
        f.write(f"attempts: {attempts}\n")
        f.write(f"token_usage: {token_usage}\n")

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

def generate_manifest(task, model_name, temperature):
    url = "http://127.0.0.1:8000/manifest"

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
    
    model_name = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
    temperature = 0.7 # default temperature 0.7

    intents = load_intents(CSV_PATH)
    
    for (example, model), tasks in intents.items():
        iteration = 0
        for task in tasks:
            iteration += 1
            result = generate_manifest(task, model_name, temperature)
            if result is not None:
                yaml_content = result.get("generated_yaml", "")
                attempts = result.get("attempts", 0)
                token_usage = result.get("token_usage", 0)
                write_yaml_to_file(model_name, temperature, example, model, iteration, attempts, yaml_content, token_usage)



    