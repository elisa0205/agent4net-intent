
import os


def write_yaml_to_file(yaml_content: str, attempt: int) -> str:

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"config_attempt_{attempt}.yaml")

    #correct from EOL to LF
    yaml_code = yaml_content.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"

    # Write the YAML file
    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write(yaml_code)

    return filename