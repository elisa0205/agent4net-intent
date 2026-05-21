
import os
import yaml
from functools import lru_cache
from langchain_litellm import ChatLiteLLM

# From generated agent text to file 
def write_yaml_to_file(yaml_content: str, attempt: int) -> str:

    results_dir = "..\\results"
    os.makedirs(results_dir, exist_ok=True)
    filename = os.path.join(results_dir, f"config_attempt_{attempt}.yaml")

    #correct from EOL to LF
    yaml_code = yaml_content.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"

    # Write the YAML file
    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write(yaml_code)

    return filename

# Load YAML file correctly
def load_prompt_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)
    
# Factory function to create the LLM based on the model name passed
# lru_cache is used to avoid creating multiple instances of the same model, which can be expensive in terms of resources and time.
# When the function is called with a model name, it checks if an instance of that model already exists in the cache. 
# If it does, it returns the cached instance. If not, it creates a new instance, stores it in the cache, and then returns it. 
@lru_cache(maxsize=8)
def create_llm(model_name: str) -> ChatLiteLLM:
    if model_name.startswith("watsonx/"):

        project_id = os.environ["WATSONX_PROJECT_ID"]
        os.environ["WATSONX_API_KEY"]
        os.environ["WATSONX_API_BASE"]

        return ChatLiteLLM(model=model_name, project_id=project_id)

    if model_name.startswith("ollama/"):
        return ChatLiteLLM(model=model_name, streaming=False)

    raise ValueError(f"Unsupported model prefix for model: {model_name}")