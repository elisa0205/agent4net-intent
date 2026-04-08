from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_litellm import ChatLiteLLM
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage


# Import for YAML validation
import subprocess
import os
import re

# Agent State
class AgentState(TypedDict):
    task: str
    generated_yaml: str
    feedback: str
    attempts: int

#model
llm : ChatLiteLLM = ChatLiteLLM(
    model="ollama/qwen3.5:2b",
    api_base="http://localhost:11434",
    streaming=False,
)

SYSTEM_PROMPT = """You are a Kubernetes YAML generator.
Return ONLY valid Kubernetes YAML.
No explanations. No markdown fences. No comments.
If multiple resources are needed, separate them with ---.
Do not overthink."""


# Nodes 
def generator_node(state: AgentState):
    """Generate or fix YAML based on the task and feedback"""

    prompt = f"Task: {state['task']}\n"
    if state['feedback']:
        prompt += f"Previous error to fix: {state['feedback']}\n YAML to correct: {state['generated_yaml']}"
        
    message = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]

    print(f"Call the LLM\n prompt: {message}\n ")

    response = llm.invoke(message)

    print(f"\n --- Generated YAML (attempt {state['attempts'] + 1}): ---\n{response.content}\n--- End of YAML ---\n")

    return {"generated_yaml": response.content, "attempts": state['attempts'] + 1}


def syntax_validator_node(state: AgentState):

    print("Syntax validator")

    yaml_code = state['generated_yaml']
    filename = "temp_config.yaml"

    #correct from EOL to LF
    yaml_code = yaml_code.replace("\r\n", "\n").replace("\r", "\n").rstrip() + "\n"

    # Write the temporary YAML file
    with open(filename, "w", encoding="utf-8", newline="\n") as f:
        f.write(yaml_code)
    
    # parsable is needed because in this way the output is machine-readable 
    result = subprocess.run(
        ["yamllint", "-f", "parsable", filename],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # No errors
        os.remove(filename)
        return {"feedback": "VALID", "attempts": state['attempts']}
    
    elif result.returncode == 1:
        error_message = result.stdout + result.stderr
        os.remove(filename)

        print(f"--- Error detected---\n {error_message} ---")
        return {"feedback": f"Yamllint Error: {error_message}", "attempts": state['attempts']}


# Logic
def should_continue(state: AgentState):
    if state['feedback'] == "VALID" or state['attempts'] > 2:
        return END
    return "generator"



workflow = StateGraph(AgentState)

workflow.add_node("generator", generator_node)
workflow.add_node("syntax_validator", syntax_validator_node)

workflow.set_entry_point("generator") 
workflow.add_edge("generator", "syntax_validator")
workflow.add_conditional_edges("syntax_validator", should_continue)

app = workflow.compile()


inputs = {
    "task": "Create a simple Service ClusterIP Manifest for a app exposing port 8080.",
    "generated_yaml": "",
    "feedback": "",
    "attempts": 0
}

for output in app.stream(inputs):
    print(output)

