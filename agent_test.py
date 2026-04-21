from typing import Annotated, TypedDict
from langchain_ollama import ChatOllama
from langchain_litellm import ChatLiteLLM
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage

from utils import write_yaml_to_file

# Import for YAML validation
import subprocess
import os
import re

# Agent State
class AgentState(TypedDict):
    task: str
    generated_yaml: str
    yaml_path: str
    feedback: str
    attempts: int

#model
llm : ChatLiteLLM = ChatLiteLLM(
    model="ollama/qwen3.5:2b",
    streaming=False,
)

SYSTEM_PROMPT = """
You are an LLM specialized in generating Kubernetes manifests.

## Domain Description
Kubernetes is a system of containers orchestrator that allows you to deploy and manage containerized applications. 
Kubernetes manifests are YAML files that describe the desired state of your Kubernetes resources.
The main resources are:
- Pods: the smallest deployable units that can be created and managed in Kubernetes.
- Deployments: a higher-level abstraction that manages a group of pods and ensures that the desired number of replicas are running.
- Services: an abstraction that defines a logical set of pods and a policy by which to access them.
- Ingress: an API object that manages external access to the services in a cluster, typically HTTP.
- ConfigMap and Secret: used to store configuration data and sensitive information, respectively.
- PersistentVolume and PersistentVolumeClaim: used for managing storage in Kubernetes.

Resources base format:
Resource:
  apiVersion: string (required)
  kind: string (required)
  metadata:
    name: string (required)
    labels: map<string,string> (optional)
  spec: object (required)

## Requirements
- Output ONLY valid Kubernetes YAML.
- Do not include explanations or comments.
- If multiple resources are needed, separate them with ---.
- Do not overthink.
"""


# Nodes 
def generator_node(state: AgentState):
    """Generate or fix YAML based on the task and feedback"""

    if state['feedback']:
        #Limit the feedback to the last 500 characters to avoid hitting token limits
        feedback_snippet = state['feedback'][-500:]
        
        prompt = f"##Previous error to fix: {feedback_snippet}\n YAML to correct: {state['generated_yaml']}"
        message = [
            SystemMessage(content=prompt)
        ]
    else:
        prompt = f"##Task:\n {state['task']}\n"
        message = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
    print(f"\nCall the LLM\n prompt: {message}\n ")

    try:
        response = llm.invoke(message)
    
    except (Exception) as e:
        print(f"LLM call failed: {e}")
        return {END}

    attempt = state["attempts"] + 1
    print(f"\n --- Generated YAML (attempt {attempt}): ---\n{response.content}\n--- End of YAML ---\n")
    
    file_path = write_yaml_to_file(response.content, attempt)

    return {"generated_yaml": response.content, 
            "yaml_path": file_path,
            "attempts": attempt}


def syntax_validator_node(state: AgentState):

    print("\nSyntax validator")

    file_path = state["yaml_path"]

    # parsable is needed because in this way the output is machine-readable 
    result = subprocess.run(
        ["yamllint", "-d", 
         "{extends: default, rules: {document-start: disable, indentation: {indent-sequences: consistent}}}", 
         "-f", "parsable", file_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # No errors
        return {"feedback": "VALID", 
                "attempts": state['attempts']}
    
    else:
        error_message = result.stdout + result.stderr

        print(f"--- Error detected---\n {error_message} ---")
        return {"feedback": f"Yamllint Error: {error_message}", 
                "attempts": state['attempts']}


def kubernetes_validator_node(state: AgentState):

    print("\nKubernetes validator")

    file_path = state['yaml_path']

    result = subprocess.run(
        ["kubectl", "apply",  "-f", file_path, "--dry-run=server"],
        capture_output=True,
        text=True
    )

    if result.returncode == 0:
        return {"feedback": "VALID",
                "attempts": state['attempts']}
    else:
        error_message = result.stdout + result.stderr
        print(f"--- Error detected---\n {error_message} ---")
        return {"feedback": f"Kubernetes Validation Error: {error_message}",
                "attempts": state['attempts']}


# Logic
def syntax_should_continue(state: AgentState):
    if state['feedback'] == "VALID":
        return "kubernetes_validator"
    elif state['attempts'] > 2:
        return END
    return "generator"

def kubernetes_should_continue(state: AgentState):
    if state['feedback'] == "VALID" or state['attempts'] > 3:
        return END
    return "generator"


workflow = StateGraph(AgentState)

workflow.add_node("generator", generator_node)
workflow.add_node("syntax_validator", syntax_validator_node)
workflow.add_node("kubernetes_validator", kubernetes_validator_node)

workflow.set_entry_point("generator") 
workflow.add_edge("generator", "syntax_validator")
workflow.add_conditional_edges("syntax_validator", syntax_should_continue)
workflow.add_conditional_edges("kubernetes_validator", kubernetes_should_continue)

app = workflow.compile()


inputs = {
    "task": "Create a simple configuration for a kubernetes cluster with a nginx server and expose it on port 80",
    "generated_yaml": "",
    "yaml_path": "",
    "feedback": "",
    "attempts": 0
}

for output in app.stream(inputs):
    print(output)

