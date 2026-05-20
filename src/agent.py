from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from utils import *

import subprocess


# Agent State
class AgentState(TypedDict):
    task: str
    model_name: str
    generated_yaml: str
    yaml_path: str
    feedback: str
    attempts: int
    consistency: str

# Models one of:
#  - "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
#  - "ollama/qwen3.5:0.8b"

prompt_config = load_prompt_config("..\prompts.yaml")

# Nodes 
def consistency_check(role: str):

    def consistency_node(state: AgentState):
        
        llm = create_llm(state["model_name"])

        if role == "semantic":
            prompt = f"Task: {state['task']}\n\nGenerated YAML:\n{state['generated_yaml']}"
            system_prompt = prompt_config["models"][state['model_name']]["semantic_consistency"]

        elif role == "scope":
            prompt = f"Task: {state['task']}\n"
            system_prompt = prompt_config["models"][state['model_name']]["scope_consistency"]

        else:
            return {"consistency": "INVALID"}

        message = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        print(f"\nConsistency check for the prompt's {role}")

        response = llm.invoke(message)

        if response.content.strip() != "VALID":
            print(f"Prompt consistency check failed:\n{response.content}")
            return {"feedback": response.content,
                    "consistency": "INVALID"}

        print("Consistency check PASSED")
        return {"consistency": "VALID"}   
    
    return consistency_node
    
scope_consistency_node = consistency_check("scope")
semantic_consistency_node = consistency_check("semantic")


def generator_node(state: AgentState):
    """Generate or fix YAML based on the task and feedback"""

    llm = create_llm(state["model_name"])

    prompt = f"Task: {state['task']}\n"
    system_prompt = prompt_config["models"][state['model_name']]["generator"]

    if state['feedback']:
        #Limit the feedback to the last 500 characters to avoid hitting token limits
        feedback_snippet = state['feedback'][-500:]
        
        prompt += f"Previous error to fix: {feedback_snippet}\n YAML to correct: {state['generated_yaml']}"
    
    message = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
        
    print(f"\nCall the LLM\n prompt: {message}\n ")

    try:
        response = llm.invoke(message)
    
    except (Exception) as e:
        print(f"LLM call failed:\n{e}")
        return {"feedback": "FAILED"}

    attempt = state["attempts"] + 1
    #print(f"\n --- Generated YAML (attempt {attempt}): ---\n{response.content}\n--- End of YAML ---\n")
    
    file_path = write_yaml_to_file(response.content, attempt)

    return {"generated_yaml": response.content, 
            "yaml_path": file_path,
            "attempts": attempt}


def syntax_validator_node(state: AgentState):

    print("\nSyntax validator")

    file_path = state["yaml_path"]

    # parsable is needed because in this way the output is machine-readable 
    result = subprocess.run(
        ["yamllint",
         "-c", "..\yamllint_config.yaml",
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
def scope_consistency_should_continue(state: AgentState):
    if state['consistency'] == "INVALID":
        return END
    return "generator"

def generator_should_continue(state: AgentState):
    if state["feedback"] == "FAILED":
        return END
    return "syntax_validator"

def syntax_should_continue(state: AgentState):
    if state['feedback'] == "VALID":
        return "kubernetes_validator"
    elif state['attempts'] > 2:
        return END
    return "generator"

def kubernetes_should_continue(state: AgentState):
    if state['feedback'] == "VALID":
        return "semantic_consistency"
    elif state['attempts'] > 2:
        return END
    return "generator"

def semantic_consistency_should_continue(state: AgentState):
    if state['consistency'] == "VALID":
        return END
    return "generator"

workflow = StateGraph(AgentState)


workflow.add_node("scope_consistency", scope_consistency_node)
workflow.add_node("generator", generator_node)
workflow.add_node("syntax_validator", syntax_validator_node)
workflow.add_node("kubernetes_validator", kubernetes_validator_node)
workflow.add_node("semantic_consistency", semantic_consistency_node)

workflow.set_entry_point("scope_consistency") 
workflow.add_conditional_edges("scope_consistency", scope_consistency_should_continue)
workflow.add_conditional_edges("generator", generator_should_continue)
workflow.add_conditional_edges("syntax_validator", syntax_should_continue)
workflow.add_conditional_edges("kubernetes_validator", kubernetes_should_continue)
workflow.add_conditional_edges("semantic_consistency", semantic_consistency_should_continue)

app = workflow.compile()


