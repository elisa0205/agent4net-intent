from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from utils import load_prompt_config, create_llm, normalize_llm_content, write_yaml_to_file
from utils import KindCluster
from pathlib import Path

import subprocess

BASE_DIR = Path(__file__).resolve().parent


# Agent State
class AgentState(TypedDict):
    task: str
    model_name: str
    generated_yaml: str
    yaml_path: str
    feedback: str
    attempts: int
    consistency: str
    temperature: float
    

prompt_config = load_prompt_config(BASE_DIR / ".." / "prompts.yaml")

# Nodes 
def consistency_check(role: str):

    def consistency_node(state: AgentState):
        
        llm = create_llm(state["model_name"], state["temperature"])

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

        response = normalize_llm_content(llm.invoke(message).content)

        if response.strip() != "VALID":
            print(f"Prompt consistency check failed:\n{response}")
            return {"feedback": f"Consistency Error: {response}",
                    "consistency": "INVALID"}

        print("PASSED")
        return {"consistency": "VALID"}   
    
    return consistency_node
    
scope_consistency_node = consistency_check("scope")
semantic_consistency_node = consistency_check("semantic")


def generator_node(state: AgentState):
    """Generate or fix YAML based on the task and feedback"""

    llm = create_llm(state["model_name"], state["temperature"])

    prompt = f"Task: {state['task']}\n"
    system_prompt = prompt_config["models"][state['model_name']]["generator"]

    if state['feedback']:
        #Limit the feedback to the last 500 characters to avoid hitting token limits
        #feedback_snippet = state['feedback'][-500:]
        feedback_snippet = state['feedback']
        
        prompt += f"Previous error to fix: {feedback_snippet}\n YAML to correct: {state['generated_yaml']}"
    
    message = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=prompt)
    ]
        
    print(f"\nCall the LLM: attempt {state['attempts'] + 1}\n")
    #print(f"prompt: {message}\n ")

    try:
        response = llm.invoke(message)
        #print(f"LLM metadata:\n{response}\n")
        response = normalize_llm_content(response.content)

    
    except (Exception) as e:
        print(f"LLM call failed:\n{e}")
        return {"feedback": "FAILED"}

    attempt = state["attempts"] + 1
    #print(f"\n --- Generated YAML (attempt {attempt}): ---\n{response}\n--- End of YAML ---\n")
    
    file_path = write_yaml_to_file(response, attempt)

    return {"generated_yaml": response, 
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
        print("PASSED")
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

    CLUSTER_CONFIG_PATH = BASE_DIR / "utils" / "cluster-config.yaml"

    try:
        with KindCluster(config = CLUSTER_CONFIG_PATH) as kc:
            try:
                kc.apply(file_path)
                print("PASSED")
                return {"feedback": "VALID",
                        "attempts": state['attempts']}
        
            except subprocess.CalledProcessError as e:
                # (getattr(e, "stdout", "") or "") + 
                err = (getattr(e, "stderr", "") or "")
                print(f"--- Error detected---\n {err} ---")
                return {"feedback": f"Kubernetes Validation Error: {err}", 
                        "attempts": state["attempts"]}
    
    except subprocess.CalledProcessError as e:
        err = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
        return {"feedback": f"Kind Creation Error: {err}"}


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
    elif state['attempts'] > 6:
        state['feedback'] = "FAILED: Maximum attempts reached"
        return END
    return "generator"

def kubernetes_should_continue(state: AgentState):
    if state['feedback'] == "VALID":
        return "semantic_consistency"
    elif state['attempts'] > 6:
        state['feedback'] = "FAILED: Maximum attempts reached"
        return END
    elif state['feedback'].startswith("Kind Creation Error"):
        state['feedback'] = "FAILED: Kind cluster creation failed"
        return END
    return "generator"

def semantic_consistency_should_continue(state: AgentState):
    if state['consistency'] == "VALID":
        return END
    elif state['attempts'] > 6:
        state['feedback'] = "FAILED: Maximum attempts reached"
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


