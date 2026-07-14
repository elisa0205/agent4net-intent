from typing import TypedDict
from unittest import result
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
from utils import load_prompt_config, create_llm, normalize_llm_content, write_yaml_to_file, extract_usage_tokens, normalize_error
from utils import KindCluster
from pathlib import Path
from time import perf_counter

import subprocess

BASE_DIR = Path(__file__).resolve().parent


# Agent State
class AgentState(TypedDict):
    task: str
    model_name: str
    generated_yaml: str
    yaml_path: str
    feedback: str
    user_override: str
    manual_feedback: str
    intervention_required: bool
    intervention_message: str
    attempts: int 
    consistency_fails: int
    syntax_fails: int
    k8s_fails: int
    k8s_validator_time: float
    consistency: str
    temperature: float
    token_usage: int
    last_error: str
    repeated_error_count: int 

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

        response = llm.invoke(message)
        #print(response)
        tokens = state["token_usage"] + extract_usage_tokens(response)
        #print(f"Token usage:{response.usage_metadata['total_tokens']}\n")

        response = normalize_llm_content(response.content)

        if response.strip() != "VALID":
            print(f"Prompt consistency check failed:\n{response}")

            if normalize_error(response) == normalize_error(state.get("last_error")):
                return {
                    "feedback": f"Consistency Error: {response}",
                    "last_error": response,
                    "repeated_error_count": state.get("repeated_error_count", 0) + 1,
                    "consistency": "INVALID",
                    "consistency_fails": state.get("consistency_fails", 0) + 1,
                }

            return {"feedback": f"Consistency Error: {response}",
                    "consistency": "INVALID",
                    "consistency_fails": state.get("consistency_fails", 0) + 1,
                    "last_error": response,
                    "repeated_error_count": 0}

        print("PASSED")
        return {"consistency": "VALID",
                "token_usage": tokens}   
    
    return consistency_node
    
scope_consistency_node = consistency_check("scope")
semantic_consistency_node = consistency_check("semantic")


def generator_node(state: AgentState):
    """Generate or fix YAML based on the task and feedback"""

    if(state["user_override"] == "manual_edit"):
        print("\nUser provided manual YAML, skipping generation")

        return {
            "generated_yaml": state["manual_feedback"],
            "yaml_path": state["yaml_path"],
            "attempts": state["attempts"],
            "token_usage": state["token_usage"],
            "user_override": False,
        }
    
    else:
        llm = create_llm(state["model_name"], state["temperature"])

        prompt = f"Task: {state['task']}\n"
        system_prompt = prompt_config["models"][state['model_name']]["generator"]

        if state['feedback']:
            #Limit the feedback to the last 500 characters to avoid hitting token limits
            #feedback_snippet = state['feedback'][-500:]
            feedback_snippet = state['feedback']
            
            prompt += f"Previous error to fix (Keep the existing valid manifest as much as possible, but remove the field flagged by the validator; do not reintroduce unsupported fields; return only the final YAML): {feedback_snippet}\n YAML to correct: {state['generated_yaml']}"
        
        message = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]
            
        print(f"\nCall the LLM: attempt {state['attempts'] + 1}\n")
        #print(f"prompt: {message}\n ")

        try:
            response = llm.invoke(message)
            #print(f"LLM response:\n{response}\n")
            tokens = state["token_usage"] + extract_usage_tokens(response)
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
                "attempts": attempt,
                "token_usage": tokens}


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

        # Detect repeated identical errors
        if normalize_error(error_message) == normalize_error(state.get("last_error")):
            return {
                "feedback": f"Yamllint Error: {error_message}",
                "syntax_fails": state.get("syntax_fails", 0) + 1,
                "last_error": error_message,
                "repeated_error_count": state.get("repeated_error_count", 0) + 1
            }

        return {
            "feedback": f"Yamllint Error: {error_message}",
            "syntax_fails": state.get("syntax_fails", 0) + 1,
            "last_error": error_message,
            "repeated_error_count": 0
        }


def kubernetes_validator_node(state: AgentState):

    print("\nKubernetes validator")

    file_path = state['yaml_path']
    CLUSTER_CONFIG_PATH = BASE_DIR / "utils" / "cluster-config.yaml"

    result = None
    start = perf_counter()

    try:
        with KindCluster(config = CLUSTER_CONFIG_PATH) as kc:
            try:
                kc.apply(file_path)
                print("PASSED")
                result = {"feedback": "VALID",
                          "attempts": state['attempts']}
        
            except subprocess.CalledProcessError as e:
                # (getattr(e, "stdout", "") or "") + 
                err = (getattr(e, "stderr", "") or "")
                print(f"--- Error detected---\n {err} ---")
    
                # Loop detection: same error repeated
                if normalize_error(err) == normalize_error(state.get("last_error")):
                    repeated = state.get("repeated_error_count", 0) + 1
                else:
                    repeated = 0

                result = {
                    "feedback": f"Kubernetes Validation Error: {err}",
                    "attempts": state["attempts"],
                    "k8s_fails": state.get("k8s_fails", 0) + 1,
                    "last_error": err,
                    "repeated_error_count": repeated
                }
    
    except subprocess.CalledProcessError as e:
        err = (getattr(e, "stdout", "") or "") + (getattr(e, "stderr", "") or "")
        result = {"feedback": f"Kind Creation Error: {err}"}

    finally:
        elapsed = perf_counter() - start
        if result is not None:
            result["k8s_validator_time"] = state.get("k8s_validator_time", 0.0) + elapsed

    return result


def user_intervention_node(state: AgentState):
    print("\nUser intervention required")

    print(f"Repeated error:\n{state['last_error']}")
    intervention_message = (
        "User intervention required. Provide user_override='continue' to keep trying, "
        "or user_override='manual_edit' with manual_yaml to resume with corrected yaml file."
    )

    user_override = state.get("user_override", "stop")

    if user_override == "continue":
        return {
            "user_override": "continue",
            "intervention_required": False,
            "intervention_message": ""
        }

    elif user_override == "manual_edit":
        new_yaml = state.get("manual_yaml", "").strip()

        if not new_yaml:
            return {
                "feedback": "FAILED: Manual yaml missing",
                "user_override": "stop",
                "intervention_required": True,
                "intervention_message": intervention_message
            }
        
        attempt = state["attempts"] + 1
        file_path = write_yaml_to_file(new_yaml, attempt)

        return {
            "user_override": "manual_edit",
            "generated_yaml": new_yaml,
            "yaml_path": file_path,
            "attempts": attempt,
            "intervention_required": False,
            "intervention_message": ""
        }

    else:
        return {
            "feedback": "FAILED: User intervention required",
            "user_override": "stop",
            "intervention_required": True,
            "intervention_message": intervention_message
        }




# Logic
def scope_consistency_should_continue(state: AgentState):
    if state['repeated_error_count'] >= 2:
        return "user_intervention"
    if state['consistency'] == "INVALID":
        return END
    return "generator"
    
def generator_should_continue(state: AgentState):
    if state["feedback"] == "FAILED":
        return END
    return "syntax_validator"

def syntax_should_continue(state: AgentState):
    if state['repeated_error_count'] >= 2:
        return "user_intervention"
    if state['feedback'] == "VALID":
        return "kubernetes_validator"
    elif state['attempts'] > 6:
        state['feedback'] = "FAILED: Maximum attempts reached"
        return END
    return "generator"

def kubernetes_should_continue(state: AgentState):
    if state['repeated_error_count'] >= 2:
        return "user_intervention"
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
workflow.add_node("user_intervention", user_intervention_node)

workflow.set_entry_point("scope_consistency") 
workflow.add_conditional_edges("scope_consistency", scope_consistency_should_continue)
workflow.add_conditional_edges("generator", generator_should_continue)
workflow.add_conditional_edges("syntax_validator", syntax_should_continue)
workflow.add_conditional_edges("kubernetes_validator", kubernetes_should_continue)
workflow.add_conditional_edges("semantic_consistency", semantic_consistency_should_continue)
workflow.add_conditional_edges("user_intervention", lambda s: (
    "generator" if s["user_override"] in ["continue", "manual_edit"] else END
    ))
app = workflow.compile()


