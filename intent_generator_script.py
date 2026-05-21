from langchain_litellm import ChatLiteLLM
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage
import os

class AgentState(TypedDict):
    task: str
    response: str
    
model_name = "watsonx/meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
project_id = os.environ["WATSONX_PROJECT_ID"]

def generate_intent_from_manifests(state: AgentState) -> dict:
    
    with open("./configuration_examples/postgres-example/complete.yaml", "r", encoding="utf-8") as f:
        testo = f.read()

    message = [
        SystemMessage(content="You are a helpful assistant that generates a concise intent describing the purpose and main characteristics of a Kubernetes manifest, based on the manifest content provided. Write it general as a human would describe it, without mentioning the manifest itself."),
        HumanMessage(content=f"Task: {state['task']}\n\nManifest:\n{testo}")
    ]

    print(f"{message}\n")

    llm = ChatLiteLLM(model=model_name, project_id=project_id)

    response = llm.invoke(message)

    print(f"{response.content}\n")

workflow = StateGraph(AgentState)
workflow.add_node("generate_intent", generate_intent_from_manifests)

workflow.set_entry_point("generate_intent")
workflow.add_edge("generate_intent", END)

app = workflow.compile()

inputs = {
    "task": "Generate an intent based on the manifest passed, describing what is the purpose of the configuration and which are the main characteristics.",
    "response": ""
}
for output in app.stream(inputs):
    print(output)