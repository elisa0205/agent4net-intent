from fastapi import FastAPI
from pydantic import BaseModel
from agent import app as agent_app

app = FastAPI() 

# Request Model
class AgentRequest(BaseModel):
    task: str
    model_name: str

# Response Model
class AgentResponse(BaseModel):
    generated_yaml: str
    yaml_path: str
    attempts: int
    status: str


@app.post("/generate_yaml")
async def generate_yaml(request: AgentRequest) -> AgentResponse:

    input = {
        "task": request.task,
        "model_name": request.model_name,
        "generated_yaml": "",
        "yaml_path": "",
        "attempts": 0,
        "feedback": "",
        "consistency": ""
    }

    try:
        
        final_state = agent_app.invoke(input)

        return AgentResponse(
            generated_yaml=final_state.get("generated_yaml", ""),
            yaml_path=final_state.get("yaml_path", ""),
            attempts=final_state.get("attempts", 0),
            status=final_state.get("consistency", "")
        )
    
    except Exception as e:
        return AgentResponse(
            generated_yaml="",
            yaml_path="",
            attempts=0,
            status=f"FAILED: {str(e)}"
        )