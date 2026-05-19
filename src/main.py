from fastapi import FastAPI
from pydantic import BaseModel
from agent import app as agent_app

app = FastAPI() 

# Request Model
class AgentRequest(BaseModel):
    task: str

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
        "generated_yaml": "",
        "yaml_path": "",
        "attempts": 0,
        "consistency": ""
    }

    try:
        result = None
        for output in agent_app.stream(input):
            result = output
        
        final_state = result[list(result.keys())[-1]]

        return AgentResponse(
            generated_yaml=final_state.get("generated_yaml", ""),
            yaml_path=final_state.get("yaml_path", ""),
            attempts=final_state.get("attempts", 0),
            status=final_state.get("status", "")
        )
    
    except Exception as e:
        return AgentResponse(
            generated_yaml="",
            yaml_path="",
            attempts=0,
            status=f"FAILED: {str(e)}"
        )