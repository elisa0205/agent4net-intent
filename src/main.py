from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import app as agent_app

app = FastAPI() 

# Request Model
class AgentRequest(BaseModel):
    task: str
    model_name: str
    temperature: float = 0.7 #default temperature 0.7

# Response Model
class AgentResponse(BaseModel):
    generated_yaml: str
    yaml_path: str
    attempts: int
    feedback: str
    consistency: str


@app.post("/manifest")
async def generate_yaml(request: AgentRequest) -> AgentResponse:

    input = {
        "task": request.task,
        "model_name": request.model_name,
        "generated_yaml": "",
        "yaml_path": "",
        "attempts": 0,
        "feedback": "",
        "consistency": "",
        "temperature": request.temperature
    }

    try:
        
        final_state = agent_app.invoke(input)
        feedback = final_state.get("feedback", "")

        if feedback.startswith("FAILED"):
            raise HTTPException(status_code=400, detail=feedback)
        elif final_state.get("consistency", "") == "INVALID":
            raise HTTPException(status_code=400, detail="Consistency check failed: " + feedback)

        return AgentResponse(
            generated_yaml=final_state.get("generated_yaml", ""),
            yaml_path=final_state.get("yaml_path", ""),
            attempts=final_state.get("attempts", 0),
            feedback=final_state.get("feedback", ""),
            consistency=final_state.get("consistency", "")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))