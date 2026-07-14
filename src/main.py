from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from agent import app as agent_app
from time import perf_counter

app = FastAPI() 

# Request Model
class AgentRequest(BaseModel):
    task: str
    model_name: str
    temperature: float = 0.7 #default temperature 0.7
    user_override: str | None = None
    manual_feedback: str | None = None

# Response Model
class AgentResponse(BaseModel):
    generated_yaml: str
    yaml_path: str
    attempts: int
    consistency_fails: int
    syntax_fails: int
    k8s_fails: int
    feedback: str
    consistency: str
    token_usage: int
    elapsed_time: float #seconds
    k8s_validator_time: float #seconds
    intervention_required: bool = False
    intervention_message: str = ""

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
        "temperature": request.temperature,
        "token_usage": 0,
        "user_override": request.user_override or "stop",
        "manual_feedback": request.manual_feedback or "",
        "intervention_required": False,
        "intervention_message": "",
        "last_error": "",
        "repeated_error_count": 0,
    }

    try:
        start_time = perf_counter()
        
        final_state = agent_app.invoke(input)

        elapsed_sec = perf_counter() - start_time

        if final_state.get("intervention_required", False):
            raise HTTPException(
                status_code=409,
                detail={
                    "message": final_state.get("intervention_message", "User intervention required."),
                    "generated_yaml": final_state.get("generated_yaml", ""),
                    "yaml_path": final_state.get("yaml_path", ""),
                    "attempts": final_state.get("attempts", 0),
                    "consistency_fails": final_state.get("consistency_fails", 0),
                    "syntax_fails": final_state.get("syntax_fails", 0),
                    "k8s_fails": final_state.get("k8s_fails", 0),
                    "feedback": final_state.get("feedback", ""),
                    "consistency": final_state.get("consistency", ""),
                    "token_usage": final_state.get("token_usage", 0),
                    "elapsed_time": elapsed_sec,
                    "k8s_validator_time": final_state.get("k8s_validator_time", 0.0)
                }
            )

        return AgentResponse(
            generated_yaml=final_state.get("generated_yaml", ""),
            yaml_path=final_state.get("yaml_path", ""),
            attempts=final_state.get("attempts", 0),
            consistency_fails=final_state.get("consistency_fails", 0),
            syntax_fails=final_state.get("syntax_fails", 0),
            k8s_fails=final_state.get("k8s_fails", 0),
            feedback=final_state.get("feedback", ""),
            consistency=final_state.get("consistency", ""),
            token_usage=final_state.get("token_usage", 0),
            elapsed_time=elapsed_sec,
            k8s_validator_time=final_state.get("k8s_validator_time", 0.0),
            intervention_required=False,
            intervention_message=""
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))