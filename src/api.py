from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Union, Literal, Optional
from eval import evaluate_response, TaskType
import uvicorn

app = FastAPI(
    title="CLEARE API",
    description="CLassification-Enhanced Automatic Response Evaluation API",
    version="1.0.0"
)

class EvaluationRequest(BaseModel):
    user_request: str
    model_response: str
    ground_truth: Optional[str] = None
    task_type: Literal["deterministic", "open-ended"]

class EvaluationResponse(BaseModel):
    evaluation: Union[str, float]
    details: str

@app.post("/evaluate", response_model=EvaluationResponse)
async def evaluate(request: EvaluationRequest):
    try:
        # Convert string task_type to TaskType enum
        task_type = TaskType(request.task_type)
        
        # Perform evaluation
        result = evaluate_response(
            user_request = request.user_request,
            model_response = request.model_response,
            ground_truth = request.ground_truth,
            task_type = task_type
        )
        
        # Prepare response details
        details = ""
        if isinstance(result, str):
            details = f"Error detected: {result}"
        else:
            # if the ground truth is present, the output can be due to exact match or BertScore
            if request.ground_truth is not None:
                if task_type == TaskType.DETERMINISTIC:
                    details = "Exact match comparison performed"
                else:   
                    details = "Semantic similarity evaluation performed using BERTScore"
            # if the ground truth is not present, the output can be due to the LLM as a judge
            else:
                details = "LLM as a judge evaluation performed"
        
        return EvaluationResponse(
            evaluation=result,
            details=details
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "name": "CLEARE API",
        "description": "CLassification-Enhanced Automatic Response Evaluation",
        "endpoints": {
            "/evaluate": "POST - Evaluate a model response",
            "/": "GET - This help message"
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000) 