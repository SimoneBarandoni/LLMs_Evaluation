import json
from typing import Optional, Union, Literal
from enum import Enum
import time
from openai import OpenAI
from bert_score import score
from utils import prompt, error_classes, judge_prompt

client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='qwen2.5'
)

client_gpt = OpenAI(
    api_key=''
)

def bertscore(label: list[str], predicted: list[str]) -> float:
    _, _, F1 = score(predicted, label, model_type="microsoft/deberta-xlarge-mnli", 
                     rescale_with_baseline=False, lang="en", verbose=False)
    print(F1.item())
    return round(F1.item(), 2)

def llm_as_a_judge(user_request: str, model_response: str) -> float:
    completion = client_gpt.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": judge_prompt
            },
            {
                "role": "user",
                "content": json.dumps({"user_request": user_request, "model_response": model_response}),
            }   
        ],
        response_format={"type": "json_object"}
    )
    # the model will return a json object with a single field named "evaluation" containing the rating
    content = completion.choices[0].message.content
    response_json = json.loads(content)
    return response_json["evaluation"]

# function that takes a user request and a model response and performs a classification task to determine if the model response is correct or incorrect
def classify(user_request: str, model_response: str) -> int:
    completion = client.chat.completions.create(
        model="qwen2.5",
        messages=[
            {
                "role": "system",
                # the prompt requests the model to read the user request and model response and classify the model response as correct or incorrect
                "content": prompt,
            },
            {
                "role": "user",
                # the second message is a json object that contains the user request and model response that the model should classify
                "content": json.dumps({"user_request": user_request, "model_response": model_response}),
            },
        ],
        response_format={"type": "json_object"}
    )
    
    content = completion.choices[0].message.content
    if content is None:
        raise ValueError("Received empty response from model")
    
    response_json = json.loads(content)
    return response_json["classification"]

class TaskType(Enum):
    DETERMINISTIC = "deterministic"
    OPEN_ENDED = "open-ended"

def evaluate_response(
    user_request: str, 
    model_response: str, 
    task_type: TaskType,
    # the ground truth is the correct answer to the user request, it can be absent if the dataset is not labeled
    ground_truth: Optional[str] = None
) -> Union[str, float]:
    
    classification = classify(user_request, model_response)
    # If there's an error, return the error class
    if classification != 1:
        return error_classes[classification]

    # if ground truth is not None, perform the regular evaluation
    if ground_truth is not None:
        # If no error, evaluate based on task type
        if task_type == TaskType.DETERMINISTIC:
            return 1.0 if model_response.strip() == ground_truth.strip() else 0.0
        else:  # TaskType.OPEN_ENDED
            return bertscore([ground_truth], [model_response])
    else:
        return llm_as_a_judge(user_request, model_response)




# CLEARE: CLassification-Enhanced Automatic Response Evaluation