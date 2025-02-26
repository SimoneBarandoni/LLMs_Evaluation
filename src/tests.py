import requests

url = "http://localhost:8000/evaluate"
data = [{
    "user_request": "Translate 'Hello' to French",
    "model_response": "Bonjour",
    "ground_truth": "Bonjour",
    "task_type": "deterministic"
},
{
    "user_request": "Translate 'Hello' to French",
    "model_response": "Translate 'Hello' to French",
    "ground_truth": "Bonjour",
    "task_type": "deterministic"
},
{
    "user_request": "Write the following words in a JSON list: apple, banana, orange. Return only the list.",
    "model_response": "JSON list: {'apple', 'banana', 'orange'}",
    "ground_truth": "['apple', 'banana', 'orange']",
    "task_type": "deterministic"
},
{
    "user_request": "Translate 'Hello' to French",
    "model_response": "I can't answer that",
    "ground_truth": "Bonjour",
    "task_type": "deterministic"
},
{
    "user_request": "Translate 'Hello' to French",
    "model_response": "I hate french, idiot",
    "ground_truth": "Bonjour",
    "task_type": "deterministic"
},
{
    "user_request": "Translate 'Hello' to French",
    "model_response": "ca va?",
    "ground_truth": "Bonjour",
    "task_type": "deterministic"
},
{
    "user_request": "What is the product of 10 and 5?",
    "model_response": "50",
    "task_type": "deterministic"
},
{
    "user_request": "Explain why the sky is blue in scientific terms.",
    "model_response": "The sky appears blue due to Rayleigh scattering of sunlight by air molecules shorter wavelength.",
    "task_type": "open-ended"
},
]

for i in data:
    response = requests.post(url, json=i)
    print(response.json())