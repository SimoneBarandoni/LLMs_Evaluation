import json
from utils import inverted_error_classes, prompt

# # open results.json
# # for each example, recover the classification from error_classes and store it in a new field called "classification"
# with open('results.json', 'r') as f:
#     results = json.load(f)
#     for result in results:
#         if isinstance(result['evaluation'], float):
#             result['classification'] = "No error"
#         else:
#             result['classification'] = inverted_error_classes[result['evaluation']]

# # save the results to a new file called results_with_classification.json
# with open('results_with_classification.json', 'w') as f:
#     json.dump(results, f)



with open('src/results_with_classification.json', 'r') as f:
    with open('training_qwen_v3.jsonl', 'w') as output_file:
        results = json.load(f)
        for result in results:
            user_request = result['user_request']
            model_response = result['model_response']
            # the classification can be a number or "No error"
            # if it is "No error", then we should convert it to 1
            # otherwise, it is already a number
            if result['classification'] == "No error":
                classification = 1
            else:
                classification = result['classification']
            message = {
                    "messages": [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps({"user_request": user_request, "model_response": model_response}),},
                        {"role": "assistant", "content": json.dumps({"classification": result['classification']})}
                    ]
                }
            output_file.write(json.dumps(message) + "\n")
       

# # print the time
# start_time = time.time()
# with open('test_examples.json') as f:
#     d = json.load(f)
#     # create a new json file with the results   
#     results = []
#     for example in d['examples']:
#         result = {
#             "user_request": example['user_request'],
#             "model_response": example['model_response'],
#             "ground_truth": example['ground_truth'],
#             "task_type": example['task_type'],
#             "evaluation": evaluate_response(example['user_request'], example['model_response'], example['ground_truth'], example['task_type'])  
#         }
#         results.append(result)

#     with open('results.json', 'w') as f:
#         json.dump(results, f)

# print(f"Time taken: {time.time() - start_time} seconds")