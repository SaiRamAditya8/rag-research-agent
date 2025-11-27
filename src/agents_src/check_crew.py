from pprint import pprint

from src.agents_src.crew import qa_crew


# input_data = {
#     "user_query": "Explain about adaptive radiation",
#     "chat_history": "{}"
# }

input_data = {
    "user_query": "Explain difference between Attention and knowledge distillation in the field of Machine Learning",
    "chat_history": {}
}

result = qa_crew.kickoff(input_data)

result_dict = result.to_dict()

pprint(result_dict)
