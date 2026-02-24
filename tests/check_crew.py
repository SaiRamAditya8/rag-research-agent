"""
Quick smoke test: kicks off the QA crew with a sample query and prints the result.
Run from the project root:
    python tests/check_crew.py
"""
from pprint import pprint

from src.agents_src.crew import qa_crew

input_data = {
    "user_query": "Hi there!",
    "chat_history": {}
}

result = qa_crew.kickoff(input_data)

result_dict = result.to_dict()

pprint(result_dict)
