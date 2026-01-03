from crewai import Crew, Process

from src.agents_src.agents.check_intent_agent import intent_agent
from src.agents_src.tasks.check_intent_task import intent_task
from src.agents_src.agents.question_answer_agent import qa_agent
from src.agents_src.tasks.question_answer_task import qa_task
from src.agents_src.agents.chitchat_agent import chitchat_agent
from src.agents_src.tasks.chitchat_task import chitchat_task

intent_crew = Crew(
    agents=[intent_agent],
    tasks=[intent_task],
    verbose=True,
)

chitchat_crew = Crew(
    agents=[chitchat_agent],
    tasks=[chitchat_task],
    verbose=True,
)

qa_crew = Crew(
    agents=[qa_agent],
    tasks=[qa_task],
    verbose=True,
)