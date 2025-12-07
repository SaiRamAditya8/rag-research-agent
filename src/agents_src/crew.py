from crewai import Crew, Process

from src.agents_src.agents.check_intent_agent import intent_agent
from src.agents_src.tasks.check_intent_task import intent_task
from src.agents_src.agents.question_answer_agent import qa_agent
from src.agents_src.tasks.question_answer_task import qa_task


qa_crew = Crew(
    agents=[intent_agent,qa_agent],
    tasks=[intent_task,qa_task],
    process=Process.sequential,
    verbose=True,
)
