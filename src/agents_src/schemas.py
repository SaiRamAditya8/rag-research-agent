from pydantic import BaseModel, Field
from typing import List, Optional


class IntentOutput(BaseModel):
    fetch: bool
    use_rag: bool
    queries: List[str]
    categories: List[str]
    request: str
    paper_filter: List[str] = Field(default_factory=list)


class AnswerStructure(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
    tool_used: Optional[str] = None
    rationale: Optional[str] = None
