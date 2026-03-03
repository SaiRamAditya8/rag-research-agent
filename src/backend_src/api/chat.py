import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.backend_src.services.chat import get_answer

logger = logging.getLogger(__name__)

router = APIRouter()


class ChatRequest(BaseModel):
    user_query: str
    project_id: str = "default"


@router.post("/chat/answer")
def chat_answer(request: ChatRequest):
    logger.info(f"Received API request: project={request.project_id}, query={request.user_query}")
    try:
        result = get_answer(user_query=request.user_query, project_id=request.project_id)
        logger.info(f"API response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in chat_answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
