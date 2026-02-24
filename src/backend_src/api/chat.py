import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from src.backend_src.services.chat import get_answer

logger = logging.getLogger(__name__)

router = APIRouter()

class ChatRequest(BaseModel):
    user_query: str
    session_id: str = "default_session"

@router.post("/chat/answer")
def chat_answer(request: ChatRequest):
    logger.info(f"Received API request: session={request.session_id}, query={request.user_query}")
    try:
        result = get_answer(user_query=request.user_query, session_id=request.session_id)
        logger.info(f"API response: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in chat_answer: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
