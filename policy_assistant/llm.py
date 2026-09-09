from langchain_google_genai import ChatGoogleGenerativeAI
from policy_assistant import config
from policy_assistant.logger import get_logger

logger = get_logger(__name__)

def get_llm():
    logger.info("Loading LLM model '%s'", config.LLM_MODEL_NAME)
    return ChatGoogleGenerativeAI(model = config.LLM_MODEL_NAME, temperature = 0)