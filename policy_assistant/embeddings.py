from langchain_community.embeddings import JinaEmbeddings
from policy_assistant import config
from policy_assistant.logger import get_logger

logger = get_logger(__name__)

def get_embeddings():
    logger.info("Loading Jina embeddings model '%s", config.EMBEDDING_MODEL_NAME)
    return JinaEmbeddings(model_name=config.EMBEDDING_MODEL_NAME)