from langchain_community.document_loaders import PyPDFLoader
from policy_assistant import config
from policy_assistant.logger import get_logger

logger = get_logger(__name__)

def load_documents(file_path: str = config.DATA_FILE_PATH):
    """Loads .pdf file and returns"""
    logger.info(f"Loading documents from {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    logger.info("Loaded %d documents", len(documents))
    return documents