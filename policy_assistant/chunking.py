from langchain_text_splitters import RecursiveCharacterTextSplitter
from policy_assistant import config
from policy_assistant.logger import get_logger
logger = get_logger(__name__)

def chunking(documents):
    """Splits documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = config.CHUNK_SIZE,
        chunk_overlap= config.CHUNK_OVERLAP,
    )
    chunks =  text_splitter.split_documents(documents)
    logger.info(f"Chunking {len(chunks)} documents")
    return chunks