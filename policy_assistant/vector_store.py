import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from policy_assistant import config
from policy_assistant.embeddings import get_embeddings
from policy_assistant.logger import get_logger
logger = get_logger(__name__)



def build_vector_store(chunks):
    logger.info(f"Building Qdrant vector store with {len(chunks)} chunks")
    embeddings_model = get_embeddings()
    vector_store = QdrantVectorStore.from_documents(
        chunks,
        embedding = embeddings_model,
        url = config.QDRANT_URL,
        api_key = config.QDRANT_API_KEY,
        collection_name = config.QDRANT_COLLECTION_NAME,
    )
    logger.info(f"Qdrant vector store built in cloud")
    return vector_store


#def load_vector_store():
    embeddings_model = get_embeddings()
    return QdrantVectorStore.from_documents(
        embedding = embeddings_model,
        url = config.QDRANT_URL,
        api_key = config.QDRANT_API_KEY,
        collection_name = config.QDRANT_COLLECTION_NAME
    )

def load_vector_store():
    logger.info(f"Loading Qdrant vector store")
    embeddings_model = get_embeddings()
    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings_model,
        url=config.QDRANT_URL,
        api_key=config.QDRANT_API_KEY,
        collection_name=config.QDRANT_COLLECTION_NAME,
    )


def vector_store_exists() -> bool:
    client = QdrantClient(
        url = config.QDRANT_URL,
        api_key = config.QDRANT_API_KEY,
    )
    return client.collection_exists(config.QDRANT_COLLECTION_NAME)


def get_retriever(vector_store, k: int = config.TOP_K_RESULTS):
    logger.info(f"Retrieving {k} results")
    return vector_store.as_retriever(search_kwargs = {"k": k})