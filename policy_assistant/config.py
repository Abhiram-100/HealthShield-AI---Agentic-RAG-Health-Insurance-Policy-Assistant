import os
from dotenv import load_dotenv
load_dotenv()

# ENV Secret Keys
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
JINA_API_KEY = os.getenv("JINA_API_KEY")

#Tracing
LANGSMITH_TRACING=os.getenv("LANGSMITH_TRACING")
LANGSMITH_ENDPOINT=os.getenv("LANGSMITH_ENDPOINT")
LANGSMITH_API_KEY=os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT=os.getenv("LANGSMITH_PROJECT")


#Vector Store configs
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "policy_cluster")

# Defining Path for DATA
DATA_FILE_PATH = os.path.join("data", "policy_file.pdf")
#loader = PyPDFLoader(DATA_FILE_PATH)
#documents = loader.load()
#print(f"")

#LLM and EMBEDDING Model Names
LLM_MODEL_NAME = "gemini-3.5-flash-lite"
EMBEDDING_MODEL_NAME = "jina-embeddings-v5-omni-small"
GUARD_MODEL_NAME = "openai/gpt-oss-safeguard-20b"

#CHUNKING Configs
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

#Retrieval results
TOP_K_RESULTS = 3

#system instructions
SYSTEM_PROMPT = (
    "You are a friendly health insurance policy assistant. Always use the search_health_policy tool to look up "
    "facts before answering. If the answer isn't in the search results, just say you don't know "
    "instead of guessing."
)

#API Key Checking
def check_api_key() -> None:
    """To indicate early if the API key is missing."""
    if not GOOGLE_API_KEY:
        raise ValueError("Missing Google LLM API key.")
    if not JINA_API_KEY:
        raise ValueError("Missing Jina Embedding API key.")