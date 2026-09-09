from langchain.tools import tool
from policy_assistant.logger import get_logger
logger = get_logger(__name__)

def create_search_tool(retriever):

    @tool
    def search_health_policy(question: str) -> str:
        """Searches the health insurance policy document for relevant details and rules regarding terms, limits, and coverage."""
        logger.info("Searching health insurance policy details for '%s'", question)
        matching_chunks = retriever.invoke(question)
        logger.info(f"Successfully found {len(matching_chunks)} matching chunks for '{question}'")
        return "\n\n".join(chunk.page_content for chunk in matching_chunks)
    return search_health_policy