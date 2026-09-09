from policy_assistant import config
from policy_assistant.agent import create_health_agent
from policy_assistant.docs_loader import load_documents
from policy_assistant.llm import get_llm
from policy_assistant.chunking import chunking
from policy_assistant.tools import create_search_tool
from policy_assistant.guardrails import DECLINE_MESSAGE, check_input, check_output

from policy_assistant.vector_store import (
    build_vector_store,
    load_vector_store,
    get_retriever,
    vector_store_exists,
)
from policy_assistant.logger import get_logger
logger = get_logger(__name__)

def build_vector_store_for_document(file_path: str = config.DATA_FILE_PATH):
    if vector_store_exists():
        print("Found an existing Qdrant Collection. Connection...")
        logger.info(f"Found an existing Qdrant Collection.")
        return load_vector_store()

    print("No existing Qdrant Collection found. Building vector store...")
    logger.info(f"No existing Qdrant Collection found. Building one from scratch...")
    documents = load_documents(file_path)
    chunks = chunking(documents)

    vector_store = build_vector_store(chunks)
    return vector_store



def build_health_assistant(file_path: str = config.DATA_FILE_PATH):
    logger.info(f"Building Health Policy Assistant...")
    config.check_api_key()

    vector_store = build_vector_store_for_document(file_path)
    retriever = get_retriever(vector_store)
    search_tool = create_search_tool(retriever)

    llm = get_llm()
    agent = create_health_agent(llm, [search_tool])
    logger.info(f"Health Policy Assistant is ready to take questions!")
    return agent

def ask(agent, question: str) -> str:
    logger.info(f"User question: {question}")

    # 1. Input Guardrail
    input_is_safe, _ = check_input(question)
    if not input_is_safe:
        return DECLINE_MESSAGE

    # 2. Agent Execution
    response = agent.invoke({"messages": [{"role": "user", "content": question}]})
    content = response["messages"][-1].content

    # 3. Extract Text Answer
    if isinstance(content, str):
        answer = content.strip()
    elif isinstance(content, list):
        text_parts = []
        for block in content:
            if isinstance(block, str):
                text_parts.append(block)
            elif isinstance(block, dict) and "text" in block:
                text_parts.append(block["text"])
            elif hasattr(block, "text"):
                text_parts.append(block.text)
        answer = "".join(text_parts).strip()
    else:
        answer = str(content).strip()

    logger.info(f"Answer: {answer}")

    # 4. Output Guardrail
    output_is_safe, _ = check_output(answer)
    if not output_is_safe:
        return DECLINE_MESSAGE

    return answer
