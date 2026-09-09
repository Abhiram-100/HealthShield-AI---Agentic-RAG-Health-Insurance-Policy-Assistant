from langchain.agents import create_agent
from policy_assistant import config
from policy_assistant.logger import get_logger
logger = get_logger(__name__)

def create_health_agent(llm, tools):
    logger.info(f"Creating Health policy agent with {len(tools)} tools")
    agent = create_agent(model=llm,
                        tools=tools,
                        system_prompt= config.SYSTEM_PROMPT)
    logger.info(f"Created Health policy agent.")
    return agent