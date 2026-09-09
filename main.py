from policy_assistant.pipeline import ask, build_health_assistant
from policy_assistant.logger import get_logger
logger = get_logger(__name__)

def main():
    logger.info(f"---CLI Run Started---")
    print("Building Health Assistant...")
    agent = build_health_assistant()
    print("Assistant ready...")

    demo_questions = [
        "What is the difference in the room rent limit between the Optima Select plan and the standard Optima Secure plan?"
    ]

    for question in demo_questions:
        print("="*50)
        print("Question:", question)
        print("-"*50)
        answer = ask(agent, question)
        print("Answer:", answer)
        print("="*50)
        print()

        logger.info(f"---CLI Run Finished---")

if __name__ == "__main__":
    main()