from langchain_google_genai import ChatGoogleGenerativeAI
from policy_assistant import config
from policy_assistant.logger import get_logger
from policy_assistant.pipeline import ask, build_health_assistant
from policy_assistant.vector_store import get_retriever, load_vector_store
from langsmith import Client
logger = get_logger(__name__)

from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT, RAG_GROUNDEDNESS_PROMPT

DATASET_NAME = "health-policy-qna"

TEST_CASES = [
    {
        "question": "How does the policy define \"Any one illness,\" and what is the specified timeframe for a relapse to be considered part of the same illness?",
        "answer": "\"Any one illness\" means a continuous period of illness and includes a relapse within 45 days from the date of the last consultation with the Hospital/Nursing Home where treatment was taken. (Section A.1.1, Def. 2, Page 2)"
    },
    {
        "question": "What are the age eligibility criteria for dependent children, and under what condition does a child above 18 become ineligible upon renewal?",
        "answer": "Dependent children must be between 90 days and 25 years of age. A child above 18 years of age becomes ineligible for coverage under subsequent renewals if they become financially independent. (Section A.1.2, Def. 13, Page 9)"
    },
    {
        "question": "What is the duration of the Moratorium Period required before a policy/claim becomes non-contestable on grounds of non-disclosure or misrepresentation?",
        "answer": "60 continuous months of coverage (including portability and migration). After completion of this period, no policy or claim can be contested except on grounds of established fraud. (Section D.1.6, Page 34)"
    },
    {
        "question": "What are the standard default durations (in days) for Pre-Hospitalization and Post-Hospitalization coverage under the Base Coverages?",
        "answer": "Pre-hospitalization expenses are covered up to 60 days immediately prior to the date of admission, and Post-hospitalization expenses are covered up to 180 days immediately following discharge. (Section B.1.5 & B.1.6, Pages 12–13)"
    },
    {
        "question": "What two specific circumstances qualify a patient's home treatment for Domiciliary Hospitalization coverage?",
        "answer": "(1) The condition of the patient is such that he/she cannot be removed or admitted to a hospital, or (2) the treatment is taken at home due to non-availability of a room in a hospital. (Section B.1.3, Page 12)"
    },
    {
        "question": "What three specific categories of expenses are explicitly excluded under the Organ Donor Expenses cover?",
        "answer": "(1) The organ donor’s Pre-Hospitalization and Post-Hospitalization Expenses, (2) Expenses related to organ transportation or preservation, and (3) Any other Medical Expenses or Hospitalization consequent to organ harvesting. (Section B.1.7.c, Page 13)"
    },
    {
        "question": "What is the maximum Preventive Health Check-Up benefit limit for an Individual Policy with a Base Sum Insured of INR 10 Lakhs?",
        "answer": "Rs. 2,000 per Policy Year. (Section B.3 / Annexure C, Page 28, Page 51)"
    },
    {
        "question": "By what percentage does the Plus Benefit increase the Base Sum Insured annually upon renewal, and what is its maximum accumulation cap?",
        "answer": "Plus Benefit adds a sum equal to 50% of the Base Sum Insured on renewal annually, up to a maximum accumulation limit of 100% of the Base Sum Insured. (Section B.2.4, Page 16)"
    },
    {
        "question": "What are the minimum continuous hospitalization hours required and the daily payout cap for availing the \"Daily Cash for Shared Room\" benefit?",
        "answer": "Hospitalization must be in shared accommodation at a Network Provider Hospital exceeding 48 consecutive hours. Benefit is paid for each completed 24 hours (excluding ICU stay) at Rs. 800 per day up to a maximum of Rs. 4,800 per policy year in Optima Secure (or Rs. 1,000 per day up to Rs. 6,000 in Optima Super Secure). (Section B.2.2, Page 15 / Annexure C, Page 51)"
    },
    {
        "question": "What per-claim deductible applies to claims under Global Health Cover (Emergency Treatments Only)?",
        "answer": "A Per Claim Deductible of Rs. 10,000 applies separately for each claim (except for E-Opinion for Critical Illness). (Section B.2.9.A.ii, Page 21)"
    },
    {
        "question": "What are the age limits and required continuous policy tenure to exercise the option to waive or reduce the Aggregate Deductible?",
        "answer": "The eldest insured person must be under 50 years of age at policy purchase, complete 5 continuous policy years with aggregate deductible, and be under 61 years of age at the time of availing the waiver option. (Section B.2.7.1, Page 19)"
    },
    {
        "question": "How many critical illnesses are covered under the E-Opinion benefit, and how many times can it be claimed in a policy year under a Family Floater policy?",
        "answer": "51 listed Major Medical Illnesses are covered. Under a Family Floater policy, the benefit can be availed once per policy year for each insured person. (Section B.2.8, Pages 19–20)"
    },
    {
        "question": "What is the standard waiting period duration for Pre-Existing Diseases (PED) under Code – Excl01?",
        "answer": "36 months of continuous coverage after the date of inception of the first policy. (Section C.1.a, Page 28)"
    },
    {
        "question": "What is the standard waiting period duration for listed conditions/surgeries like Cataract, Hernia, or Joint Replacement under Code – Excl02?",
        "answer": "24 months of continuous coverage after the date of inception of the first policy. (Section C.1.b, Page 28)"
    },
    {
        "question": "Below what dioptre threshold are eye correction surgeries for refractive error excluded under Code – Excl15?",
        "answer": "Expenses related to treatment for correction of eyesight due to refractive error of less than 7.5 dioptres are excluded. (Section C.2.l, Page 31)"
    },
    {
        "question": "What BMI threshold and severe co-morbidities allow surgical treatment for obesity under Code – Excl06?",
        "answer": "The member must be 18 years or older with BMI: (A) greater than or equal to 40, or (B) greater than or equal to 35 along with severe co-morbidities (Obesity-related cardiomyopathy, Coronary heart disease, Severe sleep apnoea, or Uncontrolled type 2 diabetes) following failure of less invasive weight loss methods. (Section C.2.c, Page 30)"
    },
    {
        "question": "What are the mandatory intimation timeframes for Emergency Hospitalization versus Planned Hospitalization?",
        "answer": "For Emergency Hospitalization: Within 24 hours from admission or before discharge, whichever is earlier. For Planned Hospitalization: At least 48 hours prior to admission. (Section E.1.1, Page 41)"
    },
    {
        "question": "Within how many days from discharge must a reimbursement claim for Hospitalization expenses be submitted?",
        "answer": "Within 30 days from the date of discharge from the Hospital. (Section E.1.6, Page 43)"
    },
    {
        "question": "What is the exact sequence specified for the utilization of Sum Insured components when a claim is processed?",
        "answer": "The sequence is: (1) Aggregate Deductible -> (2) Base Sum Insured -> (3) Cumulative Bonus / Plus Benefit -> (4) Secure Benefit -> (5) Automatic Restore Benefit. (Section 1.19, Page 39)"
    },
    {
        "question": "What are the room rent cap, ICU cap, Pre-hospitalization days, and Post-hospitalization days applicable specifically under the Optima Lite plan?",
        "answer": "Room Rent: Covered up to 1% of base sum insured per day; ICU: Covered up to 2% of base sum insured per day; Pre-hospitalization: 30 days; Post-hospitalization: 60 days. (Section B.2.13, B.2.14, B.2.15 / Annexure C, Pages 25–27, 50–51)"
    }
]

JUDGE_LLM_MODEL_NAME = "gemini-3.7-flash"

def _get_judge_llm() -> ChatGoogleGenerativeAI:
    """Return a judge model routed through Portkey, same slug as the main app."""

    return ChatGoogleGenerativeAI(
        api_key=config.GOOGLE_API_KEY,
        model=JUDGE_LLM_MODEL_NAME,  # or config.LLM_MODEL_NAME
        temperature=0
    )

# if dataset is there reuse it , if not create a new dataset
# question paper
def _ensure_dataset(client: Client):
    """Create the LangSmith dataset if it doesn't exist yet, and upload the test cases."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        logger.info("Dataset '%s' already exists, reusing it", DATASET_NAME)
        return client.read_dataset(dataset_name=DATASET_NAME)

    logger.info("Creating dataset '%s' with %d example(s)",
                DATASET_NAME, len(TEST_CASES))
    dataset = client.create_dataset(dataset_name=DATASET_NAME)
    client.create_examples(
        dataset_id=dataset.id,
        examples=[
            {"inputs": {"question": case["question"]},
             "outputs": {"answer": case["answer"]}}
            for case in TEST_CASES
        ],
    )
    return dataset

def run_evals():
    """Upload the dataset (if needed)
    and run the correctness evaluation."""
    client = Client()
    dataset = _ensure_dataset(client)

    # Built once and reused for every test case, instead of rebuilding
    # the whole agent (and reconnecting to Qdrant) 10 times over.
    agent = build_health_assistant()
    retriever = get_retriever(load_vector_store())

    def target(inputs: dict) -> dict:

        answer = ask(agent, inputs["question"])
        chunks = retriever.invoke(inputs["question"])
        context = "\n\n".join(chunk.page_content for chunk in chunks)
        return {"answer": answer, "context": context}

    correctness_evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        feedback_key="correctness",
        judge=_get_judge_llm(),
    )

    groundedness_judge = create_llm_as_judge(
        prompt=RAG_GROUNDEDNESS_PROMPT,
        feedback_key="groundedness",
        judge=_get_judge_llm(),
    )

    def groundedness_evaluator(outputs: dict, **kwargs) -> dict:
        """Check the answer is supported by the retrieved context, not invented."""
        return groundedness_judge(outputs={"answer": outputs["answer"]},
                                  context=outputs["context"])

    logger.info("Running evaluation against dataset '%s'", DATASET_NAME)
    return client.evaluate(
        target,  # positional
        data=dataset.name,
        evaluators=[correctness_evaluator, groundedness_evaluator],
        experiment_prefix="health policy evaluations",
        description="Health policy assistant correctness + groundedness evaluation",
    )