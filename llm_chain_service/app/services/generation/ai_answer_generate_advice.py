from langchain_core.messages import HumanMessage, SystemMessage

from app.services.generation.build_chat import timed_safe_ainvoke
from app.services.generation.classify_question import detect_question_language

import logging
logger = logging.getLogger(__name__)


async def generate_answer_text(question_title: str, question_text: str) -> tuple[str, dict[str, float]]:
    latency: dict[str, float] = {}

    detected_language = detect_question_language(question_title, question_text)

    step1_system = SystemMessage(
        content="You are a neutral expert. Answer only based on reliable knowledge."
    )
    step1_user = HumanMessage(
        content=f"""User's question:
Title: {question_title}
Text: {question_text}

List the key facts that will help give advice (e.g., known options, pros/cons, selection criteria).
You MUST write these facts in {detected_language}.
If you are unsure about any fact, be sure to indicate that (e.g., "I am not sure about ...").
Do not invent sources or make up information.

Format the output as:
Facts:
- fact 1
- fact 2
..."""
    )
    step1_messages = [step1_system, step1_user]
    msg1, latency["step2_sec"] = await timed_safe_ainvoke(
        2, step1_messages, max_tokens=4096, temperature=0.3
    )
    step1_result = msg1.content
    logger.info("[Advice] Step 1 (Facts) \n%s\n", step1_result)

    step2_system = SystemMessage(
        content="You are an experienced consultant on a forum. Give balanced advice, list options, their pros and cons. Tone - neutral, friendly. Do not use Markdown; you may list options using commas or paragraphs instead."
    )
    step2_user = HumanMessage(
        content=f"""Based on the facts, write an advice answer to the question:
{question_title} - {question_text}

Facts:
{step1_result}

Rules:
- You MUST answer in {detected_language}.
- Tone - neutral, friendly, without pressure.
- Suggest several options (if appropriate), indicate their pros and cons.
- Do not use Markdown (**bold**, *italic*, bullet or numbered lists, headings, code blocks).
- Instead of lists, write in connected text, enumerating options with commas or separating them with paragraphs.
- If there are not enough facts, say in {detected_language}: "I don't have enough information to give accurate advice. Please clarify."
- Do not add made-up information.
- FORBIDDEN to add any filler phrases (e.g., "if you need more info, let me know", "always happy to help", "if something is unclear just ask", "hope this helps", "good luck", or any polite closing phrases). The answer must end with the last useful sentence on the topic.

Answer:"""
    )
    step2_messages = [step2_system, step2_user]
    msg2, latency["step3_sec"] = await timed_safe_ainvoke(
        3, step2_messages, max_tokens=4096, temperature=0.3
    )
    step2_result = msg2.content
    logger.info("[Advice] Step 2 (Draft)\n%s\n", step2_result)

    step3_system = SystemMessage(
        content="You are an editor. Return only the corrected answer text, without comments, explanations, lists, numbering, or headings."
    )
    step3_user = HumanMessage(
        content=f"""Check and correct the answer.

Original question: {question_title} - {question_text}
Reliable facts: {step1_result}
Generated answer: {step2_result}

Task:
1. Ensure the answer is written in {detected_language}. If not, rewrite it completely in that language.
2. Remove any fabrications beyond the facts.
3. Remove Markdown elements (** , *, `, #, -, as well as any numbered or bullet lists). Leave only plain text, separated into paragraphs (\\n\\n).
4. If options are listed, they must be in connected text (e.g., "The first option is ..., the second is ..."), not as a list.
5. REMOVE ANY FILLER PHRASES: "if you need more information, feel free to ask", "always happy to help", "if something is unclear just ask", "hope this helps", "good luck", "best regards", "success", "feel free to reach out", "all the best", and any other polite closing phrases. The answer must end with the last useful sentence on the topic.
6. If everything is fine, return the answer unchanged.

Return only the corrected answer. No explanations."""
    )
    step3_messages = [step3_system, step3_user]
    msg3, latency["step4_sec"] = await timed_safe_ainvoke(
        4, step3_messages, max_tokens=4096, temperature=0.3
    )
    step3_result = msg3.content
    logger.info("[Advice] Step 3 (Final answer)\n%s\n", step3_result)

    return step3_result, latency
