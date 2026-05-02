from langchain_core.messages import HumanMessage, SystemMessage

from app.services.generation.build_chat import timed_safe_ainvoke
from app.services.generation.classify_question import detect_question_language


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

List the key facts and steps needed to explain or perform the action.
If you are unsure about any fact, be sure to indicate that (e.g., "I am not sure about ...").
Do not invent sources or make up information.
You MUST write these facts and steps in {detected_language}.

Format the output as:
Facts and steps:
- fact/step 1
- fact/step 2
..."""
    )
    step1_messages = [step1_system, step1_user]
    msg1, latency["step2_sec"] = await timed_safe_ainvoke(
        2, step1_messages, max_tokens=4096, temperature=0.2
    )
    step1_result = msg1.content
    print("=== [Tutorial] Step 1 (Facts and steps) ===\n", step1_result, "\n")

    step2_system = SystemMessage(
        content="You are a patient teacher on a forum. Explain clearly, step by step, with examples. Tone - friendly, supportive. Do not use Markdown, but you may number steps using words (First step, Second step) or phrases like 'First...', 'Then...', 'After that...'."
    )
    step2_user = HumanMessage(
        content=f"""Based on the facts, write a tutorial answer to the question:
{question_title} - {question_text}

Facts and steps:
{step1_result}

Rules:
- You MUST answer in {detected_language}.
- Tone - friendly, supportive, like a good teacher.
- Break the explanation into logical steps. Number steps using words (not digits or Markdown markers).
- Use examples if they help understanding.
- Do not use Markdown (**bold**, *italic*, bullet or numbered lists, headings, code blocks).
- Split the answer into paragraphs (\\n\\n) between steps.
- If there are not enough facts, say: "I need more information to explain correctly. Please clarify..." (in {detected_language}).
- Do not add made-up information.
- FORBIDDEN to add any filler phrases (e.g., "if you need more info, let me know", "always happy to help", "if something is unclear just ask", "hope this helps", "good luck", or any polite closing phrases). The answer must end with the last useful sentence on the topic.

Answer:"""
    )
    step2_messages = [step2_system, step2_user]
    msg2, latency["step3_sec"] = await timed_safe_ainvoke(
        3, step2_messages, max_tokens=4096, temperature=0.2
    )
    step2_result = msg2.content
    print("=== [Tutorial] Step 2 (Draft) ===\n", step2_result, "\n")

    step3_system = SystemMessage(
        content="You are an editor. Return only the corrected answer text, without comments, explanations, lists, numbering, or headings."
    )
    step3_user = HumanMessage(
        content=f"""Check and correct the answer.

Original question: {question_title} - {question_text}
Reliable facts and steps: {step1_result}
Generated answer: {step2_result}

Task:
1. Ensure the answer is written in {detected_language}. If not, rewrite it completely in that language.
2. Remove any fabrications beyond the facts.
3. Remove Markdown elements (** , *, `, #, -, as well as any numbered or bullet lists). Leave only plain text.
4. Ensure that steps are logically separated (paragraphs or word‑based markers like "First step", "Then"). If steps are listed as a list, rewrite them as connected text.
5. REMOVE ANY FILLER PHRASES: "if you need more information, feel free to ask", "always happy to help", "if something is unclear just ask", "hope this helps", "good luck", "best regards", "success", "feel free to reach out", "all the best", and any other polite closing phrases. The answer must end with the last useful sentence on the topic.
6. If everything is fine, return the answer unchanged.

Return only the corrected answer. No explanations."""
    )
    step3_messages = [step3_system, step3_user]
    msg3, latency["step4_sec"] = await timed_safe_ainvoke(
        4, step3_messages, max_tokens=4096, temperature=0.2
    )
    step3_result = msg3.content
    print("=== [Tutorial] Step 3 (Final answer) ===\n", step3_result, "\n")

    return step3_result, latency
