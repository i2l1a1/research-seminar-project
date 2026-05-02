from langchain_core.messages import HumanMessage, SystemMessage

from app.services.generation.build_chat import timed_safe_ainvoke
from app.services.generation.classify_question import detect_question_language


async def generate_answer_text(question_title: str, question_text: str) -> tuple[str, dict[str, float]]:
    latency: dict[str, float] = {}

    detected_language = detect_question_language(question_title, question_text)

    step1_system = SystemMessage(
        content="You are a creative expert. You help generate ideas, scenarios, names, design."
    )
    step1_user = HumanMessage(
        content=f"""User's question:
Title: {question_title}
Text: {question_text}

Describe the creative task: what needs to be created (ideas, text, name, scenario, design), what constraints exist (style, topic, length), and what goal the user wants to achieve.
If the question is not creative but falls into this category, still treat it as creative.
You MUST write your analysis in {detected_language}.

Format the output as:
Language: <language>
Creative task: <description>
Constraints: <if any>
Goal: <what the user wants>"""
    )
    step1_messages = [step1_system, step1_user]
    msg1, latency["step2_sec"] = await timed_safe_ainvoke(
        2, step1_messages, max_tokens=4096, temperature=0.8
    )
    step1_result = msg1.content
    print("=== [Creative] Step 1 (Creative task analysis) ===\n", step1_result, "\n")

    step2_system = SystemMessage(
        content="You are a creative consultant. Generate ideas, scenarios, names, design. Tone - inspiring, friendly. Offer several options (2-5) if appropriate. Do not use Markdown; list options in connected text or separate paragraphs."
    )
    step2_user = HumanMessage(
        content=f"""Based on the analysis, write a creative answer to the question:
{question_title} - {question_text}

Analysis:
{step1_result}

Rules:
- You MUST answer in {detected_language}.
- Tone - inspiring, encouraging experimentation.
- Offer several options (2 to 5) if appropriate.
- Do not use Markdown (**bold**, *italic*, bullet or numbered lists, headings, code blocks).
- List options using commas or separate paragraphs, but without Markdown.
- You may use examples and analogies.
- If the user asks for something impossible or unsafe, politely refuse and explain why.
- Hallucinations in creative tasks are allowed but not recommended.
- FORBIDDEN to add any filler phrases (e.g., "if you need more info, let me know", "always happy to help", "if something is unclear just ask", "hope this helps", "good luck", or any polite closing phrases). The answer must end with the last useful sentence on the topic.

Answer:"""
    )
    step2_messages = [step2_system, step2_user]
    msg2, latency["step3_sec"] = await timed_safe_ainvoke(
        3, step2_messages, max_tokens=4096, temperature=0.8
    )
    step2_result = msg2.content
    print("=== [Creative] Step 2 (Draft) ===\n", step2_result, "\n")

    step3_system = SystemMessage(
        content="You are an editor. Return only the corrected answer text, without comments, explanations, lists, numbering, or headings."
    )
    step3_user = HumanMessage(
        content=f"""Check and correct the answer.

Original question: {question_title} - {question_text}
Analysis: {step1_result}
Generated answer: {step2_result}

Task:
1. Ensure the answer is written in {detected_language}. If not, rewrite it completely in that language.
2. Remove any Markdown elements (** , *, `, #, -, as well as any numbered or bullet lists). Leave only plain text.
3. If options are listed as a list, rewrite them as connected text or separate paragraphs.
4. Do not remove creative ideas, even if they are unusual.
5. REMOVE ANY FILLER PHRASES: "if you need more information, feel free to ask", "always happy to help", "if something is unclear just ask", "hope this helps", "good luck", "best regards", "success", "feel free to reach out", "all the best", and any other polite closing phrases. The answer must end with the last useful sentence on the topic.
6. If everything is fine, return the answer unchanged.

Return only the corrected answer. No explanations."""
    )
    step3_messages = [step3_system, step3_user]
    msg3, latency["step4_sec"] = await timed_safe_ainvoke(
        4, step3_messages, max_tokens=4096, temperature=0.8
    )
    step3_result = msg3.content
    print("=== [Creative] Step 3 (Final answer) ===\n", step3_result, "\n")

    return step3_result, latency
