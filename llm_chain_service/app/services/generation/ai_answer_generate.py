from app.services.generation.classify_question import classify_question
from app.services.generation.ai_answer_generate_technical import generate_answer_text as technical_generate
from app.services.generation.ai_answer_generate_advice import generate_answer_text as advice_generate
from app.services.generation.ai_answer_generate_tutorial import generate_answer_text as tutorial_generate
from app.services.generation.ai_answer_generate_current import generate_answer_text as current_generate
from app.services.generation.ai_answer_generate_creative import generate_answer_text as creative_generate


async def generate_answer_text(question_title: str, question_text: str) -> tuple[str, dict[str, float]]:
    classification, step1_sec = await classify_question(question_title, question_text)
    q_type = classification.get("type", "advice")
    confidence = classification.get("confidence", 0.5)

    if confidence < 0.6:
        q_type = "advice"

    if q_type == "technical":
        text, lat = await technical_generate(question_title, question_text)
    elif q_type == "tutorial":
        text, lat = await tutorial_generate(question_title, question_text)
    elif q_type == "current":
        text, lat = await current_generate(question_title, question_text)
    elif q_type == "creative":
        text, lat = await creative_generate(question_title, question_text)
    else:
        text, lat = await advice_generate(question_title, question_text)

    latency_per_step = {
        "step1_classify_sec": step1_sec,
        **lat,
    }
    return text, latency_per_step
