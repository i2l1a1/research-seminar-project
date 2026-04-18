from app.services.generation.classify_question import classify_question
from app.services.generation.ai_answer_generate_technical import generate_answer_text as technical_generate
from app.services.generation.ai_answer_generate_advice import generate_answer_text as advice_generate
from app.services.generation.ai_answer_generate_tutorial import generate_answer_text as tutorial_generate
from app.services.generation.ai_answer_generate_current import generate_answer_text as current_generate
from app.services.generation.ai_answer_generate_creative import generate_answer_text as creative_generate


async def generate_answer_text(question_title: str, question_text: str) -> str:
    classification = await classify_question(question_title, question_text)
    q_type = classification.get("type", "advice")
    confidence = classification.get("confidence", 0.5)

    if confidence < 0.6:
        q_type = "advice"

    if q_type == "technical":
        print(f"q_type: technical")
        return await technical_generate(question_title, question_text)
    elif q_type == "tutorial":
        print(f"q_type: tutorial")
        return await tutorial_generate(question_title, question_text)
    elif q_type == "current":
        print(f"q_type: current")
        return await current_generate(question_title, question_text)
    elif q_type == "creative":
        print(f"q_type: creative")
        return await creative_generate(question_title, question_text)
    else:
        print(f"q_type: advice")
        return await advice_generate(question_title, question_text)
