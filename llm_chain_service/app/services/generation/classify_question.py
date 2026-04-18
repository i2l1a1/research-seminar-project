import json
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.generation.build_chat import _build_chat, safe_ainvoke


async def classify_question(question_title: str, question_text: str) -> dict:
    chat = _build_chat(max_tokens=512)

    system = SystemMessage(
        content="""Ты – классификатор вопросов. Твоя задача - определить тип вопроса и оценить уверенность.
        
Типы вопросов:
1. technical - вопросы с однозначным фактическим ответом (синтаксис, даты, определения, факты).
2. advice - вопросы о выборе, советах, сравнении вариантов, где нет единственно верного ответа.
3. tutorial - вопросы, требующие объяснения, пошагового руководства, обучения.
4. current - вопросы о событиях, новостях, будущем, требующие актуальных данных (модель может не знать).
5. creative - вопросы на генерацию идей, сценариев, текстов, названий, дизайна.

Верни ответ строго в формате JSON:
{"type": "один_из_пяти", "confidence": 0.95}

Не добавляй никаких других пояснений."""
    )

    user = HumanMessage(
        content=f"""Заголовок: {question_title}
Текст вопроса:
{question_text}

Определи тип вопроса и уверенность в формате JSON."""
    )

    messages = [system, user]
    response = (await safe_ainvoke(chat, messages)).content

    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()

    try:
        result = json.loads(response)
    except json.JSONDecodeError:
        result = {"type": "advice", "confidence": 0.5}
        for t in ["technical", "advice", "tutorial", "current", "creative"]:
            if t in response.lower():
                result["type"] = t
                break

    valid_types = {"technical", "advice", "tutorial", "current", "creative"}
    if result.get("type") not in valid_types:
        result["type"] = "advice"
    if not isinstance(result.get("confidence"), (int, float)):
        result["confidence"] = 0.5
    result["confidence"] = max(0.0, min(1.0, result["confidence"]))

    return result
