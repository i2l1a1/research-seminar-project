import json
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.generation.build_chat import timed_safe_ainvoke

from langdetect import detect

import json
from langchain_core.messages import HumanMessage, SystemMessage
from app.services.generation.build_chat import timed_safe_ainvoke


async def classify_question(question_title: str, question_text: str) -> tuple[dict, float]:
    system = SystemMessage(
        content="""You are a question classifier. Your task is to determine the type of question and assess your confidence.

Question types:
1. technical - questions with an unambiguous factual answer (syntax, dates, definitions, facts).
2. advice - questions about choice, advice, comparison of options, where there is no single correct answer.
3. tutorial - questions requiring explanation, step-by-step guidance, learning.
4. current - questions about events, news, the future, requiring up-to-date data (the model may not know).
5. creative - questions about generating ideas, scenarios, texts, names, design.

Return the answer strictly in JSON format:
{"type": "one_of_the_five", "confidence": 0.95}

Do not add any other explanations."""
    )

    user = HumanMessage(
        content=f"""Title: {question_title}
Question text:
{question_text}

Determine the question type and confidence in JSON format."""
    )

    messages = [system, user]
    response_msg, classify_sec = await timed_safe_ainvoke(1, messages, max_tokens=512, temperature=0.0)
    response = response_msg.content

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

    return result, classify_sec


def detect_question_language(question_title: str, question_text: str) -> str:
    full_text = f"{question_title} {question_text}"
    try:
        lang_code = detect(full_text)
    except:
        lang_code = "en"

    lang_map = {
        "en": "English",
        "de": "German",
        "fr": "French",
        "es": "Spanish",
        "it": "Italian",
        "zh-cn": "Chinese",
        "ja": "Japanese",
        "ru": "Russian",
        "af": "Afrikaans",
        "ar": "Arabic",
        "bg": "Bulgarian",
        "bn": "Bengali",
        "ca": "Catalan",
        "cs": "Czech",
        "cy": "Welsh",
        "da": "Danish",
        "el": "Greek",
        "et": "Estonian",
        "fa": "Persian",
        "fi": "Finnish",
        "gu": "Gujarati",
        "he": "Hebrew",
        "hi": "Hindi",
        "hr": "Croatian",
        "hu": "Hungarian",
        "id": "Indonesian",
        "kn": "Kannada",
        "ko": "Korean",
        "lt": "Lithuanian",
        "lv": "Latvian",
        "mk": "Macedonian",
        "ml": "Malayalam",
        "mr": "Marathi",
        "ne": "Nepali",
        "nl": "Dutch",
        "no": "Norwegian",
        "pa": "Punjabi",
        "pl": "Polish",
        "pt": "Portuguese",
        "ro": "Romanian",
        "sk": "Slovak",
        "sl": "Slovenian",
        "so": "Somali",
        "sq": "Albanian",
        "sv": "Swedish",
        "sw": "Swahili",
        "ta": "Tamil",
        "te": "Telugu",
        "th": "Thai",
        "tl": "Tagalog",
        "tr": "Turkish",
        "uk": "Ukrainian",
        "ur": "Urdu",
        "vi": "Vietnamese",
        "zh-tw": "Chinese (Traditional)",
    }
    return lang_map.get(lang_code, "English")
