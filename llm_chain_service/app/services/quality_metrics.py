import json
import logging
from typing import Optional

from app.core.config import settings
from app.services.llm_client import OpenRouterClient

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator. Assess the assistant's answer according to these rules:

- accuracy (0-5):
  0 = completely fabricated or contradicts known facts
  1 = mostly wrong, only a few correct elements
  2 = partially correct but with significant errors
  3 = correct in essence but missing details or minor inaccuracies
  4 = almost entirely correct, maybe one small omission
  5 = perfectly correct, no errors or omissions

- relevance (0-5):
  0 = completely off-topic, does not address the question
  1 = mostly irrelevant, only a tiny part relates
  2 = partially relevant, several off-topic parts
  3 = relevant but not fully focused on the question
  4 = mostly relevant, minor digressions
  5 = perfectly relevant, directly answers the question

- completeness (0-5):
  0 = no answer or almost empty
  1 = missing most key points
  2 = covers some points, missing important aspects
  3 = covers main points but lacks details
  4 = nearly complete, only small omissions
  5 = fully complete, covers all aspects of the question

- conciseness (0-5):
  0 = extremely verbose, no useful information density
  1 = very wordy, many unnecessary repetitions
  2 = somewhat concise but still includes filler
  3 = adequately concise, minor filler
  4 = concise, most sentences add value
  5 = perfectly concise, every sentence is essential

- coherence (0-5):
  0 = completely disjointed, impossible to follow
  1 = chaotic, no logical flow
  2 = some logical connections but often breaks
  3 = mostly coherent, minor jumps
  4 = clear and logical, easy to follow
  5 = perfectly structured, seamless flow

- style (0-5):
  0 = unreadable, chaotic, no structure
  1 = poor grammar, hard to follow
  2 = understandable but messy
  3 = acceptable structure, some issues
  4 = clear, well-organized, appropriate tone
  5 = excellent, perfect prose, natural flow

Return ONLY a valid JSON object with keys "accuracy", "relevance", "completeness", "conciseness", "coherence", "style".
Do not output any other text or explanation."""


async def evaluate_answer(question: str, answer: str, model_step: int = 5) -> Optional[dict]:
    if model_step == 5:
        model = settings.model_step5
    elif model_step == 1:
        model = settings.model_step1
    else:
        model = settings.model_step4

    client = OpenRouterClient(settings=settings)
    user_prompt = f"""User question: {question}

Assistant answer: {answer}

Provide evaluation JSON."""
    full_prompt = f"{JUDGE_SYSTEM_PROMPT}\n\n{user_prompt}"

    text, usage = client.generate_with_usage(prompt=full_prompt, model=model)
    if not text:
        logger.error("Judge LLM returned empty response")
        return None

    content = text.strip()
    logger.info(f"Judge response (raw): {content[:500]}")

    if content.startswith("```json"):
        content = content[7:]
    elif content.startswith("```"):
        content = content[3:]
    if content.endswith("```"):
        content = content[:-3]
    content = content.strip()

    if not content:
        logger.error("Empty response after cleaning")
        return None

    start = content.find('{')
    end = content.rfind('}')
    if start == -1 or end == -1 or end <= start:
        logger.error(f"No JSON object found in: {content[:200]}")
        return None
    json_str = content[start:end + 1]

    try:
        scores = json.loads(json_str)
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error: {e}, string: {json_str}")
        return None

    required = {"accuracy", "relevance", "completeness", "conciseness", "coherence", "style"}
    if not required.issubset(scores):
        missing = required - set(scores.keys())
        logger.error(f"Missing keys in JSON: {missing}")
        return None

    for k in required:
        if not isinstance(scores[k], (int, float)) or not (0 <= scores[k] <= 5):
            logger.error(f"Invalid value for {k}: {scores[k]}")
            return None

    return {k: max(0, min(5, int(round(float(scores[k]))))) for k in required}
