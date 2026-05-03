from prometheus_client import Counter, Histogram, Summary

# Общее число запросов к цепочке
llm_chain_requests_total = Counter(
    "llm_chain_requests_total",
    "Total number of LLM chain requests",
)

# Общее время выполнения цепочки
llm_chain_latency_seconds = Histogram(
    "llm_chain_latency_seconds",
    "LLM chain latency in seconds",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 70, 100, 120, 140, 160, 180, 200, 210, 220),
)

# Время выполнения каждого этапа - те же бакеты
llm_chain_stage_latency_seconds = Histogram(
    "llm_chain_stage_latency_seconds",
    "LLM stage latency in seconds",
    ["stage"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60, 70, 100, 120, 140, 160, 180, 200, 210, 220),
)

# Оценка стоимости
llm_chain_estimated_cost = Counter(
    "llm_chain_estimated_cost",
    "Estimated LLM chain cost (currency units depend on provider metadata)",
)

# Метрики качества ответа (каждая оценка от 0 до 5)
llm_answer_accuracy = Summary("llm_answer_accuracy", "Accuracy score of answer (0-5)")
llm_answer_relevance = Summary("llm_answer_relevance", "Relevance score of answer (0-5)")
llm_answer_completeness = Summary("llm_answer_completeness", "Completeness score of answer (0-5)")
llm_answer_conciseness = Summary("llm_answer_conciseness", "Conciseness score of answer (0-5)")
llm_answer_coherence = Summary("llm_answer_coherence", "Coherence score of answer (0-5)")
llm_answer_style = Summary("llm_answer_style", "Style score of answer (0-5)")

llm_quality_evaluations_total = Counter(
    "llm_quality_evaluations_total",
    "Total number of quality evaluation attempts",
    ["status"]
)
