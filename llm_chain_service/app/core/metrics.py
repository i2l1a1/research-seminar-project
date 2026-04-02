from prometheus_client import Counter, Histogram


# Общее число запросов к цепочке (успешные/неуспешные считаются одинаково)
llm_chain_requests_total = Counter(
    "llm_chain_requests_total",
    "Total number of LLM chain requests",
)

# Общее время выполнения цепочки
llm_chain_latency_seconds = Histogram(
    "llm_chain_latency_seconds",
    "LLM chain latency in seconds",
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

# Время выполнения каждого этапа
llm_chain_stage_latency_seconds = Histogram(
    "llm_chain_stage_latency_seconds",
    "LLM stage latency in seconds",
    ["stage"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 30, 60),
)

# Оценка стоимости
llm_chain_estimated_cost = Counter(
    "llm_chain_estimated_cost",
    "Estimated LLM chain cost (currency units depend on provider metadata)",
)

