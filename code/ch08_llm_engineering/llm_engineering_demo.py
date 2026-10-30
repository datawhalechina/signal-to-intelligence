#!/usr/bin/env python3
"""Configurable LLM engineering utilities using illustrative local data.

No model name, price, latency, or alert threshold in this file represents a
current provider offering. Production callers should inject a versioned model
catalog and thresholds derived from their own SLOs.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional


class DeploymentMode(Enum):
    """Deployment options for an application."""

    API = "api"
    SELF_HOSTED = "self_hosted"
    HYBRID = "hybrid"


@dataclass(frozen=True)
class ModelSpec:
    """Versioned model measurements or scenario assumptions."""

    name: str
    source: str
    input_cost_per_1m_tokens: float
    output_cost_per_1m_tokens: float
    latency_ms: float
    throughput_tokens_per_sec: float
    context_window: int
    supports_finetuning: bool
    deployment_mode: DeploymentMode


ILLUSTRATIVE_MODELS: Dict[str, ModelSpec] = {
    "api-small": ModelSpec(
        name="Illustrative API Small",
        source="teaching assumption",
        input_cost_per_1m_tokens=0.5,
        output_cost_per_1m_tokens=1.5,
        latency_ms=220,
        throughput_tokens_per_sec=80,
        context_window=16_000,
        supports_finetuning=True,
        deployment_mode=DeploymentMode.API,
    ),
    "api-large": ModelSpec(
        name="Illustrative API Large",
        source="teaching assumption",
        input_cost_per_1m_tokens=5.0,
        output_cost_per_1m_tokens=15.0,
        latency_ms=520,
        throughput_tokens_per_sec=25,
        context_window=128_000,
        supports_finetuning=False,
        deployment_mode=DeploymentMode.API,
    ),
    "self-hosted": ModelSpec(
        name="Illustrative Self-hosted",
        source="teaching assumption; infrastructure cost excluded",
        input_cost_per_1m_tokens=0.0,
        output_cost_per_1m_tokens=0.0,
        latency_ms=700,
        throughput_tokens_per_sec=18,
        context_window=32_000,
        supports_finetuning=True,
        deployment_mode=DeploymentMode.SELF_HOSTED,
    ),
}


class ModelSelector:
    """Filter an injected model catalog by measured requirements."""

    def __init__(self, models: Optional[Mapping[str, ModelSpec]] = None):
        self.models = dict(models or ILLUSTRATIVE_MODELS)

    def select_by_criteria(
        self,
        max_latency_ms: float = float("inf"),
        max_cost_per_1m_tokens: float = float("inf"),
        min_context_window: int = 0,
        requires_finetuning: bool = False,
    ) -> List[ModelSpec]:
        """Filter candidates; cost limit applies to both input and output price."""
        candidates = []
        for model in self.models.values():
            cost_within_limit = max(
                model.input_cost_per_1m_tokens,
                model.output_cost_per_1m_tokens,
            ) <= max_cost_per_1m_tokens
            if (
                model.latency_ms <= max_latency_ms
                and cost_within_limit
                and model.context_window >= min_context_window
                and (not requires_finetuning or model.supports_finetuning)
            ):
                candidates.append(model)
        return sorted(candidates, key=lambda item: item.latency_ms)

    def compare_models(self, model_names: List[str]) -> Dict[str, Dict[str, object]]:
        """Return raw, inspectable values instead of presentation strings."""
        return {
            key: vars(self.models[key]).copy()
            for key in model_names
            if key in self.models
        }


@dataclass(frozen=True)
class CostEstimate:
    daily_requests: int
    avg_input_tokens: int
    avg_output_tokens: int
    model_name: str
    daily_cost: float
    monthly_cost: float
    yearly_cost: float
    excludes_infrastructure: bool


class CostAnalyzer:
    """Calculate token charges from an injected catalog."""

    def __init__(self, model_selector: ModelSelector, billing_days_per_month: int = 30):
        if billing_days_per_month <= 0:
            raise ValueError("billing_days_per_month must be positive")
        self.selector = model_selector
        self.billing_days_per_month = billing_days_per_month

    def estimate_cost(
        self,
        model_name: str,
        daily_requests: int,
        avg_input_tokens: int,
        avg_output_tokens: int,
    ) -> CostEstimate:
        """Estimate token charges; self-hosted infrastructure is not inferred."""
        if model_name not in self.selector.models:
            raise ValueError(f"Unknown model: {model_name}")
        if min(daily_requests, avg_input_tokens, avg_output_tokens) < 0:
            raise ValueError("traffic values must be non-negative")

        model = self.selector.models[model_name]
        daily_cost = daily_requests * (
            avg_input_tokens * model.input_cost_per_1m_tokens
            + avg_output_tokens * model.output_cost_per_1m_tokens
        ) / 1_000_000
        return CostEstimate(
            daily_requests=daily_requests,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            model_name=model_name,
            daily_cost=daily_cost,
            monthly_cost=daily_cost * self.billing_days_per_month,
            yearly_cost=daily_cost * 365,
            excludes_infrastructure=model.deployment_mode is not DeploymentMode.API,
        )


class RetryStrategy:
    """Exponential-backoff policy for idempotent, retryable operations."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay_ms: float = 100,
        max_delay_ms: float = 10_000,
        exponential_base: float = 2.0,
    ):
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.exponential_base = exponential_base

    def get_delay_ms(self, attempt: int) -> float:
        if attempt < 0:
            raise ValueError("attempt must be non-negative")
        return min(
            self.initial_delay_ms * self.exponential_base**attempt,
            self.max_delay_ms,
        )

    def should_retry(self, error_code: int) -> bool:
        return error_code in {429, 500, 502, 503, 504}

    def execute_with_retry(self, func, *args, **kwargs):
        """Retry exceptions; caller must ensure the operation is idempotent."""
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as exc:  # demonstration hook for arbitrary callables
                last_error = exc
                if attempt < self.max_retries:
                    time.sleep(self.get_delay_ms(attempt) / 1000)
        raise last_error


@dataclass(frozen=True)
class MetricSnapshot:
    timestamp: float
    requests_per_sec: float
    avg_latency_ms: float
    error_rate: float
    cost_per_hour: float
    tokens_per_request: float


@dataclass(frozen=True)
class AlertThresholds:
    """Application-specific thresholds, injected rather than universal."""

    max_error_rate: float = 0.05
    max_avg_latency_ms: float = 1_000
    max_requests_per_sec: float = 100


class MetricsCollector:
    def __init__(self, thresholds: Optional[AlertThresholds] = None):
        self.thresholds = thresholds or AlertThresholds()
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0.0
        self.total_tokens = 0
        self.start_time = time.time()

    def record_request(self, latency_ms: float, tokens: int, error: bool = False):
        if latency_ms < 0 or tokens < 0:
            raise ValueError("latency and tokens must be non-negative")
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens
        self.error_count += int(error)

    def get_current_metrics(self) -> MetricSnapshot:
        elapsed_sec = time.time() - self.start_time
        count = self.request_count
        return MetricSnapshot(
            timestamp=time.time(),
            requests_per_sec=count / elapsed_sec if elapsed_sec > 0 else 0.0,
            avg_latency_ms=self.total_latency_ms / count if count else 0.0,
            error_rate=self.error_count / count if count else 0.0,
            cost_per_hour=0.0,
            tokens_per_request=self.total_tokens / count if count else 0.0,
        )

    def check_alerts(self, metrics: MetricSnapshot) -> List[str]:
        alerts = []
        if metrics.error_rate > self.thresholds.max_error_rate:
            alerts.append(f"HIGH_ERROR_RATE: {metrics.error_rate:.1%}")
        if metrics.avg_latency_ms > self.thresholds.max_avg_latency_ms:
            alerts.append(f"HIGH_LATENCY: {metrics.avg_latency_ms:.0f}ms")
        if metrics.requests_per_sec > self.thresholds.max_requests_per_sec:
            alerts.append(f"HIGH_THROUGHPUT: {metrics.requests_per_sec:.1f} req/s")
        return alerts


def main():
    selector = ModelSelector()
    print("Illustrative model catalog (replace before production):")
    for key, model in selector.models.items():
        print(f"- {key}: {model.name}; source={model.source}")

    estimate = CostAnalyzer(selector).estimate_cost(
        "api-small",
        daily_requests=10_000,
        avg_input_tokens=500,
        avg_output_tokens=200,
    )
    print(f"Illustrative token charge: ${estimate.monthly_cost:.2f}/month")
    print("This excludes retrieval, storage, retries, operations, and failure cost.")


if __name__ == "__main__":
    main()
