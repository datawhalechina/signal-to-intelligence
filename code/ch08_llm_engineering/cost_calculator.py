#!/usr/bin/env python3
"""Cost arithmetic with explicit, illustrative price assumptions."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class PriceCard:
    name: str
    input_per_1m: float
    output_per_1m: float
    source: str = "illustrative teaching assumption"


@dataclass
class TrafficProfile:
    daily_requests: int
    avg_input_tokens: int
    avg_output_tokens: int
    full_response_cache_hit_rate: float = 0.0


@dataclass
class CostBreakdown:
    model_name: str
    billable_input_tokens: float
    billable_output_tokens: float
    daily_cost: float
    monthly_cost: float


ILLUSTRATIVE_PRICES = {
    "large": PriceCard("large", input_per_1m=5.0, output_per_1m=15.0),
    "small": PriceCard("small", input_per_1m=0.5, output_per_1m=1.5),
}
# Backward-compatible name; values are not current provider prices.
DEFAULT_PRICES = ILLUSTRATIVE_PRICES


def estimate_single_model_cost(price: PriceCard, traffic: TrafficProfile) -> CostBreakdown:
    """Estimate token charges assuming a hit avoids the full model request."""
    if min(traffic.daily_requests, traffic.avg_input_tokens, traffic.avg_output_tokens) < 0:
        raise ValueError("traffic values must be non-negative")
    if not 0 <= traffic.full_response_cache_hit_rate <= 1:
        raise ValueError("full_response_cache_hit_rate must be in [0, 1]")

    billable_ratio = 1 - traffic.full_response_cache_hit_rate
    input_tokens = traffic.daily_requests * traffic.avg_input_tokens * billable_ratio
    output_tokens = traffic.daily_requests * traffic.avg_output_tokens * billable_ratio
    daily_cost = (
        input_tokens * price.input_per_1m / 1_000_000
        + output_tokens * price.output_per_1m / 1_000_000
    )
    return CostBreakdown(
        model_name=price.name,
        billable_input_tokens=input_tokens,
        billable_output_tokens=output_tokens,
        daily_cost=daily_cost,
        monthly_cost=daily_cost * 30,
    )


def estimate_routing_cost(
    traffic: TrafficProfile,
    large_model_share: float,
    prices: Dict[str, PriceCard] = ILLUSTRATIVE_PRICES,
) -> Dict[str, CostBreakdown]:
    """Estimate cost when easy traffic is routed to a smaller model."""
    if not 0 <= large_model_share <= 1:
        raise ValueError("large_model_share must be in [0, 1]")

    large_traffic = TrafficProfile(
        daily_requests=int(traffic.daily_requests * large_model_share),
        avg_input_tokens=traffic.avg_input_tokens,
        avg_output_tokens=traffic.avg_output_tokens,
        full_response_cache_hit_rate=traffic.full_response_cache_hit_rate,
    )
    small_traffic = TrafficProfile(
        daily_requests=traffic.daily_requests - large_traffic.daily_requests,
        avg_input_tokens=traffic.avg_input_tokens,
        avg_output_tokens=traffic.avg_output_tokens,
        full_response_cache_hit_rate=traffic.full_response_cache_hit_rate,
    )
    return {
        "large": estimate_single_model_cost(prices["large"], large_traffic),
        "small": estimate_single_model_cost(prices["small"], small_traffic),
    }


def total_monthly_cost(breakdowns: List[CostBreakdown]) -> float:
    """Sum monthly costs across model routes."""
    return sum(item.monthly_cost for item in breakdowns)


def compare_cost_strategies() -> Dict[str, float]:
    """Compare baseline, cache, and routing strategies."""
    base_traffic = TrafficProfile(
        daily_requests=100_000,
        avg_input_tokens=900,
        avg_output_tokens=250,
    )
    cached_traffic = TrafficProfile(
        daily_requests=100_000,
        avg_input_tokens=900,
        avg_output_tokens=250,
        full_response_cache_hit_rate=0.35,
    )

    baseline = estimate_single_model_cost(ILLUSTRATIVE_PRICES["large"], base_traffic).monthly_cost
    cache_only = estimate_single_model_cost(ILLUSTRATIVE_PRICES["large"], cached_traffic).monthly_cost
    routed = estimate_routing_cost(cached_traffic, large_model_share=0.25)
    routing_with_cache = total_monthly_cost(list(routed.values()))

    return {
        "baseline_large_model": baseline,
        "cache_only": cache_only,
        "routing_with_cache": routing_with_cache,
        "monthly_savings": baseline - routing_with_cache,
    }


def main() -> None:
    comparison = compare_cost_strategies()
    print("Chapter 8.3: illustrative cost strategy comparison")
    print("Replace ILLUSTRATIVE_PRICES with a dated, sourced price card.")
    for name, value in comparison.items():
        print(f"{name:22s}: ${value:,.2f}/month")


if __name__ == "__main__":
    main()
