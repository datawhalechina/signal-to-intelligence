#!/usr/bin/env python3
"""Inference simulation; outputs are estimates, never measured benchmarks."""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class InferenceConfig:
    name: str
    batch_size: int
    prompt_tokens: int
    output_tokens: int
    num_layers: int = 32
    hidden_size: int = 4096
    num_attention_heads: int = 32
    kv_heads: int = 32
    bytes_per_value: int = 2
    prefill_ms_per_token: float = 0.08
    decode_ms_per_token: float = 4.0
    batch_efficiency: float = 0.72


@dataclass
class BenchmarkResult:
    name: str
    batch_size: int
    kv_cache_mb: float
    prefill_ms: float
    decode_ms: float
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float
    throughput_tokens_per_sec: float
    data_kind: str = "simulated_estimate"


def estimate_kv_cache_mb(config: InferenceConfig) -> float:
    """Estimate KV cache memory for one active batch."""
    if min(
        config.batch_size,
        config.prompt_tokens,
        config.output_tokens,
        config.num_layers,
        config.hidden_size,
        config.num_attention_heads,
        config.kv_heads,
        config.bytes_per_value,
    ) <= 0:
        raise ValueError("configuration values must be positive")
    if config.hidden_size % config.num_attention_heads:
        raise ValueError("hidden_size must be divisible by num_attention_heads")
    if config.kv_heads > config.num_attention_heads:
        raise ValueError("kv_heads cannot exceed num_attention_heads")

    tokens = config.prompt_tokens + config.output_tokens
    head_dim = config.hidden_size // config.num_attention_heads
    values = (
        config.batch_size
        * tokens
        * config.num_layers
        * 2
        * config.kv_heads
        * head_dim
    )
    return values * config.bytes_per_value / 1024 / 1024


def simulate_inference(config: InferenceConfig) -> BenchmarkResult:
    """Estimate prefill/decode latency and output throughput."""
    batch_gain = max(1.0, config.batch_size * config.batch_efficiency)
    prefill_ms = config.prompt_tokens * config.prefill_ms_per_token * config.batch_size / batch_gain
    decode_ms = config.output_tokens * config.decode_ms_per_token * config.batch_size / batch_gain
    total_ms = prefill_ms + decode_ms
    generated_tokens = config.batch_size * config.output_tokens
    throughput = generated_tokens / (total_ms / 1000)

    return BenchmarkResult(
        name=config.name,
        batch_size=config.batch_size,
        kv_cache_mb=estimate_kv_cache_mb(config),
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        ttft_ms=prefill_ms,
        tpot_ms=decode_ms / config.output_tokens,
        total_latency_ms=total_ms,
        throughput_tokens_per_sec=throughput,
    )


def compare_serving_modes() -> List[BenchmarkResult]:
    """Compare typical serving configurations for the chapter example."""
    configs = [
        InferenceConfig(name="single request", batch_size=1, prompt_tokens=512, output_tokens=128),
        InferenceConfig(name="continuous batching", batch_size=8, prompt_tokens=512, output_tokens=128),
        InferenceConfig(name="long context", batch_size=4, prompt_tokens=4096, output_tokens=128),
    ]
    return [simulate_inference(config) for config in configs]


def summarize_benchmark(results: List[BenchmarkResult]) -> Dict[str, str]:
    """Return the highest-throughput and lowest-memory serving modes."""
    fastest = max(results, key=lambda item: item.throughput_tokens_per_sec)
    smallest_cache = min(results, key=lambda item: item.kv_cache_mb)
    return {
        "highest_throughput": fastest.name,
        "lowest_kv_cache": smallest_cache.name,
    }


def main() -> None:
    results = compare_serving_modes()
    print("Chapter 8.2: inference simulation (not measured benchmark data)")
    for item in results:
        print(
            f"{item.name:20s} batch={item.batch_size:<2} "
            f"kv_cache={item.kv_cache_mb:8.1f} MB "
            f"ttft={item.ttft_ms:7.1f} ms tpot={item.tpot_ms:5.2f} ms "
            f"throughput={item.throughput_tokens_per_sec:6.1f} tok/s"
        )
    print("\nSummary:", summarize_benchmark(results))


if __name__ == "__main__":
    main()
