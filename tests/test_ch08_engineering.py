"""Tests for Chapter 8: LLM Engineering."""


class TestCompressionScripts:
    """Test compression trade-off utilities."""

    def test_quantization_memory_error_tradeoff(self, load_code_module):
        quantization = load_code_module("code/ch08_llm_engineering/quantization_demo.py")
        result = quantization.run_compression_comparison()

        q16, q8, q4 = result["quantization"]

        assert q16.memory_kb > q8.memory_kb > q4.memory_kb
        assert q16.mean_abs_error <= q8.mean_abs_error <= q4.mean_abs_error
        assert result["distillation"]["size_reduction"] > 0.8

    def test_pruning_needs_sparse_kernel_for_speedup(self, load_code_module):
        quantization = load_code_module("code/ch08_llm_engineering/quantization_demo.py")
        result = quantization.run_compression_comparison()

        no_sparse_kernel, sparse_kernel = result["pruning"]

        assert no_sparse_kernel.needs_sparse_kernel
        assert no_sparse_kernel.illustrative_speedup_upper_bound == 1.0
        assert sparse_kernel.illustrative_speedup_upper_bound > 1.0
        assert result["data_kind"] == "synthetic_weights_and_illustrative_quality"


class TestInferenceBenchmarkScripts:
    """Test inference benchmark estimates."""

    def test_kv_cache_grows_with_context_and_batch(self, load_code_module):
        benchmark = load_code_module("code/ch08_llm_engineering/inference_benchmark.py")

        short = benchmark.InferenceConfig(
            name="short", batch_size=1, prompt_tokens=512, output_tokens=128
        )
        long = benchmark.InferenceConfig(
            name="long", batch_size=4, prompt_tokens=4096, output_tokens=128
        )

        assert benchmark.estimate_kv_cache_mb(long) > benchmark.estimate_kv_cache_mb(short)

        grouped_query = benchmark.InferenceConfig(
            name="gqa", batch_size=1, prompt_tokens=512, output_tokens=128, kv_heads=8
        )
        assert benchmark.estimate_kv_cache_mb(grouped_query) < benchmark.estimate_kv_cache_mb(short)

    def test_batching_improves_throughput(self, load_code_module):
        benchmark = load_code_module("code/ch08_llm_engineering/inference_benchmark.py")
        results = benchmark.compare_serving_modes()
        by_name = {item.name: item for item in results}

        assert (
            by_name["continuous batching"].throughput_tokens_per_sec
            > by_name["single request"].throughput_tokens_per_sec
        )
        assert by_name["long context"].kv_cache_mb > by_name["single request"].kv_cache_mb
        assert all(item.data_kind == "simulated_estimate" for item in results)
        assert all(item.ttft_ms == item.prefill_ms for item in results)


class TestCostCalculatorScripts:
    """Test cost calculator scenarios."""

    def test_cache_hit_rate_reduces_cost(self, load_code_module):
        costs = load_code_module("code/ch08_llm_engineering/cost_calculator.py")
        price = costs.DEFAULT_PRICES["large"]

        no_cache = costs.estimate_single_model_cost(
            price,
            costs.TrafficProfile(10_000, 1_000, 200, full_response_cache_hit_rate=0.0),
        )
        cached = costs.estimate_single_model_cost(
            price,
            costs.TrafficProfile(10_000, 1_000, 200, full_response_cache_hit_rate=0.5),
        )

        assert cached.monthly_cost == no_cache.monthly_cost * 0.5

    def test_routing_with_cache_beats_baseline(self, load_code_module):
        costs = load_code_module("code/ch08_llm_engineering/cost_calculator.py")
        comparison = costs.compare_cost_strategies()

        assert comparison["baseline_large_model"] > comparison["cache_only"]
        assert comparison["cache_only"] > comparison["routing_with_cache"]
        assert comparison["monthly_savings"] > 0


class TestModelSelection:
    """Test model selection utilities."""

    def test_model_selector_filters_by_latency_and_cost(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        selector = engineering.ModelSelector()

        candidates = selector.select_by_criteria(
            max_latency_ms=400,
            max_cost_per_1m_tokens=10,
            min_context_window=16_000,
        )

        candidate_names = {model.name for model in candidates}
        assert candidate_names == {"Illustrative API Small"}
        assert all(model.source == "teaching assumption" for model in candidates)

    def test_model_selector_requires_finetuning(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        selector = engineering.ModelSelector()

        candidates = selector.select_by_criteria(requires_finetuning=True)

        assert candidates
        assert all(model.supports_finetuning for model in candidates)


class TestCostOptimization:
    """Test cost analysis utilities."""

    def test_cost_estimation_scales_with_requests(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        analyzer = engineering.CostAnalyzer(engineering.ModelSelector())

        small = analyzer.estimate_cost("api-small", 100, 1_000, 200)
        large = analyzer.estimate_cost("api-small", 200, 1_000, 200)

        assert small.daily_cost > 0
        assert large.daily_cost == 2 * small.daily_cost
        assert small.monthly_cost == small.daily_cost * 30
        assert small.yearly_cost == small.daily_cost * 365

    def test_unknown_model_raises_error(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        analyzer = engineering.CostAnalyzer(engineering.ModelSelector())

        try:
            analyzer.estimate_cost("missing-model", 100, 1_000, 200)
        except ValueError as exc:
            assert "Unknown model" in str(exc)
        else:
            raise AssertionError("Expected ValueError for unknown model")


class TestRetryStrategy:
    """Test retry strategy."""

    def test_exponential_backoff_is_capped(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        strategy = engineering.RetryStrategy(initial_delay_ms=100, max_delay_ms=500)

        assert strategy.get_delay_ms(0) == 100
        assert strategy.get_delay_ms(1) == 200
        assert strategy.get_delay_ms(10) == 500

    def test_retryable_error_codes(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        strategy = engineering.RetryStrategy()

        assert strategy.should_retry(429)
        assert strategy.should_retry(503)
        assert not strategy.should_retry(400)


class TestMonitoring:
    """Test monitoring utilities."""

    def test_metrics_collector_aggregates_requests(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        collector = engineering.MetricsCollector()

        collector.record_request(latency_ms=100, tokens=50)
        collector.record_request(latency_ms=300, tokens=150, error=True)
        metrics = collector.get_current_metrics()

        assert metrics.avg_latency_ms == 200
        assert metrics.error_rate == 0.5
        assert metrics.tokens_per_request == 100

    def test_metrics_collector_alerts(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        collector = engineering.MetricsCollector()
        metrics = engineering.MetricSnapshot(
            timestamp=0,
            requests_per_sec=120,
            avg_latency_ms=1500,
            error_rate=0.1,
            cost_per_hour=0.0,
            tokens_per_request=100,
        )

        alerts = collector.check_alerts(metrics)

        assert "HIGH_ERROR_RATE: 10.0%" in alerts
        assert "HIGH_LATENCY: 1500ms" in alerts
        assert "HIGH_THROUGHPUT: 120.0 req/s" in alerts
