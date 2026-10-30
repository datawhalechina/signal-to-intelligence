"""Guards for application-method boundaries and engineering claims."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CODE = ROOT / "code"


def read_doc(relative_path):
    return DOCS.joinpath(relative_path).read_text(encoding="utf-8")


def read_code(relative_path):
    return CODE.joinpath(relative_path).read_text(encoding="utf-8")


def test_prompt_methods_are_routed_by_failure_cause_not_fixed_ladder():
    chapter = read_doc("06_llm_applications/README.md")
    prompt = read_doc("06_llm_applications/01_prompt_engineering.md")

    assert "不应被理解成必须依次升级的四个阶段" in chapter
    assert "缺外部知识 → 评估 RAG" in prompt
    assert "长示例、长上下文和反复调用" in prompt
    assert "完整内部推理过程" in prompt
    assert "应用侧做解析、校验和失败重试" in prompt


def test_finetuning_has_no_universal_sample_or_memory_threshold():
    chapter = read_doc("06_llm_applications/02_finetuning.md")
    advanced = read_doc("06_llm_applications/extensions/finetuning_advanced.md")

    assert "不存在脱离任务的通用“最低条数”" in chapter
    assert "通过学习曲线决定" in advanced
    assert "不是跨模型推荐值" in advanced
    assert "没有跨模型通用正常区间" in advanced
    assert "50-200 条" not in chapter
    assert "速度比标准 PEFT 快" not in advanced


def test_rag_keeps_retrieval_security_and_embedding_boundaries():
    rag = read_doc("06_llm_applications/03_rag.md")

    assert "粒度、训练目标和使用方式不同" in rag
    assert "不能直接把语言模型的 token embedding" in rag
    assert "仍需保护索引、日志与检索结果" in rag
    assert "引用存在不代表结论必然正确" in rag
    assert "200-500 tokens" not in rag


def test_agent_is_distinguished_from_deterministic_tool_workflows():
    agent = read_doc("06_llm_applications/04_agent.md")
    system = read_doc("06_llm_applications/05_system_design.md")

    assert "普通工作流通常更简单、可测" in agent
    assert "步骤通常不独立" in agent
    assert "拆成多个 Agent 不会自动提高可靠性" in agent
    assert "需要工具”和“需要 Agent" in agent
    assert "若查询路径固定，可使用确定性 API 工作流" in system
    assert "必须用 Agent" not in system


def test_deployment_extension_requires_workload_specific_measurement():
    deployment = read_doc("06_llm_applications/extensions/inference_deployment.md")

    assert "不提供跨硬件通用的“最佳框架”或固定加速倍数" in deployment
    assert "TTFT / TPOT / 端到端延迟分位数" in deployment
    assert r"C_{\text{operations}}+C_{\text{failure}}" in deployment
    assert "50M" not in deployment
    assert "1.5-2" not in deployment


def test_engineering_metrics_and_quantization_claims_are_conditioned():
    compression = read_doc("08_llm_engineering/01_quantization_distillation.md")
    inference = read_doc("08_llm_engineering/02_inference_optimization.md")

    assert "权重、激活或 KV cache" in compression
    assert "是否加速取决于硬件和内核" in compression
    assert "数值结果不要求逐位一致" in inference
    assert "p50/p95/p99" in inference
    assert "业务质量和错误率" in inference


def test_production_evaluation_covers_components_and_side_effects():
    evaluation = read_doc("08_llm_engineering/04_evaluation_benchmark.md")
    production = read_doc("08_llm_engineering/05_production_system.md")

    assert "候选召回、重排序、上下文覆盖" in evaluation
    assert "评分器给出的数字不是天然客观真值" in evaluation
    assert "不可信数据" in production
    assert "allowlist、最小权限、参数 schema" in production
    assert "幂等键、预览、确认或审批" in production


def test_teaching_scripts_label_simulated_and_illustrative_data():
    prompt = read_code("ch06_llm_applications/prompt_demo.py")
    system = read_code("ch06_llm_applications/system_design_demo.py")
    inference = read_code("ch08_llm_engineering/inference_benchmark.py")
    costs = read_code("ch08_llm_engineering/cost_calculator.py")

    assert '"data_kind": "illustrative_scenario"' in prompt
    assert '"data_kind": "illustrative_profiles"' in system
    assert 'data_kind: str = "simulated_estimate"' in inference
    assert "ILLUSTRATIVE_PRICES" in costs


def test_agent_demo_uses_safe_arithmetic_and_structural_lora_count():
    agent = read_code("ch06_llm_applications/agent_demo.py")

    assert "ast.parse" in agent
    assert "eval(" not in agent
    assert '"32 * 2 * 2 * 4096 * 8"' in agent
    assert "70_000_000_000 * 0.01" not in agent


def test_engineering_catalog_has_no_stale_provider_models_or_prices():
    engineering = read_code("ch08_llm_engineering/llm_engineering_demo.py")

    assert "ILLUSTRATIVE_MODELS" in engineering
    assert "teaching assumption" in engineering
    for stale_name in ("GPT-4", "GPT-3.5", "Claude 3", "Llama 2"):
        assert stale_name not in engineering
