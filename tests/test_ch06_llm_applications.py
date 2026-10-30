"""Tests for Chapter 6: LLM Applications"""

class TestRAGSystem:
    """Test RAG (Retrieval-Augmented Generation) system"""

    def test_document_retrieval(self, load_code_module):
        """Test document retrieval"""
        rag_demo = load_code_module("code/ch06_llm_applications/rag_demo.py")
        rag = rag_demo.SimpleRAG()

        retrieved_docs = rag.retrieve("什么是Transformer？", top_k=2)

        assert len(retrieved_docs) == 2
        assert retrieved_docs[0]["title"] == "什么是Transformer"

    def test_context_augmentation(self, load_code_module):
        """Test context augmentation"""
        rag_demo = load_code_module("code/ch06_llm_applications/rag_demo.py")
        rag = rag_demo.SimpleRAG()

        query = "What is attention?"
        retrieved_docs = [
            {"title": "Attention", "content": "Attention computes weighted sums of values."},
            {"title": "Self-attention", "content": "Each position attends to all positions."},
        ]

        augmented_context = rag.generate_with_context(query, retrieved_docs)

        assert query in augmented_context
        assert all(doc["title"] in augmented_context for doc in retrieved_docs)
        assert all(doc["content"] in augmented_context for doc in retrieved_docs)


class TestFineTuning:
    """Test fine-tuning"""

    def test_lora_trainable_parameters_scale_with_rank(self, load_code_module):
        """Test that LoRA trainable parameters scale linearly with rank."""
        finetuning_demo = load_code_module("code/ch06_llm_applications/finetuning_demo.py")

        params_r4 = finetuning_demo.lora_params(4)
        params_r8 = finetuning_demo.lora_params(8)
        params_r16 = finetuning_demo.lora_params(16)

        assert params_r4 > 0
        assert params_r8 == 2 * params_r4
        assert params_r16 == 2 * params_r8
        assert params_r16 < finetuning_demo.total_params


class TestAgentFramework:
    """Test agent framework"""

    def test_agent_decision_making(self, load_code_module):
        """Test agent decision making"""
        agent_demo = load_code_module("code/ch06_llm_applications/agent_demo.py")

        decision = agent_demo.simulate_llm_reason("什么是 RAG？", history=[])

        assert decision["action"] == "search"
        assert "RAG" in decision["input"]

    def test_agent_calculator_tool(self, load_code_module):
        """Test agent calculator tool."""
        agent_demo = load_code_module("code/ch06_llm_applications/agent_demo.py")

        assert agent_demo.calculate("2 + 2") == "4"
        assert "计算错误" in agent_demo.calculate("__import__('os').getcwd()")

    def test_agent_run_experiment(self, load_code_module):
        agent_demo = load_code_module("code/ch06_llm_applications/agent_demo.py")
        result = agent_demo.run_experiment()
        expected_answer = (
            "RAG（检索增强生成）先从知识库检索相关文档，"
            "再用文档增强 LLM 的生成过程。"
        )

        assert result["answer1"] == expected_answer
        assert "4,194,304" in result["answer2"]
        assert all("rationale" in item for item in result["transcript2"])
        assert len(result["transcript1"]) >= 1
        assert len(result["transcript2"]) >= 2
        assert result["chart"]["steps"] == list(range(1, 11))


class TestPromptAndSystemDesign:
    """Test prompt engineering and system design demos"""

    def test_prompt_demo_run_experiment(self, load_code_module):
        prompt_demo = load_code_module("code/ch06_llm_applications/prompt_demo.py")
        result = prompt_demo.run_experiment()

        assert result["data_kind"] == "illustrative_scenario"
        assert len(result["tasks"]) == 4
        assert result["zero_shot"].shape == (4,)
        assert result["few_shot"].shape == (4,)
        assert result["cot"].shape == (4,)
        assert len(result["steps"]) == 4
        assert result["few_shot"][0] > result["zero_shot"][0]
        assert result["cot"][1] > result["zero_shot"][1]

    def test_system_design_run_experiment(self, load_code_module):
        system_design = load_code_module("code/ch06_llm_applications/system_design_demo.py")
        result = system_design.run_experiment()

        assert result["data_kind"] == "illustrative_profiles"
        assert len(result["categories"]) == 6
        assert set(result["approaches"]) == {"Prompt", "Fine-tuning", "RAG", "Agent"}
        assert result["setup_cost"].shape == (4,)
        assert result["run_cost"].shape == (4,)
        assert result["maintain"].shape == (4,)
        assert result["totals"].shape == (4,)
        assert result["totals"][1] > result["totals"][0]
