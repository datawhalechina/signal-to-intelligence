"""Cross-chapter guards for notation, formulas, and stated assumptions."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def read_doc(relative_path):
    return DOCS.joinpath(relative_path).read_text(encoding="utf-8")


def test_transformer_formulas_match_the_authoritative_reference():
    attention = read_doc("04_transformer/01_attention.md")
    architecture = read_doc("04_transformer/04_architecture.md")
    multihead = read_doc("04_transformer/02_multihead.md")
    position = read_doc("04_transformer/03_positional_encoding.md")

    assert r"\mathbf Q,\mathbf K\in\mathbb R^{n\times d_k}" in attention
    assert r"\operatorname{softmax}_{\text{key}}" in attention
    assert r"\mathbf O=\mathbf A\mathbf V\in\mathbb R^{n\times d_v}" in attention

    assert r"\sqrt{\sigma^2+\epsilon}" in architecture
    assert r"\text{LayerNorm}(x) = \gamma \frac{x - \mu}{\sigma} + \beta" not in architecture
    assert "标准多头注意力的 Q/K/V 与输出投影参数量仍约为 $4d^2$" in multihead
    assert "能计算编码”不等于模型能在更长序列上保持任务性能" in position


def test_information_theory_keeps_discrete_and_continuous_boundaries():
    theory = read_doc("05_llm_basics/extensions/information_theory_basics.md")

    assert r"h(X) = -\int p(x) \log p(x) dx" in theory
    assert "微分熵依赖坐标和单位，可以为负" in theory
    assert r"\mathbb E_P\left[-\log\frac{Q(X)}{P(X)}\right]" in theory
    assert r"\mathbb E_P\left[\frac{P(x)}{Q(x)}\right]" not in theory
    assert "KL 散度**：分布之间的距离" not in theory
    assert "应该单调下降" not in theory


def test_llm_explanations_avoid_guarantees_and_stale_score_claims():
    pretraining = read_doc("05_llm_basics/01_pretraining.md")
    evaluation = read_doc("05_llm_basics/06_evaluation.md")

    assert "保证输入输出语义空间一致" not in pretraining
    assert "不能作为跨模型固定常数" in pretraining
    assert "锁定数据集版本、评分脚本和提示模板" in evaluation
    assert "顶级模型也只有 50-80%" not in evaluation
    assert "比人工评估便宜 100 倍" not in evaluation


def test_contrastive_learning_uses_complete_and_symmetric_objectives():
    chapter = read_doc("07_multimodal_llm/01_vision_language.md")
    details = read_doc("07_multimodal_llm/extensions/multimodal_training_details.md")
    theory = read_doc("05_llm_basics/extensions/information_theory_basics.md")

    assert r"L_{\mathrm{contrastive}}" in chapter
    assert r"L_{I\to T}+L_{T\to I}" in chapter
    assert r"\exp(s(x,x^+)/\tau)+\sum_{x^-}" in theory
    assert r"\mathbf v_i\leftarrow\frac{\mathbf v_i}{\|\mathbf v_i\|_2}" in details

    assert "**梯度爆炸：** 小" not in chapter
    assert "**梯度消失：** 大" not in chapter
    assert "**典型值：** $\tau = 0.07$" not in chapter


def test_target_chapters_follow_the_representation_term_convention():
    chapter_roots = [
        DOCS / "04_transformer",
        DOCS / "05_llm_basics",
        DOCS / "07_multimodal_llm",
    ]
    for chapter_root in chapter_roots:
        for path in chapter_root.rglob("*.md"):
            assert "表征" not in path.read_text(encoding="utf-8"), path
