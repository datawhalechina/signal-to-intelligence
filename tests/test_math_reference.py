"""Guardrails for the authoritative math reference and prerequisite map."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
APPENDIX = ROOT / "docs" / "appendix"


def read_reference(name):
    return APPENDIX.joinpath(name).read_text(encoding="utf-8")


def test_math_reference_uses_dimensionally_explicit_core_formulas():
    reference = read_reference("A_math_reference.md")

    assert r"\operatorname{atan2}(b,a)" in reference
    assert r"\mathbf{J}_{\mathbf{y}\leftarrow\mathbf{x}}^{\mathsf T}" in reference
    assert r"\widehat{\mathbf{m}}_t" in reference
    assert r"\widehat{\mathbf{v}}_t" in reference
    assert r"\mathbf{g}_t^{\odot 2}" in reference
    assert r"\sum_{i,j} |A_{ij}|^2" in reference
    assert r"\mathbf Q=\mathbf X\mathbf W_Q\in\mathbb R^{n\times d_k}" in reference
    assert r"\operatorname{softmax}_{\text{key}}" in reference
    assert r"p_\theta(x_{1:T})=\prod_{t=1}^T" in reference
    assert r"\Delta\mathbf W=\frac{\alpha}{r}\mathbf B\mathbf A" in reference
    assert r"L_{\mathrm{contrastive}}=\frac12" in reference

    assert r"\arctan(b/a)" not in reference
    assert r"\frac{m_t}{\sqrt{v_t} + \epsilon}" not in reference


def test_math_reference_covers_the_tutorial_concept_path():
    reference = read_reference("A_math_reference.md")

    required_sections = [
        "## A.4 信号处理",
        "## A.5 优化与统计学习",
        "## A.6 深度学习",
        "## A.7 Transformer",
        "## A.8 LLM 训练与生成",
        "## A.9 多模态表示与对齐",
        "## A.10 数值计算与评价指标",
    ]
    for section in required_sections:
        assert section in reference


def test_quick_reference_is_a_prerequisite_map_not_a_second_formula_table():
    reference = read_reference("D_math_quick_reference.md")

    assert "本附录回答“为了读某一章，我需要先补什么”" in reference
    assert "不再维护第二套完整公式" in reference
    assert "## D.1 章节前置总览" in reference
    assert "第 7 章多模态" in reference
    assert reference.count("A_math_reference.md#") >= 20

    assert r"h \sim \epsilon^{1/3}" in reference
    assert "宽平稳" in reference
    assert "不等同于严格平稳" in reference
    assert "因果有理系统" in reference
    assert "通常缺少通用的全局最优保证" in reference
    assert "交叉熵使用自然对数时" in reference

    assert r"h \approx \sqrt{\epsilon}" not in reference
    assert "**版本历史：**" not in reference
    assert "| 概念 | 公式" not in reference
