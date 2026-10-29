"""Guardrails for corrected formulas and their stated assumptions."""

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

    assert r"\arctan(b/a)" not in reference
    assert r"\frac{m_t}{\sqrt{v_t} + \epsilon}" not in reference


def test_quick_reference_states_numerical_and_model_assumptions():
    reference = read_reference("D_math_quick_reference.md")

    assert r"h \sim \epsilon^{1/3}" in reference
    assert "宽平稳过程" in reference
    assert "不等同于严格平稳" in reference
    assert "因果有理离散时间 LTI 系统" in reference
    assert "通常缺少通用的全局最优保证" in reference
    assert "交叉熵使用自然对数时" in reference

    assert r"h \approx \sqrt{\epsilon}" not in reference
    assert "**版本历史：**" not in reference
