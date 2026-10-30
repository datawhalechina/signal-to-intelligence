"""Regression guards for the DSP-to-Transformer foundation chapters."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"


def read_doc(relative_path):
    return DOCS.joinpath(relative_path).read_text(encoding="utf-8")


def test_sampling_and_white_noise_keep_their_assumptions():
    fourier = read_doc("01_dsp/02_fourier.md")
    random_signals = read_doc("01_dsp/05_random_signals.md")

    assert r"f_s>2f_{\max}" in fourier
    assert "恰好位于边界" in fourier
    assert "不存在可直接套用的语言版 Nyquist 定理" in fourier
    assert "只有在联合高斯等附加条件下，才能进一步推出独立" in random_signals
    assert "均值遍历不自动推出自相关也遍历" in random_signals


def test_covariance_is_centered_and_sample_normalization_is_explicit():
    decomposition = read_doc("01_dsp/08_matrix_decomposition.md")

    assert r"\mathbf C" in decomposition
    assert r"(\mathbf x-\boldsymbol\mu)(\mathbf x-\boldsymbol\mu)^{\mathsf H}" in decomposition
    assert "未中心化二阶矩" in decomposition
    assert r"\frac1{N-1}" in decomposition


def test_detection_formula_and_optimality_are_conditioned():
    detection = read_doc("01_dsp/06_signal_detection.md")

    assert "两个完全指定的简单假设" in detection
    assert r"s^*[N-1-n]" in detection
    assert r"Q^{-1}(P_{fa})-\sqrt{\operatorname{SNR}_{\mathrm{out}}}" in detection
    assert r"\sqrt{2 \cdot \text{SNR}} - Q^{-1}(P_{fa})" not in detection
    assert "该观测区间内的信号能量" in detection


def test_estimation_states_rank_and_regular_conditions():
    estimation = read_doc("01_dsp/07_signal_estimation.md")

    assert "若 $\\mathbf X$ 满列秩" in estimation
    assert r"\mathbf X^+\mathbf y" in estimation
    assert "模型可辨识" in estimation
    assert "向量参数" in estimation
    assert r"\mathbf I(\boldsymbol\theta)" in estimation


def test_system_and_filter_claims_are_scoped():
    systems = read_doc("01_dsp/extensions/linear_systems.md")
    filters = read_doc("01_dsp/03_filters.md")

    assert "因果有理离散时间 LTI 系统" in systems
    assert "系数有限时冲激响应绝对可和" in systems
    assert "多数框架实际计算不翻转核的互相关" in filters


def test_convexity_does_not_overpromise_existence_or_convergence():
    convexity = read_doc("02_optimization/extensions/convex_analysis.md")

    assert "凸性本身不保证最优解一定存在、唯一" in convexity
    assert "凸函数上梯度下降必然收敛" in convexity
    assert "Slater 条件" in convexity
    assert "过参数化一定更容易优化" in convexity


def test_deep_learning_bridge_uses_structural_claims():
    cnn = read_doc("03_deep_learning_fast/02_cnn_essence.md")
    sequence = read_doc("03_deep_learning_fast/05_sequence_models_and_1d_signals.md")
    transformer = read_doc("03_deep_learning_fast/06_why_transformer_better.md")

    assert "平移等变性" in cnn
    assert "不能只由卷积运算保证" in cnn
    assert "Jacobian" in sequence
    assert r"\operatorname{softmax}_{\text{key}}" in transformer
    assert "由注意力 Mask 决定" in transformer


def test_foundation_chapters_use_representation_term_consistently():
    for chapter in ("01_dsp", "02_optimization", "03_deep_learning_fast"):
        for path in (DOCS / chapter).rglob("*.md"):
            assert "表征" not in path.read_text(encoding="utf-8"), path
