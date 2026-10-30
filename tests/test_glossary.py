"""Guardrails for the cross-chapter terminology reference."""

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
GLOSSARY_PATH = DOCS / "appendix" / "G_glossary.md"


def glossary_text():
    return GLOSSARY_PATH.read_text(encoding="utf-8")


def test_glossary_covers_the_cross_chapter_core_vocabulary():
    glossary = glossary_text()
    required_terms = (
        "表示",
        "嵌入",
        "token",
        "注意力",
        "模型推理",
        "推理能力",
        "预测",
        "生成",
        "Prompt",
        "微调",
        "检索增强生成",
        "Agent",
        "表示对齐",
        "偏好对齐",
        "融合",
        "信号 token",
        "量化",
        "端侧部署",
    )

    for term in required_terms:
        assert term in glossary

    table_entries = re.findall(r"^\| \*\*[^|]+\*\* \|", glossary, re.MULTILINE)
    assert len(table_entries) >= 60


def test_glossary_disambiguates_overloaded_terms():
    glossary = glossary_text()

    assert "统计推断" in glossary
    assert "模型推理" in glossary
    assert "推理能力" in glossary
    assert "一个 token 会对应一个嵌入向量，但 token 本身不是向量" in glossary
    assert "信号检测与目标检测" in glossary
    assert "数学公式仍维护在附录 A" in glossary


def test_glossary_is_linked_from_tutorial_entrypoints():
    linked_files = (
        DOCS / "index.md",
        DOCS / "appendix" / "README.md",
        DOCS / "appendix" / "INDEX.md",
        DOCS / "00_introduction" / "05_how_to_use_this_tutorial.md",
        DOCS / ".vitepress" / "config.mts",
    )

    for path in linked_files:
        content = path.read_text(encoding="utf-8")
        assert "G_glossary" in content
