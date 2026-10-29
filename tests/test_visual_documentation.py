"""Guardrails for chapter-level SVG diagrams and their documentation links."""

import re
from pathlib import Path
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
CORE_CHAPTERS = (
    "00_introduction",
    "01_dsp",
    "02_optimization",
    "03_deep_learning_fast",
    "04_transformer",
    "05_llm_basics",
    "06_llm_applications",
    "07_multimodal_llm",
    "08_llm_engineering",
)
LEARNING_LOOP_CHAPTERS = CORE_CHAPTERS[1:]
DIAGRAM_PATTERN = re.compile(r"!\[[^]]*\]\((/diagrams/[^)]+\.svg)\)")
SVG_NAMESPACE = "{http://www.w3.org/2000/svg}"
DIAGRAM_DIR = DOCS / "public" / "diagrams"


def chapter_diagrams():
    """Return chapter README paths and their declared SVG diagram paths."""
    diagrams = []
    for chapter in CORE_CHAPTERS:
        readme_path = DOCS / chapter / "README.md"
        content = readme_path.read_text(encoding="utf-8")
        assert "*章节主图：" in content, f"{chapter} must explain its chapter diagram"
        matches = DIAGRAM_PATTERN.findall(content)
        assert matches, f"{chapter} must reference a chapter SVG diagram"
        for diagram_url in matches:
            diagram_path = DOCS / "public" / diagram_url.removeprefix("/")
            assert diagram_path.is_file(), f"Missing diagram: {diagram_url}"
        diagrams.extend(matches)
    return diagrams


def test_every_core_chapter_has_a_maintained_svg_main_diagram():
    diagram_paths = chapter_diagrams()

    assert len(diagram_paths) == len(CORE_CHAPTERS)


def test_all_maintained_svg_diagrams_are_accessible_and_scalable():
    diagram_paths = sorted(DIAGRAM_DIR.glob("*.svg"))

    assert diagram_paths
    for diagram_path in diagram_paths:

        root = ElementTree.parse(diagram_path).getroot()
        assert root.tag == f"{SVG_NAMESPACE}svg"
        assert root.attrib.get("role") == "img"
        assert root.attrib.get("viewBox")
        assert root.find(f"{SVG_NAMESPACE}title") is not None
        assert root.find(f"{SVG_NAMESPACE}desc") is not None


def test_chapters_one_to_eight_connect_diagram_practice_and_code():
    for chapter in LEARNING_LOOP_CHAPTERS:
        content = (DOCS / chapter / "README.md").read_text(encoding="utf-8")

        assert "## 本章学习闭环" in content
        assert "## 最低实践任务" in content
        assert "## 代码实验" in content


def test_signal_language_alignment_and_edge_acceptance_are_core_sections():
    chapter_seven = (DOCS / "07_multimodal_llm/README.md").read_text(encoding="utf-8")
    chapter_eight = (DOCS / "08_llm_engineering/README.md").read_text(encoding="utf-8")
    vitepress_config = (DOCS / ".vitepress/config.mts").read_text(encoding="utf-8")

    assert "02_signal_language_alignment.md" in chapter_seven
    assert "06_edge_multimodal_deployment.md" in chapter_eight
    assert "/07_multimodal_llm/02_signal_language_alignment" in vitepress_config
    assert "/08_llm_engineering/06_edge_multimodal_deployment" in vitepress_config
