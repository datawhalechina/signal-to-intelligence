"""Tests for project metadata consistency."""

import re
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
STALE_LINK_PATTERNS = (
    "github.com/your-repo",
    "github.com/lynn_jingjing/signal-to-intelligence",
)
CHAPTER_DIRS = (
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
GENERATED_INDEX_HEADER = (
    "<!-- AUTO-GENERATED from README.md. Do not edit index.md directly. -->\n\n"
)


def project_metadata():
    with (ROOT / "pyproject.toml").open("rb") as f:
        return tomllib.load(f)["project"]


def test_readme_metadata_matches_pyproject():
    project = project_metadata()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    major_minor = ".".join(project["version"].split(".")[:2])

    assert f"**版本：** v{major_minor}" in readme
    assert f"📖 **在线文档：** {project['urls']['Documentation']}" in readme
    assert f"🔗 **GitHub 仓库：** {project['urls']['Repository']}" in readme


def test_public_authorship_and_license_metadata_are_explicit():
    project = project_metadata()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    author_names = {author["name"] for author in project["authors"]}

    assert author_names == {"Datawhale", "lynn_jingjing"}
    assert project["license"]["text"] == "CC-BY-NC-SA-4.0"
    assert "CC BY-NC-SA 4.0" in readme
    assert "MIT License" not in project.get("classifiers", [])


def test_vitepress_links_match_pyproject():
    project = project_metadata()
    config = (ROOT / "docs/.vitepress/config.mts").read_text(encoding="utf-8")
    repository = project["urls"]["Repository"]

    assert f"link: '{repository}'" in config
    assert f"pattern: '{repository}/blob/main/docs/:path'" in config

    github_links = set(re.findall(r"https://github.com/[^'\\\"]+", config))
    unexpected_links = {
        link
        for link in github_links
        if link.startswith("https://github.com/")
        and not link.startswith(repository)
    }
    assert unexpected_links == set()


def test_no_stale_repository_links():
    checked_files = [
        *ROOT.glob("*.md"),
        *ROOT.glob("*.toml"),
        *ROOT.glob("*.json"),
        *ROOT.glob(".github/workflows/*"),
        *ROOT.glob("docs/**/*.md"),
        *ROOT.glob("docs/.vitepress/**/*.mts"),
    ]

    stale_matches = []
    for path in checked_files:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        for pattern in STALE_LINK_PATTERNS:
            if pattern in text:
                stale_matches.append(f"{path.relative_to(ROOT)} contains {pattern}")

    assert stale_matches == []


def test_chapter_indexes_are_synced_from_readmes():
    for chapter in CHAPTER_DIRS:
        chapter_dir = ROOT / "docs" / chapter
        source = chapter_dir.joinpath("README.md").read_text(encoding="utf-8")
        normalized = "\n".join(line.rstrip() for line in source.splitlines()) + "\n"
        generated = chapter_dir.joinpath("index.md").read_text(encoding="utf-8")

        assert generated == GENERATED_INDEX_HEADER + normalized


def test_public_tutorial_has_no_prescriptive_study_time_or_empty_placeholders():
    checked_files = [ROOT / "README.md", *ROOT.glob("docs/**/*.md")]
    forbidden = (
        "**学习时间：**",
        "**预计时长：**",
        "## 每天的学习流程",
        "## 每周的学习计划",
        "TBD - 即将补充",
        "（待补充）",
        "<repo-url>",
    )

    matches = []
    for path in checked_files:
        text = path.read_text(encoding="utf-8")
        for marker in forbidden:
            if marker in text:
                matches.append(f"{path.relative_to(ROOT)} contains {marker}")

    assert matches == []

    time_range = re.compile(r"\b\d+\s*[-–]\s*\d+\s*(?:h|小时)\b", re.IGNORECASE)
    learning_terms = ("学习", "通道", "路径", "预计", "建议时长")
    timed_lines = []
    for path in checked_files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if time_range.search(line) and any(term in line for term in learning_terms):
                timed_lines.append(f"{path.relative_to(ROOT)}:{line_number}")

    assert timed_lines == []


def test_docs_do_not_duplicate_release_dates():
    dated_pages = []
    for path in ROOT.glob("docs/**/*.md"):
        if "**最后更新：**" in path.read_text(encoding="utf-8"):
            dated_pages.append(str(path.relative_to(ROOT)))

    assert dated_pages == []
