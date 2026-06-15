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
