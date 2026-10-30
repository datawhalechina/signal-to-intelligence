"""Validate public entrypoints, local links, notebooks, and Make targets."""

import re
from pathlib import Path
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LINK_PATTERN = re.compile(r"\[[^\]]*\]\(([^)]+)\)")


def markdown_files():
    yield ROOT / "README.md"
    yield from DOCS.rglob("*.md")
    yield ROOT / "notebooks/README.md"
    yield from (ROOT / "projects").rglob("*.md")


def local_target_candidates(source: Path, raw_target: str):
    target = raw_target.strip().strip("<>")
    if not target or target.startswith("#"):
        return []
    if re.match(r"^[a-z][a-z0-9+.-]*:", target, re.IGNORECASE):
        return []

    path_part = unquote(target.split("#", 1)[0].split("?", 1)[0])
    if not path_part:
        return []

    if path_part.startswith("/"):
        relative = path_part.removeprefix("/")
        if relative.startswith(("assets/", "diagrams/")):
            base = DOCS / "public" / relative
        else:
            base = DOCS / relative
    else:
        base = source.parent / path_part

    candidates = [base]
    if not base.suffix:
        candidates.extend([base.with_suffix(".md"), base / "index.md"])
    return candidates


def test_local_markdown_links_resolve_to_files_or_routes():
    broken = []
    seen = set()
    for source in markdown_files():
        if not source.is_file() or source in seen:
            continue
        seen.add(source)
        text = source.read_text(encoding="utf-8")
        for line_number, line in enumerate(text.splitlines(), 1):
            for raw_target in LINK_PATTERN.findall(line):
                candidates = local_target_candidates(source, raw_target)
                if candidates and not any(candidate.exists() for candidate in candidates):
                    broken.append(
                        f"{source.relative_to(ROOT)}:{line_number} -> {raw_target}"
                    )
    assert broken == []


def test_quickstart_commands_have_make_targets():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    targets = set(re.findall(r"^([a-zA-Z0-9_-]+):", makefile, re.MULTILINE))

    for command in re.findall(r"\bmake ([a-zA-Z0-9_-]+)", readme):
        assert command in targets, command


def test_documented_notebooks_exist():
    usage = (DOCS / "00_introduction/05_how_to_use_this_tutorial.md").read_text(
        encoding="utf-8"
    )
    notebook_names = set(re.findall(r"`([^`]+\.ipynb)`", usage))

    assert len(notebook_names) == 8
    for name in notebook_names:
        assert (ROOT / "notebooks" / name).is_file(), name


def test_chapter_eight_make_target_runs_all_documented_scripts():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    target_body = makefile.split("run-exp-ch08:\n", 1)[1].split("\n\n", 1)[0]
    expected = {
        "quantization_demo.py",
        "inference_benchmark.py",
        "cost_calculator.py",
        "llm_engineering_demo.py",
    }

    assert expected == set(re.findall(r"ch08_llm_engineering/([^\s]+\.py)", target_body))


def test_chapter_eight_practice_distinguishes_simulation_from_measurement():
    chapter = (DOCS / "08_llm_engineering/README.md").read_text(encoding="utf-8")

    assert "成本计算或推理模拟脚本" in chapter
    assert "模拟结果不能写成硬件实测" in chapter
    assert "推理基准脚本" not in chapter


def test_code_guide_does_not_expose_secrets_or_unverified_cloud_entrypoints():
    guide = (DOCS / "appendix/C_code_guide.md").read_text(encoding="utf-8")

    assert "print(os.environ.get('OPENAI_API_KEY'))" not in guide
    assert "推荐：** CNB" not in guide
    assert "Colab 有免费 T4 GPU" not in guide
    assert "当前没有经过维护者验证的 CNB 一键入口" in guide
