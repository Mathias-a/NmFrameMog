from __future__ import annotations

import re
from pathlib import Path

DOC_PATH = Path("AGENTS.md")
SKILL_DOCS = sorted(Path(".claude/skills").glob("astar-*/SKILL.md"))

EXPECTED_LANE_SKILLS = {
    "Capture": {"astar-refresh-dataset"},
    "Solve": {"astar-solve-round", "astar-contract-check"},
    "Evaluate": {"astar-benchmark-suite", "astar-evaluate-solution"},
    "Report": {"astar-regression-review"},
}


def _read_doc() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


def _read_skill_doc(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _lane_section(text: str, lane_name: str) -> str:
    pattern = rf"### {lane_name}\n(.*?)(?=\n### |\n## |\Z)"
    match = re.search(pattern, text, flags=re.DOTALL)
    assert match is not None, f"Missing lane section for {lane_name}."
    return match.group(1)


def _skill_name(skill_doc: Path) -> str:
    match = re.search(r"^name:\s*(\S+)\s*$", _read_skill_doc(skill_doc), flags=re.MULTILINE)
    assert match is not None, f"Missing skill name in {skill_doc}."
    return match.group(1)


def _skill_commands(skill_doc: Path) -> set[str]:
    return set(
        re.findall(
            r"^(?:python -m nmframemog\.astar_island .+|PYTHONPATH=\. uv run --no-project --with pytest pytest tests/astar -q)$",
            _read_skill_doc(skill_doc),
            flags=re.MULTILINE,
        )
    )


def test_agents_doc_alignment_contains_required_sections_commands_skills_and_artifacts() -> None:
    text = _read_doc()

    required_sections = (
        "## Purpose",
        "## Source of truth",
        "## Artifact layout",
        "## Lane ownership",
        "## Required gates",
        "## Out of scope",
        "### Capture",
        "### Solve",
        "### Evaluate",
        "### Report",
    )
    for section in required_sections:
        assert section in text

    required_commands = (
        "python -m nmframemog.astar_island refresh-dataset",
        "python -m nmframemog.astar_island solve-round",
        "python -m nmframemog.astar_island evaluate-solution benchmark",
        "python -m nmframemog.astar_island evaluate-solution promote",
        "python -m nmframemog.astar_island render-debug",
        "python -m nmframemog.astar_island validate-prediction",
        "PYTHONPATH=. uv run --no-project --with pytest pytest tests/astar -q",
    )
    for command in required_commands:
        assert command in text

    required_skills = (
        "astar-contract-check",
        "astar-refresh-dataset",
        "astar-solve-round",
        "astar-benchmark-suite",
        "astar-evaluate-solution",
        "astar-regression-review",
    )
    for skill_name in required_skills:
        assert skill_name in text

    required_artifacts = (
        ".artifacts/astar-island/datasets/<version>/",
        "manifest.json",
        "hashes.json",
        "query-trace.json",
        "predictions/<run-id>/",
        "runs/<run-id>/summary.json",
        "debug/<run-id>/seed-XX/",
        ".sisyphus/evidence/",
    )
    for artifact in required_artifacts:
        assert artifact in text


def test_agents_doc_alignment_gives_each_lane_inputs_outputs_and_stop_conditions() -> None:
    text = _read_doc()

    for lane_name in ("Capture", "Solve", "Evaluate", "Report"):
        section = _lane_section(text, lane_name)
        for label in ("Owners:", "Skills:", "Command:", "Reads:", "Writes:", "Stop:"):
            assert label in section, f"Missing {label} in {lane_name} lane."


def test_agents_doc_alignment_maps_expected_skills_to_each_lane() -> None:
    text = _read_doc()

    for lane_name, expected_skills in EXPECTED_LANE_SKILLS.items():
        section = _lane_section(text, lane_name)
        for skill_name in expected_skills:
            assert skill_name in section, f"Missing {skill_name} in {lane_name} lane."


def test_agents_doc_alignment_makes_evaluation_lane_the_only_promotion_authority() -> None:
    text = _read_doc()

    assert "Only `astar-evaluate-solution` may promote a candidate." in text
    assert (
        "Capture, solve, and report lanes must not bless or replace references." in text
    )
    assert "use `astar-evaluate-solution promote`, never ad hoc comparisons" in text


def test_agents_doc_drift_mentions_every_skill_doc_name_and_command() -> None:
    text = _read_doc()

    assert SKILL_DOCS, "Expected Astar skill docs to exist."

    for skill_doc in SKILL_DOCS:
        assert _skill_name(skill_doc) in text
        for command in _skill_commands(skill_doc):
            assert command in text, f"Missing {command!r} from AGENTS.md."


def test_agents_doc_drift_limits_documented_workflows_to_known_skill_commands() -> None:
    text = _read_doc()

    documented_commands = set(
        re.findall(r"python -m nmframemog\.astar_island [^`\n]+", text)
    )
    known_skill_commands: set[str] = set()
    for skill_doc in SKILL_DOCS:
        known_skill_commands.update(
            command
            for command in _skill_commands(skill_doc)
            if command.startswith("python -m nmframemog.astar_island ")
        )

    assert documented_commands <= known_skill_commands


def test_agents_doc_stays_minimal_and_avoids_repo_wide_policy_duplication() -> None:
    text = _read_doc()
    lines = text.splitlines()

    assert len(lines) <= 70

    forbidden_headings = (
        "## Competition",
        "## Tech Stack",
        "## Code Style",
        "## Agent Workflow",
        "## Critical Competition Rules",
    )
    for heading in forbidden_headings:
        assert heading not in text

    forbidden_phrases = (
        "motivational prose",
        "generic agent theory",
        "broad repo coding policy",
    )
    for phrase in forbidden_phrases:
        assert phrase not in text.lower()
