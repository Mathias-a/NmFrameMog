from __future__ import annotations

import re
from pathlib import Path

SKILL_DOCS = {
    "astar-contract-check": Path(".claude/skills/astar-contract-check/SKILL.md"),
    "astar-refresh-dataset": Path(".claude/skills/astar-refresh-dataset/SKILL.md"),
    "astar-solve-round": Path(".claude/skills/astar-solve-round/SKILL.md"),
    "astar-benchmark-suite": Path(".claude/skills/astar-benchmark-suite/SKILL.md"),
    "astar-evaluate-solution": Path(".claude/skills/astar-evaluate-solution/SKILL.md"),
    "astar-regression-review": Path(".claude/skills/astar-regression-review/SKILL.md"),
}

COMMON_SECTION_HEADINGS = (
    "## Purpose",
    "## Allowed inputs",
    "## Exact commands",
    "## Artifacts produced",
    "## Evidence paths",
    "## Refusal conditions",
)


def _read_skill(name: str) -> str:
    return SKILL_DOCS[name].read_text(encoding="utf-8")


def _astar_commands(text: str) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"^python -m nmframemog\.astar_island .+$",
            text,
            flags=re.MULTILINE,
        )
    )


def _runnable_wrapper_commands(text: str) -> tuple[str, ...]:
    return tuple(
        re.findall(
            r"^uv run --no-project python -m nmframemog\.astar_island .+$",
            text,
            flags=re.MULTILINE,
        )
    )


def test_skill_docs_exist_with_yaml_frontmatter_and_required_sections() -> None:
    for name, path in SKILL_DOCS.items():
        assert path.exists(), f"Missing skill doc for {name}."
        text = path.read_text(encoding="utf-8")
        assert text.startswith("---\n")
        assert f"name: {name}" in text
        for heading in COMMON_SECTION_HEADINGS:
            assert heading in text, f"Missing {heading} in {path}."


def test_skill_docs_define_one_lane_specific_command_surface() -> None:
    expected_commands = {
        "astar-contract-check": (
            "python -m nmframemog.astar_island validate-prediction <prediction-file>",
        ),
        "astar-refresh-dataset": (
            "uv run --no-project python -m nmframemog.astar_island "
            "refresh-dataset --round-id <completed-round-id> --cache-dir "
            ".artifacts/astar-island --dataset-version <version>",
        ),
        "astar-solve-round": (
            "python -m nmframemog.astar_island solve-round --round-detail-file "
            ".artifacts/astar-island/datasets/<version>/rounds/<round-id>.json "
            "--cache-dir .artifacts/astar-island",
        ),
        "astar-benchmark-suite": (
            "uv run --no-project python -m nmframemog.astar_island "
            "evaluate-solution benchmark --cache-dir .artifacts/astar-island "
            "--dataset-version <version> --candidate <candidate-id>",
        ),
        "astar-evaluate-solution": (
            "uv run --no-project python -m nmframemog.astar_island "
            "evaluate-solution promote --cache-dir .artifacts/astar-island "
            "--dataset-version <version> --candidate <candidate-id>",
        ),
        "astar-regression-review": (
            "python -m nmframemog.astar_island render-debug --input <trace.json> "
            "--output-dir .artifacts/astar-island/debug/<run-id>/seed-00",
        ),
    }

    for name, commands in expected_commands.items():
        text = _read_skill(name)
        for command in commands:
            assert command in text, f"Missing command in {name}: {command}"


def test_skill_docs_lock_single_operator_command_for_refresh_benchmark_and_promote(
) -> None:
    offline_gate = (
        "PYTHONPATH=. uv run --no-project --with pytest pytest tests/astar -q"
    )
    expected_operator_commands = {
        "astar-refresh-dataset": (
            "uv run --no-project python -m nmframemog.astar_island "
            "refresh-dataset --round-id <completed-round-id> --cache-dir "
            ".artifacts/astar-island --dataset-version <version>"
        ),
        "astar-benchmark-suite": (
            "uv run --no-project python -m nmframemog.astar_island "
            "evaluate-solution benchmark --cache-dir .artifacts/astar-island "
            "--dataset-version <version> --candidate <candidate-id>"
        ),
        "astar-evaluate-solution": (
            "uv run --no-project python -m nmframemog.astar_island "
            "evaluate-solution promote --cache-dir .artifacts/astar-island "
            "--dataset-version <version> --candidate <candidate-id>"
        ),
    }

    for name, expected_command in expected_operator_commands.items():
        text = _read_skill(name)
        assert _runnable_wrapper_commands(text) == (expected_command,)
        assert text.count(offline_gate) == 1
        assert "only operator-facing" in text
        assert "offline quality and CI gate" in text


def test_skill_docs_keep_benchmark_and_promote_aligned_with_cli_entrypoints() -> None:
    cli_text = Path("idk_2/astar_island_dr_plan_1/cli.py").read_text(
        encoding="utf-8"
    )

    assert '"refresh-dataset"' in cli_text
    assert "def _refresh_dataset(args: argparse.Namespace) -> int:" in cli_text
    assert '"evaluate-solution"' in cli_text
    assert 'for evaluate_mode in ("benchmark", "promote"):' in cli_text
    assert "dest=\"evaluate_mode\"" in cli_text


def test_skill_docs_describe_required_artifacts_and_evidence_paths() -> None:
    refresh_text = _read_skill("astar-refresh-dataset")
    assert ".artifacts/astar-island/datasets/<version>/manifest.json" in refresh_text
    assert ".artifacts/astar-island/datasets/<version>/hashes.json" in refresh_text
    assert ".artifacts/astar-island/datasets/<version>/query-trace.json" in refresh_text

    solve_text = _read_skill("astar-solve-round")
    assert "predictions/<run-id>/" in solve_text
    assert "runs/<run-id>/summary.json" in solve_text
    assert "debug/<run-id>/seed-XX/" in solve_text

    report_text = _read_skill("astar-regression-review")
    assert ".artifacts/astar-island/debug/<run-id>/seed-XX/" in report_text
    assert ".sisyphus/evidence/" in report_text


def test_skill_docs_include_frozen_dataset_and_lane_authority_guards() -> None:
    benchmark_text = _read_skill("astar-benchmark-suite")
    assert "frozen dataset version" in benchmark_text
    assert "live API calls during offline replay or benchmarking" in benchmark_text
    assert "mix multiple dataset versions" in benchmark_text
    assert "bless or promote a candidate" in benchmark_text

    evaluate_text = _read_skill("astar-evaluate-solution")
    assert "only skill that may promote a candidate" in evaluate_text
    assert "same frozen dataset version" in evaluate_text
    assert (
        "ad hoc promotion logic outside `astar-evaluate-solution promote`"
        in evaluate_text
    )


def test_skill_docs_refusal_sections_prevent_scope_drift() -> None:
    refusal_checks = {
        "astar-contract-check": (
            "more than one prediction file",
            "benchmarking, promotion, dataset refresh, or debug rendering",
        ),
        "astar-refresh-dataset": (
            "round is still live or incomplete",
            "mix artifacts from multiple dataset versions",
        ),
        "astar-solve-round": (
            "live fetching by round id",
            "benchmark, promote, or bless",
        ),
        "astar-benchmark-suite": (
            "live API calls during offline replay or benchmarking",
            "bless or promote a candidate",
        ),
        "astar-evaluate-solution": (
            "benchmarking has not already been completed",
            "ad hoc promotion logic outside `astar-evaluate-solution promote`",
        ),
        "astar-regression-review": (
            "compute new predictions, benchmark results, or promotion verdicts",
            "mix traces or reports from different frozen dataset versions",
        ),
    }

    for name, required_phrases in refusal_checks.items():
        text = _read_skill(name)
        refusal_section = text.split("## Refusal conditions", maxsplit=1)[1]
        for phrase in required_phrases:
            assert phrase in refusal_section, (
                f"Missing refusal phrase in {name}: {phrase}"
            )
