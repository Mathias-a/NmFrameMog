from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast


class DoctorArgs(argparse.Namespace):
    root: str


REQUIRED_PATHS = (
    Path("AGENTS.md"),
    Path(".claude/skills/ng-data-eval/SKILL.md"),
    Path(".claude/skills/ng-detector/SKILL.md"),
    Path(".claude/skills/ng-submission/SKILL.md"),
    Path("scripts"),
    Path("src/ng_data/cloud"),
    Path("src/ng_data/data"),
    Path("src/ng_data/eval"),
    Path("src/ng_data/detector"),
    Path("src/ng_data/classifier"),
    Path("src/ng_data/retrieval"),
    Path("src/ng_data/pipeline"),
    Path("src/ng_data/submission"),
    Path("tests/unit/test_project_structure.py"),
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check the minimal NorgesGruppen project structure."
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Project root to validate. Defaults to the current directory.",
    )
    return parser


def find_missing_paths(root: Path) -> list[Path]:
    return [path for path in REQUIRED_PATHS if not (root / path).exists()]


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = cast(DoctorArgs, parser.parse_args(argv))

    root = Path(args.root)
    missing_paths = find_missing_paths(root)
    if missing_paths:
        missing_display = ", ".join(str(path) for path in missing_paths)
        raise SystemExit(f"Missing required project paths: {missing_display}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
