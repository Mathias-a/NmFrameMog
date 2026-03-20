from __future__ import annotations

import json
from pathlib import Path


class LocalCache:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.rounds_dir = root / "rounds"
        self.queries_dir = root / "queries"
        self.analysis_dir = root / "analysis"
        self.predictions_dir = root / "predictions"
        self.runs_dir = root / "runs"
        self.debug_dir = root / "debug"
        self.mapping_dir = root / "mapping"

    def ensure(self) -> None:
        for directory in (
            self.root,
            self.rounds_dir,
            self.queries_dir,
            self.analysis_dir,
            self.predictions_dir,
            self.runs_dir,
            self.debug_dir,
            self.mapping_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def save_json(self, path: Path, payload: object) -> None:
        if not _is_json_value(payload):
            raise ValueError(f"Path {path} received a non-JSON-compatible payload.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def load_json(self, path: Path) -> object:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
        if not _is_json_value(payload):
            raise ValueError(f"File {path} does not contain JSON-compatible data.")
        return payload

    def round_detail_path(self, round_id: str) -> Path:
        return self.rounds_dir / f"{round_id}.json"

    def query_response_path(
        self, round_id: str, seed_index: int, viewport_key: str
    ) -> Path:
        return (
            self.queries_dir
            / round_id
            / f"seed-{seed_index:02d}"
            / f"{viewport_key}.json"
        )

    def analysis_path(self, round_id: str, seed_index: int) -> Path:
        return self.analysis_dir / round_id / f"seed-{seed_index:02d}.json"

    def prediction_path(self, run_id: str, seed_index: int) -> Path:
        return self.predictions_dir / run_id / f"seed-{seed_index:02d}.json"

    def debug_output_dir(self, run_id: str, seed_index: int) -> Path:
        return self.debug_dir / run_id / f"seed-{seed_index:02d}"

    def run_summary_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id / "summary.json"

    def run_config_path(self, run_id: str) -> Path:
        return self.runs_dir / run_id / "config.json"

    def mapping_artifact_path(self) -> Path:
        return self.mapping_dir / "terrain_mapping.json"


def _is_json_value(value: object) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(
            isinstance(key, str) and _is_json_value(item) for key, item in value.items()
        )
    return False
