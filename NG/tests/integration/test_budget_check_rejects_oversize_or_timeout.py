from __future__ import annotations

import json
import zipfile
from importlib import import_module
from pathlib import Path
from typing import Any, cast

budget_check = cast(Any, import_module("src.ng_data.submission.budget_check"))


def _write_submission_zip(zip_path: Path, extra_members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("run.py", "from pathlib import Path\n")
        for name, payload in extra_members.items():
            archive.writestr(name, payload)


def _write_input_dir(input_dir: Path) -> None:
    input_dir.mkdir()
    (input_dir / "img_00001.jpg").write_bytes(b"fixture")


def test_budget_check_rejects_oversize_weight_file_budget_before_smoke_run(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    typed_monkeypatch = cast(Any, monkeypatch)
    typed_capsys = cast(Any, capsys)
    zip_path = tmp_path / "oversize.zip"
    out_path = tmp_path / "budget.json"
    input_dir = tmp_path / "images"
    _write_input_dir(input_dir)
    _write_submission_zip(
        zip_path,
        {
            "one.pt": b"1",
            "two.pt": b"2",
            "three.pt": b"3",
            "four.pt": b"4",
        },
    )

    def _unexpected_smoke_run(
        *_args: object, **_kwargs: object
    ) -> list[dict[str, object]]:
        raise AssertionError("budget check should reject before smoke run")

    typed_monkeypatch.setattr(
        budget_check, "run_smoke_submission", _unexpected_smoke_run
    )

    exit_code = budget_check.budget_check_main(
        ["--zip", str(zip_path), "--input", str(input_dir), "--out", str(out_path)]
    )
    captured = typed_capsys.readouterr()
    evidence = json.loads(out_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert "weight_file_count exceeded 3 with observed 4" in captured.err
    assert evidence["status"] == "fail"
    assert evidence["failed_checks"] == ["weight_file_count"]
    assert evidence["observed"]["weight_file_count"] == 4
    assert evidence["observed"]["smoke_runtime_seconds"] is None


def test_budget_check_rejects_timeout_threshold_breach(
    tmp_path: Path,
    monkeypatch: object,
    capsys: object,
) -> None:
    typed_monkeypatch = cast(Any, monkeypatch)
    typed_capsys = cast(Any, capsys)
    zip_path = tmp_path / "slow.zip"
    out_path = tmp_path / "budget.json"
    input_dir = tmp_path / "images"
    _write_input_dir(input_dir)
    _write_submission_zip(zip_path, {})

    timer_values = iter((100.0, 341.25))

    def _fake_perf_counter() -> float:
        return next(timer_values)

    def _fake_smoke_run(
        _zip_path: Path, _input_dir: Path, _output_path: Path
    ) -> list[dict[str, object]]:
        return [
            {
                "bbox": [1.0, 2.0, 3.0, 4.0],
                "category_id": 0,
                "image_id": 1,
                "score": 0.9,
            }
        ]

    typed_monkeypatch.setattr(budget_check, "perf_counter", _fake_perf_counter)
    typed_monkeypatch.setattr(budget_check, "run_smoke_submission", _fake_smoke_run)

    exit_code = budget_check.budget_check_main(
        ["--zip", str(zip_path), "--input", str(input_dir), "--out", str(out_path)]
    )
    captured = typed_capsys.readouterr()
    evidence = json.loads(out_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert "smoke_runtime_seconds exceeded 240.0 with observed 241.25" in captured.err
    assert evidence["status"] == "fail"
    assert evidence["failed_checks"] == ["smoke_runtime_seconds"]
    assert evidence["observed"]["prediction_count"] == 1
    assert evidence["observed"]["smoke_runtime_seconds"] == 241.25
