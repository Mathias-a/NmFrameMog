from __future__ import annotations

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast


def _load_main() -> Callable[[], int]:
    project_root = Path(__file__).resolve().parent
    source_root = project_root / "src"
    source_root_text = str(source_root)
    if source_root_text not in sys.path:
        sys.path.insert(0, source_root_text)

    cli_path = source_root / "task_norgesgruppen_data" / "cli.py"
    spec = importlib.util.spec_from_file_location(
        "task_norgesgruppen_data.cli", cli_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not load NorgesGruppen runner module from {cli_path}"
        )

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    main = getattr(module, "main", None)
    if not callable(main):
        raise RuntimeError(
            "Loaded NorgesGruppen runner module does not expose a callable main()."
        )

    return cast(Callable[[], int], main)


if __name__ == "__main__":
    raise SystemExit(_load_main()())
