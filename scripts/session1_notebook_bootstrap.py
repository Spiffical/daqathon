from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def setup_jsonable(value: Any) -> Any:
    """Convert common notebook setup values into JSON-printable objects."""

    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except TypeError:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, Mapping):
        return {str(key): setup_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [setup_jsonable(item) for item in value]
    return value


def show_setup_json(value: Mapping[str, Any]) -> None:
    """Print a small setup dictionary as readable JSON."""

    print(json.dumps(setup_jsonable(value), indent=2, ensure_ascii=False))


def first_existing_path(candidates: Iterable[str | Path | None]) -> Path | None:
    """Return the first candidate path that exists."""

    for candidate in candidates:
        if candidate is None:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists():
            return candidate_path
    return None


def first_existing_csv_dir(candidates: Iterable[str | Path | None]) -> Path | None:
    """Return the first directory candidate that contains at least one CSV file."""

    for candidate in candidates:
        if candidate is None:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists() and candidate_path.is_dir() and any(candidate_path.glob("*.csv")):
            return candidate_path
    return None


def find_notebook_root(start: str | Path | None = None) -> Path:
    """Find the cloned repo root from a notebook working directory."""

    current = Path.cwd() if start is None else Path(start).expanduser()
    for candidate_root in [current, *current.parents]:
        if (candidate_root / "notebooks").exists() and (candidate_root / "scripts").exists():
            return candidate_root
    return current


def build_notebook_bootstrap_namespace(start: str | Path | None = None) -> dict[str, Any]:
    """Resolve the small set of paths shared by the Session 1 notebooks."""

    notebook_root = find_notebook_root(start)
    shared_root_candidates = [
        Path("/project/def-kmoran/shared/daqathon"),
        Path("/project/6062898/shared/daqathon"),
        Path.home() / "projects" / "def-kmoran" / "shared" / "daqathon",
    ]
    shared_daqathon_root = first_existing_path(shared_root_candidates)
    local_cache_dir = notebook_root / "data" / "cache" / "session1"
    shared_cache_dir = (
        first_existing_path([shared_daqathon_root / "data" / "cache" / "session1"])
        if shared_daqathon_root
        else None
    )

    return {
        "NOTEBOOK_ROOT": notebook_root,
        "SHARED_DAQATHON_ROOT": shared_daqathon_root,
        "LOCAL_CACHE_DIR": local_cache_dir,
        "SHARED_CACHE_DIR": shared_cache_dir,
        "setup_jsonable": setup_jsonable,
        "show_setup_json": show_setup_json,
        "first_existing_path": first_existing_path,
        "first_existing_csv_dir": first_existing_csv_dir,
    }
