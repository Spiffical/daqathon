#!/usr/bin/env python3
"""Command-line entry point for building ONC scalar parquet caches.

This wrapper gives the reusable ONC cache pipeline a descriptive public name.
New code should prefer this filename or the functions in
``onc_scalar_cache_utils.py``.
"""

from __future__ import annotations

try:  # pragma: no cover - supports package imports
    from .prepare_scalar_session1_data import *  # noqa: F401,F403
    from .prepare_scalar_session1_data import main
except ImportError:  # pragma: no cover - supports direct CLI execution
    from prepare_scalar_session1_data import *  # type: ignore # noqa: F401,F403
    from prepare_scalar_session1_data import main  # type: ignore


if __name__ == "__main__":
    main()
