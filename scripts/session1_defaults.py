"""Compatibility shim for Session 1 defaults.

Participant-editable defaults now live in ``workshop_config/session1_defaults.py``.
This module remains so older scripts that import ``scripts.session1_defaults``
continue to work.
"""

from __future__ import annotations

from workshop_config.session1_defaults import *  # noqa: F401,F403

