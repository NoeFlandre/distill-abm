"""Backward-compatible compatibility namespace.

This package preserves historical import paths for external callers while the
canonical location is now ``distill_abm.compat``.
"""

from distill_abm.compat import *  # noqa: F403

__all__ = [name for name in dir() if not name.startswith("_")]
