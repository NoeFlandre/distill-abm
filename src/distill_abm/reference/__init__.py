"""Reference-compatibility namespace.

This package preserves import paths for callers expecting the older
``distill_abm.reference`` surface.
"""

from distill_abm.compat import *  # noqa: F403

__all__ = [name for name in dir() if not name.startswith("_")]
