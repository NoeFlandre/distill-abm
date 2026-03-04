"""Reference-compatibility namespace.

This package preserves import paths for callers expecting the older
``distill_abm.reference`` surface.
"""

from typing import TYPE_CHECKING

from distill_abm.compat import *  # noqa: F403
from distill_abm.compat.compat_io import append_to_csv, create_collage, process_documentation
from distill_abm.compat.compat_text import clean_context_response

if TYPE_CHECKING:
    from distill_abm.compat.compat_io import append_to_csv, create_collage, process_documentation
    from distill_abm.compat.compat_text import clean_context_response

__all__ = [
    "append_to_csv",
    "clean_context_response",
    "create_collage",
    "process_documentation",
]

_reference_exports = (append_to_csv, create_collage, clean_context_response, process_documentation)

__all__ = [name for name in dir() if not name.startswith("_")]
