from __future__ import annotations

import distill_abm.compat as compat
import distill_abm.reference.compat as reference_compat


def test_reference_compat_module_reexports_compat_symbols() -> None:
    assert reference_compat.clean_context_response == compat.clean_context_response
    assert reference_compat.process_documentation == compat.process_documentation
    assert reference_compat.__doc__ is not None
