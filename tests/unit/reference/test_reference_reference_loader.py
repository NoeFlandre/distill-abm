from __future__ import annotations

import distill_abm.compat.reference_loader as compat_loader
import distill_abm.reference.reference_loader as reference_loader


def test_reference_reference_loader_module_is_compat_alias() -> None:
    assert reference_loader.get_notebook_function == compat_loader.get_notebook_function
    assert reference_loader.should_dispatch_notebook == compat_loader.should_dispatch_notebook
    assert reference_loader.REQUIRED_NOTEBOOK_FUNCTIONS == compat_loader.REQUIRED_NOTEBOOK_FUNCTIONS
