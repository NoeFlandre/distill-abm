from __future__ import annotations

import distill_abm.compat.reference_loader as compat_notebook_loader
import distill_abm.reference.notebook_loader as reference_notebook_loader


def test_reference_notebook_loader_module_is_compat_alias() -> None:
    assert reference_notebook_loader.should_dispatch_notebook == compat_notebook_loader.should_dispatch_notebook
    assert reference_notebook_loader.__doc__ is not None
