from __future__ import annotations

import distill_abm.compat as compat
import distill_abm.reference as reference


def test_reference_package_reexports_compat_surface() -> None:
    assert hasattr(reference, "append_to_csv")
    assert hasattr(reference, "process_documentation")
    assert hasattr(reference, "create_collage")
    assert set(("append_to_csv", "process_documentation", "create_collage")) <= set(reference.__all__)
    assert reference.append_to_csv == compat.append_to_csv
    assert reference.create_collage == compat.create_collage
