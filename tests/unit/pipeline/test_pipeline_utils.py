"""Tests for internal pipeline utility helpers."""

from pathlib import Path

from distill_abm.pipeline import helpers


def test_slug_normalizes_case_and_symbols() -> None:
    """Test that _slug correctly normalizes strings for artifact naming."""
    assert helpers._slug("My Metric Name") == "my_metric_name"
    assert helpers._slug("Metric/With/Slashes") == "metric_with_slashes"
    assert helpers._slug("  Spaces  ") == "spaces"


def test_append_plot_description_handles_empty_input() -> None:
    """Test that append_plot_description returns base prompt when description is empty."""
    base = "Base prompt"
    assert helpers.append_plot_description(base, "") == base
    assert helpers.append_plot_description(base, "  ") == base


def test_append_plot_description_appends_correctly() -> None:
    """Test that append_plot_description appends non-empty descriptions with double newline."""
    base = "Base prompt"
    desc = "Plot description"
    assert helpers.append_plot_description(base, desc) == "Base prompt\n\nPlot description"


def test_encode_image_returns_base64_string(tmp_path: Path) -> None:
    """Test that encode_image correctly base64 encodes a file."""
    image_file = tmp_path / "test.png"
    # Smallest valid PNG-like byte sequence for testing
    data = b"\x89PNG\r\n\x1a\n"
    image_file.write_bytes(data)

    encoded = helpers.encode_image(image_file)
    import base64

    assert encoded == base64.b64encode(data).decode("utf-8")
