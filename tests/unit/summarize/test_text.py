from distill_abm.summarize.text import chunk_text, clean_markdown_symbols, strip_think_prefix


def test_strip_think_prefix() -> None:
    assert strip_think_prefix("<think>x</think>final") == "final"


def test_clean_markdown_symbols() -> None:
    assert clean_markdown_symbols("# Hello *world*") == "Hello world"


def test_chunk_text_max_chars() -> None:
    chunks = chunk_text("abcdefghijklmnopqrstuvwxyz", max_chars=10)
    assert chunks == ["abcdefghij", "klmnopqrst", "uvwxyz"]
