"""Tests for model card template rendering."""

from pathlib import Path


def test_template_exists():
    """Model card template file should exist."""
    template = Path(__file__).parent.parent / "model-cards" / "template.md"
    assert template.exists()


def test_template_has_placeholders():
    """Template should contain replacement placeholders."""
    template = Path(__file__).parent.parent / "model-cards" / "template.md"
    content = template.read_text()
    assert "{MODEL_NAME}" in content
    assert "{PRIMARY_BITS}" in content
    assert "{SIZE_GB}" in content
