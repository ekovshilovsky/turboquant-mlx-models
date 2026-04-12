"""Tests for the validation script."""

import json
import tempfile
from pathlib import Path


def test_validate_metadata_valid():
    """Valid TQ config.json should pass validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {
            "model_type": "test",
            "quantization_config": {
                "quantization_method": "turboquant",
                "tq_version": "1",
                "tq_primary_bits": 4,
                "tq_residual_bits": 4,
            }
        }
        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(json.dumps(config))

        # Import and test
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from validate import validate_metadata
        assert validate_metadata(Path(tmpdir)) is True


def test_validate_metadata_missing_method():
    """Config without turboquant method should fail."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = {"model_type": "test"}
        config_path = Path(tmpdir) / "config.json"
        config_path.write_text(json.dumps(config))

        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from validate import validate_metadata
        assert validate_metadata(Path(tmpdir)) is False
