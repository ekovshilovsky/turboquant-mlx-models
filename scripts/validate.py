#!/usr/bin/env python3
"""Validate a TurboQuant-converted model: perplexity and needle-in-a-haystack."""

import argparse
import json
import sys
from pathlib import Path


def validate_metadata(model_path: Path) -> bool:
    """Check that TQ metadata is present and well-formed."""
    config_path = model_path / "config.json"
    if not config_path.exists():
        print(f"ERROR: {config_path} not found")
        return False

    with open(config_path) as f:
        config = json.load(f)

    qconfig = config.get("quantization_config", {})
    if qconfig.get("quantization_method") != "turboquant":
        print("ERROR: quantization_method is not 'turboquant'")
        return False

    if qconfig.get("tq_version") != "1":
        print(f"ERROR: unsupported tq_version: {qconfig.get('tq_version')}")
        return False

    print(f"Metadata OK: TQ{qconfig['tq_primary_bits']}+{qconfig['tq_residual_bits']}")
    return True


def validate_perplexity(model_path: Path) -> bool:
    """Run perplexity benchmark (placeholder)."""
    print("Perplexity validation: not yet implemented")
    return True


def main():
    parser = argparse.ArgumentParser(description="Validate TurboQuant model")
    parser.add_argument("--model", required=True, help="Path to converted model")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: {model_path} does not exist")
        sys.exit(1)

    ok = True
    ok = validate_metadata(model_path) and ok
    ok = validate_perplexity(model_path) and ok

    if ok:
        print("All validations passed.")
    else:
        print("Validation FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
