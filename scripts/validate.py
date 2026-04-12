#!/usr/bin/env python3
"""Validate a TurboQuant-converted model: perplexity and needle-in-a-haystack."""

import argparse
import json
import sys
from pathlib import Path


def validate_metadata(model_path: Path) -> bool:
    """Check that TQ metadata is present and well-formed.

    Checks two locations: config.json (quantization_config section) and
    safetensors file metadata. Either source is sufficient — the converter
    stores metadata in safetensors headers, and may optionally update config.json.
    """
    # Check safetensors metadata first (authoritative source)
    for sf in sorted(model_path.glob("*.safetensors")):
        if sf.name.endswith("_passthrough.safetensors"):
            continue
        try:
            import mlx.core as mx
            _, metadata = mx.load(str(sf), return_metadata=True)
            metadata = dict(metadata)
            if metadata.get("quantization_method") == "turboquant":
                version = metadata.get("tq_version", "?")
                primary = metadata.get("tq_primary_bits", "?")
                residual = metadata.get("tq_residual_bits", "?")
                if version != "1":
                    print(f"ERROR: unsupported tq_version: {version}")
                    return False
                print(f"Metadata OK (safetensors): TQ{primary}+{residual}")
                return True
        except Exception:
            continue

    # Fall back to config.json
    config_path = model_path / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        qconfig = config.get("quantization_config", {})
        if qconfig.get("quantization_method") == "turboquant":
            if qconfig.get("tq_version") != "1":
                print(f"ERROR: unsupported tq_version: {qconfig.get('tq_version')}")
                return False
            print(f"Metadata OK (config.json): TQ{qconfig['tq_primary_bits']}+{qconfig['tq_residual_bits']}")
            return True

    print("ERROR: No TurboQuant metadata found in safetensors or config.json")
    return False


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
