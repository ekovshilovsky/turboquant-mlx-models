#!/usr/bin/env python3
"""Convert a HuggingFace model to TurboQuant format.

Downloads the model from HuggingFace Hub (or uses a local path),
converts to safetensors if needed, then runs tq-convert.

Usage:
    python scripts/convert_from_hf.py Qwen/Qwen2.5-Coder-3B --output ./converted/Qwen2.5-Coder-3B-TQ8
    python scripts/convert_from_hf.py ./local/model/path --bits 4 --residual-bits 4
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def find_tq_convert():
    """Locate the tq-convert binary."""
    candidates = [
        Path(__file__).parent.parent.parent / "turboquant-mlx-core" / "build" / "tq-convert",
        Path.home() / "Code" / "turboquant-mlx-core" / "build" / "tq-convert",
        Path("/usr/local/bin/tq-convert"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    # Try PATH
    import shutil
    found = shutil.which("tq-convert")
    if found:
        return found
    print("ERROR: tq-convert not found. Build turboquant-mlx-core first.")
    sys.exit(1)


def download_from_hf(model_name: str, local_dir: str) -> str:
    """Download a model from HuggingFace Hub."""
    from huggingface_hub import snapshot_download
    print(f"Downloading {model_name} from HuggingFace Hub...")
    path = snapshot_download(
        model_name,
        local_dir=local_dir,
        ignore_patterns=["*.gguf", "*.bin", "*.pt", "*.onnx", "original/*"],
    )
    return path


def convert_gguf_to_safetensors(gguf_path: str, output_dir: str) -> str:
    """Convert a GGUF file to fp16 safetensors via mlx-lm.

    NOTE: Ollama GGUF models are already quantized (Q4_K_M, Q8_0, etc.).
    Dequantizing and re-quantizing with TurboQuant introduces double
    quantization error. For best quality, download the original fp16
    source from HuggingFace instead of converting from GGUF.
    """
    try:
        from mlx_lm import convert as mlx_convert
    except ImportError:
        print("ERROR: mlx-lm not installed. Run: uv pip install mlx-lm")
        sys.exit(1)

    print(f"Converting GGUF to safetensors via mlx-lm: {gguf_path}")
    print("WARNING: GGUF source is already quantized. For best quality,")
    print("         use the fp16 HuggingFace source instead.")
    mlx_convert(gguf_path, mlx_path=output_dir, dequantize=True)
    return output_dir


# Map Ollama model names to their HuggingFace fp16 source repositories.
# Using the original fp16 weights avoids double quantization error from
# dequantizing Ollama's GGUF (Q4_K_M/Q8_0) before TurboQuant compression.
OLLAMA_TO_HF = {
    "qwen2.5-coder:3b": "Qwen/Qwen2.5-Coder-3B",
    "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B",
    "gemma3:4b":         "google/gemma-3-4b-pt",
    "gemma3:27b":        "google/gemma-3-27b-pt",
    "qwen3:32b":         "Qwen/Qwen3-32B",
}


def resolve_ollama_model(model_name: str) -> str | None:
    """Resolve an Ollama model name to a HuggingFace repo for fp16 download."""
    hf_name = OLLAMA_TO_HF.get(model_name)
    if hf_name:
        print(f"Ollama model '{model_name}' -> HuggingFace '{hf_name}' (fp16 source)")
        return hf_name
    print(f"WARNING: No HuggingFace mapping for Ollama model '{model_name}'")
    print(f"Known models: {', '.join(OLLAMA_TO_HF.keys())}")
    return None


def main():
    parser = argparse.ArgumentParser(description="Convert model to TurboQuant format")
    parser.add_argument("model", help="HuggingFace model name, local path, or Ollama model (e.g., qwen3:32b)")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--bits", type=int, default=4, help="Primary quantization bits (default: 4)")
    parser.add_argument("--residual-bits", type=int, default=4, help="Residual bits (default: 4)")
    parser.add_argument("--block-size", type=int, default=512, help="WHT block size (default: 512)")
    parser.add_argument("--cache-dir", default="/tmp/tq-convert-cache", help="Cache for downloaded models")
    args = parser.parse_args()

    tq_convert = find_tq_convert()
    model_path = args.model

    # Determine source type and resolve to a safetensors directory
    if ":" in model_path and not os.path.exists(model_path):
        # Ollama model name (e.g., "qwen3:32b") — resolve to HuggingFace fp16 source
        hf_name = resolve_ollama_model(model_path)
        if hf_name:
            local_dir = os.path.join(args.cache_dir, hf_name.replace("/", "_"))
            if not os.path.exists(local_dir) or not any(Path(local_dir).glob("*.safetensors")):
                model_path = download_from_hf(hf_name, local_dir)
            else:
                model_path = local_dir
                print(f"Using cached: {model_path}")
        else:
            print(f"ERROR: Cannot resolve Ollama model '{model_path}'")
            sys.exit(1)
    elif "/" in model_path and not os.path.exists(model_path):
        # HuggingFace model name (e.g., "Qwen/Qwen2.5-Coder-3B")
        local_dir = os.path.join(args.cache_dir, model_path.replace("/", "_"))
        if not os.path.exists(local_dir) or not any(Path(local_dir).glob("*.safetensors")):
            model_path = download_from_hf(model_path, local_dir)
        else:
            model_path = local_dir
            print(f"Using cached: {model_path}")
    # else: local path, use directly

    # Verify safetensors exist
    st_files = list(Path(model_path).glob("*.safetensors"))
    if not st_files:
        print(f"ERROR: No .safetensors files found in {model_path}")
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        base = Path(model_path).name.replace("_safetensors", "")
        output_path = f"./converted/{base}-TQ{args.bits}"

    # Run tq-convert
    cmd = [
        tq_convert,
        "--model", model_path,
        "--output", output_path,
        "--bits", str(args.bits),
        "--residual-bits", str(args.residual_bits),
        "--block-size", str(args.block_size),
    ]
    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd)

    if result.returncode != 0:
        print(f"\nERROR: tq-convert failed with exit code {result.returncode}")
        sys.exit(1)

    print(f"\nConverted model saved to: {output_path}")

    # Print size comparison
    orig_size = sum(f.stat().st_size for f in Path(model_path).rglob("*.safetensors"))
    conv_size = sum(f.stat().st_size for f in Path(output_path).rglob("*.safetensors"))
    if orig_size > 0:
        ratio = conv_size / orig_size * 100
        print(f"Size: {orig_size/(1024**3):.1f} GB -> {conv_size/(1024**3):.1f} GB ({ratio:.0f}%)")


if __name__ == "__main__":
    main()
