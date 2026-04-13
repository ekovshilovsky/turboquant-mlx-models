#!/usr/bin/env python3
"""Evaluate perplexity of TurboQuant-compressed models vs their fp16 originals.

Validates the quantization quality gate: PPL delta should decrease with model
size (0.5B: ~5%, 3B: ~2%, 7B: ~1%, 27B: <0.1%).

Pipeline:
  1. Dequant TQ model to fp16 safetensors using the C++ dequant tool (ground truth)
  2. Load both original and dequanted models via mlx-lm
  3. Evaluate both on a standard text corpus
  4. Compare PPL values

The C++ dequant is the single source of truth for weight reconstruction.
No parallel Python dequant implementation — that path had bugs and would
need to be kept in sync with C++ changes (sign generation, block_size, etc.).

Usage:
  python scripts/eval_ppl.py --tq-model ./converted/Qwen2.5-0.5B-TQ8 \
                              --original /tmp/Qwen2.5-0.5B \
                              --max-tokens 512
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


def find_cpp_dequant_tool():
    """Locate the C++ tq-dequant binary (built from tq_dequant_model.cpp)."""
    candidates = [
        Path(__file__).parent.parent.parent / "turboquant-mlx-core" / "build" / "tq-dequant",
        Path.home() / "Code" / "turboquant-mlx-core" / "build" / "tq-dequant",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    found = shutil.which("tq-dequant")
    if found:
        return found
    return None


def dequant_via_cpp(tq_model_path: str, output_path: str) -> bool:
    """Dequant TQ model to fp16 safetensors using the C++ tool.

    Falls back to building the tool from source if not found.
    """
    tool = find_cpp_dequant_tool()
    if tool:
        result = subprocess.run([tool, tq_model_path, output_path],
                                capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
            return True
        print(f"tq-dequant failed: {result.stderr}")
        return False

    # Tool not built yet — build it inline
    core_dir = Path(__file__).parent.parent.parent / "turboquant-mlx-core"
    if not core_dir.exists():
        core_dir = Path.home() / "Code" / "turboquant-mlx-core"

    build_dir = core_dir / "build"
    if not (build_dir / "libturboquant_mlx.dylib").exists():
        print("ERROR: turboquant-mlx-core not built. Run cmake --build build first.")
        return False

    # Build the dequant tool from inline source
    tool_src = core_dir / "tools" / "tq_dequant_model.cpp"
    if not tool_src.exists():
        print(f"ERROR: {tool_src} not found")
        return False

    tool_bin = build_dir / "tq-dequant"
    compile_cmd = [
        "c++", "-std=c++17", "-O2",
        "-I", str(core_dir / "include"),
        "-I", "/opt/homebrew/include",
        "-L", str(build_dir),
        "-L", "/opt/homebrew/lib",
        "-lturboquant_mlx", "-lmlx",
        "-framework", "Metal", "-framework", "Accelerate", "-framework", "Foundation",
        f"-Wl,-rpath,{build_dir}", "-Wl,-rpath,/opt/homebrew/lib",
        "-o", str(tool_bin),
        str(tool_src),
    ]
    result = subprocess.run(compile_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Build failed: {result.stderr}")
        return False

    return dequant_via_cpp(tq_model_path, output_path)


def compute_ppl(model, tokenizer, text: str, max_tokens: int = 512) -> float:
    """Compute perplexity of a model on a text string."""
    tokens = tokenizer.encode(text)
    if len(tokens) > max_tokens + 1:
        tokens = tokens[:max_tokens + 1]

    input_ids = mx.array([tokens[:-1]])
    targets = mx.array([tokens[1:]])
    mx.eval(input_ids)

    logits = model(input_ids)
    mx.eval(logits)

    loss = nn.losses.cross_entropy(logits, targets, reduction="mean")
    mx.eval(loss)
    return float(mx.exp(loss).item())


EVAL_CORPUS = (
    "The tower is 324 metres tall, about the same height as an 81-storey building, "
    "and the tallest structure in Paris. Its base is square, measuring 125 metres on "
    "each side. During its construction, the Eiffel Tower surpassed the Washington "
    "Monument to become the tallest man-made structure in the world, a title it held "
    "for 41 years until the Chrysler Building in New York City was finished in 1930. "
    "It was the first structure to reach a height of 300 metres. Due to the addition "
    "of a broadcasting aerial at the top of the tower in 1957, it is now taller than "
    "the Chrysler Building by 5.2 metres. Excluding transmitters, the Eiffel Tower is "
    "the second tallest free-standing structure in France after the Millau Viaduct. "
    "The tower has three levels for visitors, with restaurants on the first and second "
    "levels. The top level's upper platform is 276 m above the ground, the highest "
    "observation deck accessible to the public in the European Union. Tickets can be "
    "purchased to ascend by stairs or lift to the first and second levels. The climb "
    "from ground level to the first level is over 300 steps, as is the climb from the "
    "first level to the second. Although there is a staircase to the top level, it is "
    "usually accessible only by lift. "
    "Machine learning is a subset of artificial intelligence that provides systems the "
    "ability to automatically learn and improve from experience without being explicitly "
    "programmed. Machine learning focuses on the development of computer programs that "
    "can access data and use it to learn for themselves. The process of learning begins "
    "with observations or data, such as examples, direct experience, or instruction, in "
    "order to look for patterns in data and make better decisions in the future based on "
    "the examples that we provide. The primary aim is to allow the computers to learn "
    "automatically without human intervention or assistance and adjust actions accordingly. "
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate TQ model perplexity vs fp16 original")
    parser.add_argument("--tq-model", required=True, help="Path to TQ-converted model")
    parser.add_argument("--original", required=True, help="Path to original fp16 model")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for evaluation")
    parser.add_argument("--dataset", default=None, help="Path to text file for evaluation")
    args = parser.parse_args()

    from mlx_lm import load

    eval_text = EVAL_CORPUS
    if args.dataset:
        with open(args.dataset) as f:
            eval_text = f.read()

    # Evaluate original fp16 model
    print(f"Loading original fp16 model: {args.original}")
    t0 = time.time()
    model_fp16, tokenizer = load(args.original)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print(f"Computing fp16 PPL (max_tokens={args.max_tokens})...")
    ppl_fp16 = compute_ppl(model_fp16, tokenizer, eval_text, args.max_tokens)
    print(f"  fp16 PPL: {ppl_fp16:.4f}")
    del model_fp16

    # Dequant TQ model via C++ (ground truth) and evaluate
    print(f"\nDequanting TQ model via C++: {args.tq_model}")
    with tempfile.TemporaryDirectory() as tmpdir:
        dequant_path = os.path.join(tmpdir, "dequanted")
        t0 = time.time()
        if not dequant_via_cpp(args.tq_model, dequant_path):
            print("ERROR: C++ dequant failed")
            sys.exit(1)
        print(f"  Dequanted in {time.time()-t0:.1f}s")

        print(f"Loading dequanted TQ model...")
        t0 = time.time()
        model_tq, _ = load(dequant_path)
        print(f"  Loaded in {time.time()-t0:.1f}s")

        print(f"Computing TQ PPL (max_tokens={args.max_tokens})...")
        ppl_tq = compute_ppl(model_tq, tokenizer, eval_text, args.max_tokens)
        print(f"  TQ PPL: {ppl_tq:.4f}")

    delta = abs(ppl_tq - ppl_fp16) / ppl_fp16 * 100
    print(f"\n{'='*50}")
    print(f"fp16 PPL:  {ppl_fp16:.4f}")
    print(f"TQ8  PPL:  {ppl_tq:.4f}")
    print(f"Delta:     {delta:.4f}%")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
