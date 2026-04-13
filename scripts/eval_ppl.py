#!/usr/bin/env python3
"""Evaluate perplexity of TurboQuant-compressed models vs their fp16 originals.

Validates the quantization quality gate: PPL delta must be < 0.1%.

Workflow:
  1. Load the TQ model's safetensors, dequant all weight tensors back to fp16
  2. Write dequanted weights as standard safetensors (temp file)
  3. Load both original and dequanted models via mlx-lm
  4. Evaluate both on a standard text corpus
  5. Compare PPL values

Usage:
  python scripts/eval_ppl.py --tq-model ./converted/Qwen2.5-0.5B-TQ8 \
                              --original /tmp/Qwen2.5-0.5B \
                              --max-tokens 512
"""

import argparse
import json
import math
import sys
import tempfile
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn


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


def _hash_sign(seed: int, index: int) -> float:
    """Match the hash-based sign function used by rotation.cpp and Metal kernel."""
    h = (seed * 2654435761 + index * 2246822519) & 0xFFFFFFFF
    h ^= h >> 16
    h = (h * 0x45d9f3b) & 0xFFFFFFFF
    h ^= h >> 16
    return 1.0 if (h & 1) else -1.0


def _fwht_inplace(data: list, n: int) -> None:
    """In-place Fast Walsh-Hadamard Transform (butterfly)."""
    h = 1
    while h < n:
        i = 0
        while i < n:
            for j in range(i, i + h):
                a = data[j]
                b = data[j + h]
                data[j] = a + b
                data[j + h] = a - b
            i += h * 2
        h *= 2


def _dequant_layer(
    packed_primary: list, packed_residual_data, cb_p: list, cb_r: list,
    norms_list: list, seed_primary: int, seed_residual: int,
    out_features: int, in_features: int, block_size: int
) -> list:
    """Pure Python dequantization of a single layer."""
    inv_scale = 1.0 / math.sqrt(in_features)
    inv_sqrt_bs = 1.0 / math.sqrt(block_size)
    num_blocks = in_features // block_size

    signs_primary = [_hash_sign(seed_primary, i) for i in range(block_size)]
    signs_residual = [_hash_sign(seed_residual, i) for i in range(block_size)]

    has_residual = False
    if packed_residual_data is not None:
        for row in packed_residual_data:
            for v in row:
                if v != 0:
                    has_residual = True
                    break
            if has_residual:
                break

    weight = []
    for r in range(out_features):
        row = [0.0] * in_features

        for b in range(num_blocks):
            block = [0.0] * block_size
            for j in range(block_size):
                col = b * block_size + j
                byte_idx = col // 2
                if col % 2 == 0:
                    idx = packed_primary[r][byte_idx] & 0x0F
                else:
                    idx = (packed_primary[r][byte_idx] >> 4) & 0x0F
                block[j] = cb_p[idx]

            _fwht_inplace(block, block_size)
            for j in range(block_size):
                row[b * block_size + j] = block[j] * inv_scale * signs_primary[j] * inv_sqrt_bs

        if has_residual and packed_residual_data is not None:
            for b in range(num_blocks):
                block = [0.0] * block_size
                for j in range(block_size):
                    col = b * block_size + j
                    byte_idx = col // 2
                    if col % 2 == 0:
                        idx = packed_residual_data[r][byte_idx] & 0x0F
                    else:
                        idx = (packed_residual_data[r][byte_idx] >> 4) & 0x0F
                    block[j] = cb_r[idx]

                _fwht_inplace(block, block_size)
                for j in range(block_size):
                    row[b * block_size + j] += block[j] * inv_scale * signs_residual[j] * inv_sqrt_bs

        norm = norms_list[r]
        for j in range(in_features):
            row[j] *= norm

        weight.append(row)

    return weight


def dequant_tq_to_fp16(tq_model_path: str, output_path: str) -> None:
    """Dequant a TQ model's packed weights back to fp16 safetensors."""
    import shutil
    tq_path = Path(tq_model_path)
    out_path = Path(output_path)
    out_path.mkdir(parents=True, exist_ok=True)

    for f in tq_path.iterdir():
        if not f.name.endswith(".safetensors"):
            shutil.copy2(f, out_path / f.name)

    # Skip passthrough files in the outer loop — they're merged into the
    # main shard below. This avoids double-processing and file conflicts.
    for sf in sorted(tq_path.glob("*.safetensors")):
        if sf.name.endswith("_passthrough.safetensors"):
            continue

        tensors, metadata = mx.load(str(sf), return_metadata=True)

        if metadata.get("quantization_method") != "turboquant":
            shutil.copy2(sf, out_path / sf.name)
            continue

        cb_primary = tensors.get("tq_codebook_primary")
        cb_residual = tensors.get("tq_codebook_residual")
        if cb_primary is None:
            print(f"  WARNING: No codebook in {sf.name}, skipping")
            continue

        mx.eval(cb_primary)
        mx.eval(cb_residual)
        cb_p = cb_primary.tolist()
        cb_r = cb_residual.tolist() if cb_residual is not None else [0.0] * len(cb_p)

        layer_names = set()
        for name in tensors:
            if name.endswith(".packed_primary"):
                layer_names.add(name.replace(".packed_primary", ""))

        dequanted = {}
        for layer in sorted(layer_names):
            packed_p = tensors[f"{layer}.packed_primary"]
            packed_r = tensors.get(f"{layer}.packed_residual")
            norms = tensors[f"{layer}.norms"]
            seeds = tensors[f"{layer}.seeds"]

            mx.eval(packed_p, norms, seeds)
            if packed_r is not None:
                mx.eval(packed_r)

            out_features = packed_p.shape[0]
            packed_cols = packed_p.shape[1]
            in_features = packed_cols * 2

            seed_data = seeds.tolist()
            seed_primary = int(seed_data[0])
            seed_residual = int(seed_data[1]) if len(seed_data) > 1 else 0

            # Read block_size from seeds[2] if present (written by quantizer
            # since the seeds-stores-block_size change). Fall back to adaptive
            # computation for legacy models that only stored 2 seed values.
            if len(seed_data) >= 3:
                block_size = int(seed_data[2])
            else:
                block_size = 1
                while block_size * 2 <= in_features and block_size < 512:
                    block_size *= 2

            pp_data = packed_p.astype(mx.uint8).tolist()
            pr_data = packed_r.astype(mx.uint8).tolist() if packed_r is not None else None
            norms_list = norms.tolist()

            weight = _dequant_layer(
                pp_data, pr_data, cb_p, cb_r, norms_list,
                seed_primary, seed_residual,
                out_features, in_features, block_size
            )
            dequanted[f"{layer}.weight"] = mx.array(weight).astype(mx.float16)

        pt_path = tq_path / sf.name.replace(".safetensors", "_passthrough.safetensors")
        if pt_path.exists():
            pt_tensors, _ = mx.load(str(pt_path), return_metadata=True)
            for name, tensor in pt_tensors.items():
                dequanted[name] = tensor

        mx.save_safetensors(str(out_path / sf.name), dequanted, metadata={})
        print(f"  Dequanted {len(layer_names)} layers from {sf.name}")


# Standard evaluation corpus — first paragraphs of Wikipedia articles
# covering diverse topics for representative perplexity measurement.
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

    # Dequant TQ model and evaluate
    print(f"\nDequanting TQ model: {args.tq_model}")
    with tempfile.TemporaryDirectory() as tmpdir:
        dequant_path = Path(tmpdir) / "dequanted"
        t0 = time.time()
        dequant_tq_to_fp16(args.tq_model, str(dequant_path))
        print(f"  Dequanted in {time.time()-t0:.1f}s")

        print(f"Loading dequanted TQ model...")
        t0 = time.time()
        model_tq, _ = load(str(dequant_path))
        print(f"  Loaded in {time.time()-t0:.1f}s")

        print(f"Computing TQ PPL (max_tokens={args.max_tokens})...")
        ppl_tq = compute_ppl(model_tq, tokenizer, eval_text, args.max_tokens)
        print(f"  TQ PPL: {ppl_tq:.4f}")

    delta = abs(ppl_tq - ppl_fp16) / ppl_fp16 * 100
    print(f"\n{'='*50}")
    print(f"fp16 PPL:  {ppl_fp16:.4f}")
    print(f"TQ8  PPL:  {ppl_tq:.4f}")
    print(f"Delta:     {delta:.4f}%")
    print(f"Target:    < 0.1%")
    print(f"Result:    {'PASS' if delta < 0.1 else 'FAIL'}")
    print(f"{'='*50}")

    if delta >= 0.1:
        sys.exit(1)


if __name__ == "__main__":
    main()
