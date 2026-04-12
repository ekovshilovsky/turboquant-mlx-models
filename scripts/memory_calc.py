#!/usr/bin/env python3
"""Standalone memory budget calculator for TurboQuant models."""

import argparse


def calculate_memory(
    params_b: float,
    bits: int,
    context: int,
    layers: int,
    heads: int,
    head_dim: int,
    kv_bits: int,
    window: int,
) -> dict:
    weight_gb = params_b * bits / 8
    kv_gb = context * layers * 2 * heads * head_dim * kv_bits / 8 / 1e9
    window_gb = window * layers * 2 * heads * head_dim * 2 / 1e9
    overhead_gb = 5.0
    total_gb = weight_gb + kv_gb + window_gb + overhead_gb
    return {
        "weights": weight_gb,
        "kv_cache": kv_gb,
        "decode_window": window_gb,
        "overhead": overhead_gb,
        "total": total_gb,
    }


def main():
    parser = argparse.ArgumentParser(description="TurboQuant memory calculator")
    parser.add_argument("--params", type=float, required=True, help="Model params in billions")
    parser.add_argument("--bits", type=int, default=8, help="Bits per weight")
    parser.add_argument("--context", type=int, default=1_000_000, help="Context length")
    parser.add_argument("--layers", type=int, required=True)
    parser.add_argument("--heads", type=int, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--kv-bits", type=int, default=4)
    parser.add_argument("--window", type=int, default=131_072)
    parser.add_argument("--device-memory", type=float, default=128.0, help="Device memory in GB")
    args = parser.parse_args()

    mem = calculate_memory(
        args.params, args.bits, args.context,
        args.layers, args.heads, args.head_dim,
        args.kv_bits, args.window,
    )

    print(f"Model weights ({args.bits}-bit):          {mem['weights']:>8.1f} GB")
    print(f"KV cache ({args.kv_bits}-bit, {args.context:,} ctx): {mem['kv_cache']:>8.1f} GB")
    print(f"Decode window (fp16, {args.window:,}):  {mem['decode_window']:>8.1f} GB")
    print(f"Activations + overhead:          {mem['overhead']:>8.1f} GB")
    print(f"{'─' * 45}")
    print(f"Total estimated:                 {mem['total']:>8.1f} GB")
    print(f"Device memory:                   {args.device_memory:>8.1f} GB")
    print()
    if mem["total"] <= args.device_memory:
        headroom = (1 - mem["total"] / args.device_memory) * 100
        print(f"FITS ({headroom:.0f}% headroom)")
    else:
        deficit = mem["total"] - args.device_memory
        print(f"DOES NOT FIT (need {deficit:.1f} GB more)")


if __name__ == "__main__":
    main()
