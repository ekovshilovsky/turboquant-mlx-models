---
license: mit
tags:
  - turboquant
  - mlx
  - apple-silicon
  - quantized
  - qwen
---

# Qwen3.5-27B-TQ8

TurboQuant-compressed version of Qwen3.5-27B for Apple Silicon inference.

## Quantization Details

| Property | Value |
|---|---|
| Method | TurboQuant 4+4 residual |
| Effective bits/weight | 8 |
| Model size | ~27 GB |
| PPL vs fp16 | Pending benchmarks |

## Hardware Requirements

- Apple Silicon Mac with 32 GB+ unified memory
- 128 GB recommended for 1M context
