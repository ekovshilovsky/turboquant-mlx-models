---
license: mit
tags:
  - turboquant
  - mlx
  - apple-silicon
  - quantized
  - llama
---

# Llama-3.1-70B-TQ8

TurboQuant-compressed version of Llama 3.1 70B for Apple Silicon.

## Quantization Details

| Property | Value |
|---|---|
| Method | TurboQuant 4+4 residual |
| Effective bits/weight | 8 |
| Model size | ~70 GB |
| PPL vs fp16 | Pending benchmarks |

## Hardware Requirements

- Apple Silicon Mac with 96 GB+ unified memory
- Multi-Mac cluster recommended for fast inference
