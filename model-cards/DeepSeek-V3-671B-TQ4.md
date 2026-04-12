---
license: mit
tags:
  - turboquant
  - mlx
  - apple-silicon
  - quantized
  - deepseek
  - moe
---

# DeepSeek-V3-671B-TQ4

TurboQuant-compressed version of DeepSeek-V3 671B for distributed Apple Silicon clusters.

## Quantization Details

| Property | Value |
|---|---|
| Method | TurboQuant 4-bit primary only |
| Effective bits/weight | 4 |
| Model size | ~335 GB |
| PPL vs fp16 | Pending benchmarks |

## Hardware Requirements

- Multi-Mac cluster required (4x M5 Max 128GB recommended)
- JACCL/RDMA via Thunderbolt 5 for optimal performance
