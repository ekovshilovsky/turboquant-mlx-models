# Benchmark Results

All measurements on M5 Max 128GB, macOS 26.4, MLX 0.31.1.

## Weight Compression Quality (PPL Delta vs fp16)

### Baseline (shared codebook, no sensitive layers)

| Model | Type | fp16 PPL | TQ8 PPL | Delta |
|---|---|---|---|---|
| Qwen2.5-0.5B | Dense base | 1.72 | 1.80 | 4.86% |
| Qwen2.5-7B | Dense base | 1.47 | 1.53 | 3.86% |
| Qwen2.5-Coder-7B | Coder | 1.46 | 1.47 | 0.63% |
| Qwen2.5-32B | Dense base | 1.41 | 1.44 | 1.75% |
| Qwen3.5-27B | Hybrid | 1.45 | 1.45 | 0.26% |

### Optimized (T1: sensitive-4 + T4: per-layer codebooks)

| Model | Type | fp16 PPL | TQ8 PPL | Delta | Improvement |
|---|---|---|---|---|---|
| Qwen2.5-32B | Dense base | 1.41 | 1.42 | 0.09% | 19.4x better |
| Qwen3.5-27B | Hybrid | 1.45 | 1.46 | 0.18% | 1.4x better |

### Technique Ablation (Qwen2.5-32B)

| Config | Delta | Size | Convert Time |
|---|---|---|---|
| Baseline 4+4 | 1.75% | 31 GB | 7 min |
| + T1 (sensitive layers 4) | 1.22% | 34 GB | 6 min |
| + T1+T2 (norm refinement) | 1.23% | 34 GB | 7 min |
| + T1+T3 (5+3 asymmetric) | 5.54% | 48 GB | 7 min |
| + T1+T4 (per-layer codebooks) | 0.09% | 34 GB | 239 min |

T3 (5+3 asymmetric) rejected: worse quality AND larger model.
T2 (norm refinement) negligible: norms were already correct.
T4 (per-layer codebooks) is the game changer: 1.22% → 0.09%.

## Inference Speed (4096x4096, batch=1)

| Implementation | Time/token | Throughput | vs float32 |
|---|---|---|---|
| float32 matmul | 0.28 ms | 2788 tok/s | 1.0x |
| MLX q8_0 (affine, group=64) | 0.21 ms | 4738 tok/s | 0.75x |
| TQ8 pre-optimization | 2.84 ms | 352 tok/s | 6.77x |
| TQ8 post-optimization | 1.81 ms | 551 tok/s | 5.35x |

### Speed Optimization Techniques

| Technique | Impact |
|---|---|
| Shared rotation (single WHT) | ~40% speedup (biggest win) |
| Compiled metallib dispatch | Eliminates JIT overhead |
| Simdgroup shuffles (WHT stages 0-4) | 70% throughput on real layers |
| 256-entry LUT (combined codebook) | Eliminates per-element codebook lookups |

### Real Model E2E (Qwen2.5-Coder-3B, 151936x2048 layer)

| Stage | Time/token | Throughput |
|---|---|---|
| Pre-optimization | 39.8 ms | 25.1 tok/s |
| Post-optimization | 21.6 ms | 46.2 tok/s |

## Compression Ratios

Consistent ~50% across all model sizes (8 effective bits from 16-bit source):

| Model | fp16 | TQ8 | Ratio |
|---|---|---|---|
| Qwen2.5-0.5B | 942 MB | 484 MB | 51% |
| Qwen2.5-Coder-3B | 5.8 GB | 2.9 GB | 50% |
| Qwen2.5-Coder-7B | 14 GB | 7.1 GB | 51% |
| Qwen2.5-32B | 61 GB | 34 GB | 56% (sensitive layers add ~3 GB) |
| Qwen3.5-27B | 52 GB | 29 GB | 56% (sensitive layers add ~3 GB) |
