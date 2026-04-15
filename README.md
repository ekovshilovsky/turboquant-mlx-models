# TurboQuant-MLX Models

Pre-converted TurboQuant-compressed model weights for [turboquant-mlx-core](https://github.com/ekovshilovsky/turboquant-mlx-core).

## Available Models

| Model | Architecture | fp16 Size | TQ8 Size | PPL Delta | Status |
|---|---|---|---|---|---|
| Qwen3.5-27B-TQ8 | Hybrid (linear + full attn) | 52 GB | 26 GB | 0.18% | Validated |
| Qwen2.5-32B-TQ8 | Dense base | 61 GB | 34 GB | 0.09% | Validated |

## Validated PPL Scaling Data

Full perplexity measurements across all tested models (short eval corpus):

| Model | Type | fp16 PPL | TQ8 PPL | Delta | Config |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | Dense base | 1.72 | 1.80 | 4.86% | Baseline 4+4 |
| Qwen2.5-7B | Dense base | 1.47 | 1.53 | 3.86% | Baseline 4+4 |
| Qwen2.5-Coder-7B | Coder | 1.46 | 1.47 | 0.63% | Baseline 4+4 |
| Qwen2.5-32B | Dense base | 1.41 | 1.44 | 1.75% | Baseline 4+4 |
| **Qwen2.5-32B** | **Dense base** | **1.41** | **1.42** | **0.09%** | **T1+T4 optimized** |
| Qwen3.5-27B | Hybrid | 1.45 | 1.45 | 0.26% | Baseline 4+4 |
| **Qwen3.5-27B** | **Hybrid** | **1.45** | **1.46** | **0.18%** | **T1+T4 optimized** |

Key findings:
- Dense base models: 4.86% at 0.5B → 0.09% at 32B with per-layer codebooks
- Hybrid architectures quantize 6x better than dense at same scale
- Coder/instruct models quantize significantly better than base models

## Optimal Conversion Config

```bash
tq-convert --model /path/to/model \
  --sensitive-layers 4 \
  --per-layer-codebooks \
  --bits 4 --residual-bits 4
```

- `--sensitive-layers 4`: Keep first/last 4 layers at fp16 (30% delta reduction)
- `--per-layer-codebooks`: Fit codebooks to each layer's distribution (biggest quality win)
- Shared rotation enabled by default (single WHT pass at inference)

Conversion time: ~4 hours for 27-32B models (per-layer Lloyd-Max fitting).
Runtime inference: identical speed regardless of codebook type.

## Deferred Models

| Model | Reason | Status |
|---|---|---|
| Qwen3.5-35B-A3B (MoE) | Expert routing architecture needs dedicated support | Planned |
| Llama-3.1-70B | Requires distributed inference for validation (>128GB) | Planned |
| DeepSeek-V3-671B | Cluster-only (4x M5 Max minimum) | Planned |

## Convert Your Own

### From HuggingFace

```bash
# Setup
pip install huggingface_hub mlx-lm

# Convert with optimal settings
python scripts/convert_from_hf.py Qwen/Qwen3.5-27B \
  --output ./converted/Qwen3.5-27B-TQ8

# From Ollama model name (downloads fp16 from HuggingFace)
python scripts/convert_from_hf.py qwen2.5-coder:3b
```

### Direct conversion (local safetensors)

```bash
# Build turboquant-mlx-core
brew install cmake mlx
git clone https://github.com/ekovshilovsky/turboquant-mlx-core
cd turboquant-mlx-core && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# Convert with full optimizations
./build/tq-convert --model /path/to/model \
  --sensitive-layers 4 --per-layer-codebooks \
  --bits 4 --residual-bits 4

# Validate
python scripts/validate.py --model /path/to/model-tq8
```

### PPL evaluation

```bash
# Dequant to fp16 for PPL comparison via mlx-lm
./build/tq-dequant /path/to/tq-model /path/to/dequanted
python scripts/eval_ppl.py --tq-model /path/to/tq-model --original /path/to/fp16-model
```

### Supported Ollama models

The converter maps Ollama model names to their HuggingFace fp16 source
repositories. This avoids double quantization error from converting
Ollama's pre-quantized GGUF files.

| Ollama | HuggingFace Source |
|---|---|
| `qwen2.5-coder:3b` | `Qwen/Qwen2.5-Coder-3B` |
| `qwen2.5-coder:7b` | `Qwen/Qwen2.5-Coder-7B` |
| `gemma3:4b` | `google/gemma-3-4b-pt` |
| `gemma3:27b` | `google/gemma-3-27b-pt` |
| `qwen3:32b` | `Qwen/Qwen3-32B` |

## License

MIT License - Copyright (c) 2026 Eugene Kovshilovsky
