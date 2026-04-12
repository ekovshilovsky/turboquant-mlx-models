# TurboQuant-MLX Models

Pre-converted TurboQuant-compressed model weights for [turboquant-mlx-core](https://github.com/ekovshilovsky/turboquant-mlx-core).

## Available Models

| Model | Format | Size | PPL vs fp16 | HuggingFace |
|---|---|---|---|---|
| Qwen3.5-27B-TQ8 | 4+4 residual | ~27 GB | TBD | [link](https://huggingface.co/ekovshilovsky/Qwen3.5-27B-TQ8) |
| Qwen3.5-35B-A3B-TQ8 | 4+4 residual | ~35 GB | TBD | [link](https://huggingface.co/ekovshilovsky/Qwen3.5-35B-A3B-TQ8) |
| Llama-3.1-70B-TQ8 | 4+4 residual | ~70 GB | TBD | [link](https://huggingface.co/ekovshilovsky/Llama-3.1-70B-TQ8) |
| DeepSeek-V3-671B-TQ4 | 4-bit primary | ~335 GB | TBD | [link](https://huggingface.co/ekovshilovsky/DeepSeek-V3-671B-TQ4) |

## Convert Your Own

### From HuggingFace

```bash
# Setup
pip install huggingface_hub mlx-lm

# Convert from HuggingFace model name
python scripts/convert_from_hf.py Qwen/Qwen2.5-Coder-3B

# Convert from Ollama model name (downloads fp16 from HuggingFace)
python scripts/convert_from_hf.py qwen2.5-coder:3b

# Custom output and settings
python scripts/convert_from_hf.py Qwen/Qwen3-32B \
  --output ./converted/Qwen3-32B-TQ8 \
  --bits 4 --residual-bits 4
```

### Direct conversion (local safetensors)

```bash
# Build tq-convert
brew install cmake mlx
git clone https://github.com/ekovshilovsky/turboquant-mlx-core
cd turboquant-mlx-core && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# Convert
./build/tq-convert --model /path/to/model --bits 4 --residual-bits 4

# Validate
python scripts/validate.py --model /path/to/model-tq8
```

### Supported Ollama models

The converter maps Ollama model names to their HuggingFace fp16 source
repositories automatically. This avoids double quantization error from
converting Ollama's pre-quantized GGUF files.

| Ollama | HuggingFace Source |
|---|---|
| `qwen2.5-coder:3b` | `Qwen/Qwen2.5-Coder-3B` |
| `qwen2.5-coder:7b` | `Qwen/Qwen2.5-Coder-7B` |
| `gemma3:4b` | `google/gemma-3-4b-pt` |
| `gemma3:27b` | `google/gemma-3-27b-pt` |
| `qwen3:32b` | `Qwen/Qwen3-32B` |

## License

MIT License - Copyright (c) 2026 Eugene Kovshilovsky
