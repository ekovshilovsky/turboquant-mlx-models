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

```bash
# Install turboquant-mlx-core
brew install cmake mlx
git clone https://github.com/ekovshilovsky/turboquant-mlx-core
cd turboquant-mlx-core && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build

# Convert
./build/tq-convert --model /path/to/model --bits 4 --residual-bits 4

# Validate
python scripts/validate.py --model /path/to/model-tq8
```

## License

MIT License - Copyright (c) 2026 Eugene Kovshilovsky
