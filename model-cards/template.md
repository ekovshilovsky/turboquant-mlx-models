---
license: mit
tags:
  - turboquant
  - mlx
  - apple-silicon
  - quantized
---

# {MODEL_NAME}-TQ{BITS}

TurboQuant-compressed version of [{MODEL_NAME}]({UPSTREAM_URL}) for Apple Silicon inference via [turboquant-mlx-core](https://github.com/ekovshilovsky/turboquant-mlx-core).

## Quantization Details

| Property | Value |
|---|---|
| Method | TurboQuant (Zandieh et al., ICLR 2026) |
| Primary bits | {PRIMARY_BITS} |
| Residual bits | {RESIDUAL_BITS} |
| Effective bits/weight | {TOTAL_BITS} |
| Model size | {SIZE_GB} GB |
| PPL vs fp16 | {PPL_DELTA} |

## Usage

```bash
# With SwiftLM
SwiftLM --model ekovshilovsky/{MODEL_NAME}-TQ{BITS} --port 5413
```

## Hardware Requirements

- Apple Silicon Mac (M1+)
- {MIN_MEMORY} GB unified memory minimum
- macOS 14+

## Benchmarks

| Metric | Value |
|---|---|
| Prefill tok/s | {PREFILL_TOKS} |
| Decode tok/s | {DECODE_TOKS} |
| TTFT (2K context) | {TTFT_MS} ms |

## License

MIT - Copyright (c) 2026 Eugene Kovshilovsky
