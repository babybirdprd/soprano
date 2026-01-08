# Soprano (Candle Port)

Soprano is a high-fidelity, high-performance Text-to-Speech (TTS) model. This repository contains the **Rust implementation** using the [Candle](https://github.com/huggingface/candle) ML framework.

## Key Features

- **32 kHz High-Fidelity Audio**: Synthesizes speech with exceptional clarity.
- **Vocos-based Neural Decoder**: Extremely fast waveform generation.
- **Infinite Generation Length**: Supports processing long texts by splitting them into independent sentences.
- **Lightweight**: Optimized for performance on both CPU and GPU.

## Getting Started

### Prerequisites

- Rust (latest stable)
- CUDA (optional, for GPU acceleration)

### Usage

To generate speech from text:

```bash
cargo run --release -- --prompt "Hello world! This is Soprano running in Rust." --output output.wav
```

### Options

- `--prompt`: The text to synthesize.
- `--output`: Path to the output `.wav` file (default: `output.wav`).
- `--cpu`: Force CPU inference even if CUDA is available.

## Model Details

This port uses the **Soprano-80M** model. The weights and configurations are automatically downloaded from Hugging Face if not present locally.

- **Qwen-based Token Generator**: Based on the Qwen architecture.
- **Vocos Decoder**: High-quality, low-latency neural vocoder.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

---
*Note: The original Python implementation is available in the `soprano/` folder for reference.*
