# Soprano (Candle Port)

A high-fidelity, high-performance Text-to-Speech (TTS) engine implemented in **Rust** using the [Candle](https://github.com/huggingface/candle) ML framework.

This is a port of the original [Soprano TTS](https://huggingface.co/ekwek/Soprano-80M) Python implementation, targeting production deployment with native performance.

## Key Features

- **32 kHz High-Fidelity Audio** - Synthesizes speech with exceptional clarity
- **Vocos-based Neural Decoder** - Extremely fast waveform generation
- **Infinite Generation Length** - Processes text sentence-by-sentence for unlimited length
- **Lightweight** - Only 80M parameters, runs on CPU or GPU
- **Native Performance** - No Python runtime, optimized Rust implementation

## Getting Started

### Prerequisites

- Rust (latest stable)
- CUDA (optional, for GPU acceleration)

### Installation

```bash
git clone https://github.com/your-repo/soprano-candle.git
cd soprano-candle
cargo build --release
```

### Usage

```bash
# Basic usage
cargo run --release -- --prompt "Hello world! This is Soprano running in Rust."

# Specify output file
cargo run --release -- --prompt "Your text here." --output speech.wav

# Force CPU inference
cargo run --release -- --prompt "Your text here." --cpu
```

### CLI Options

| Option | Description | Default |
|--------|-------------|---------|
| `--prompt`, `-p` | Text to synthesize | (required) |
| `--output` | Output WAV file path | `output.wav` |
| `--cpu` | Force CPU inference | `false` |

## Model Details

This port uses the **Soprano-80M** model. Weights are automatically downloaded from Hugging Face on first run.

- **Token Generator**: Qwen-based architecture (80M parameters)
- **Audio Decoder**: Vocos neural vocoder
- **Sample Rate**: 32 kHz
- **Audio Tokens**: ~15 tokens/second

---

## Roadmap

### ✅ Phase 1: Core Functionality (Complete)
- [x] Qwen token generation with KV caching
- [x] Vocos audio decoder
- [x] Sentence-level processing for infinite context
- [x] Basic CLI interface
- [x] CPU and CUDA support

### 🔄 Phase 2: Parity & Quality (In Progress)
- [ ] Full text normalization pipeline
  - [ ] Number expansion ($2.47 → "two dollars forty seven cents")
  - [ ] Abbreviation expansion (Mr. → "mister", TTS → "text to speech")
  - [ ] Date/time handling (12:00 → "twelve o'clock")
  - [ ] Phone number formatting
  - [ ] Special character replacement
- [ ] RTF (real-time factor) metrics and latency reporting
- [ ] Configurable sampling parameters via CLI (temperature, top_p, etc.)

### 📦 Phase 3: Modular Architecture
- [ ] Refactor into Cargo workspace
  - [ ] `soprano-core` - Library crate with `SopranoTTS` struct
  - [ ] `soprano-cli` - CLI binary
  - [ ] `soprano-server` - HTTP API server
- [ ] Public Rust API with builder pattern
- [ ] Configuration structs (`InferConfig`)

### 🌊 Phase 4: Streaming
- [ ] Token-by-token streaming inference
- [ ] Chunked audio decoding with finite receptive field
- [ ] `--stream` CLI flag with real-time output
- [ ] <15ms first-chunk latency (matching Python)

### 🚀 Phase 5: Server & API
- [ ] HTTP server with Axum
- [ ] OpenAI TTS API compatibility (`/v1/audio/speech`)
- [ ] ElevenLabs API compatibility
- [ ] Server-Sent Events (SSE) for streaming
- [ ] WebSocket support

### ⚡ Phase 6: Advanced Features
- [ ] Batched inference for throughput
- [ ] INT8/INT4 quantization
- [ ] Voice cloning (speaker embeddings)
- [ ] Multi-speaker support
- [ ] ONNX export option

---

## Performance

| Metric | Python (LMDeploy) | Rust (Candle) | Notes |
|--------|-------------------|---------------|-------|
| RTF | ~2000x | TBD | Real-time factor |
| First chunk latency | <15ms | TBD | Streaming mode |
| Memory usage | <1GB VRAM | TBD | GPU mode |

*Performance benchmarks in progress*

---

## Architecture

```
soprano/
├── src/
│   ├── main.rs      # CLI entry point
│   ├── qwen.rs      # Qwen token generator
│   └── vocos.rs     # Vocos audio decoder
├── soprano/         # Python reference implementation
└── README.md
```

**Future workspace structure:**
```
soprano/
├── crates/
│   ├── soprano-core/    # Library: SopranoTTS
│   ├── soprano-cli/     # Binary: CLI
│   └── soprano-server/  # Binary: HTTP API
├── soprano/             # Python reference
└── Cargo.toml           # Workspace root
```

---

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Soprano](https://huggingface.co/ekwek/Soprano-80M) - Original model by ekwek
- [Candle](https://github.com/huggingface/candle) - Rust ML framework by Hugging Face
- [Vocos](https://github.com/gemelo-ai/vocos) - Neural vocoder architecture

---

*Note: The original Python implementation is available in the `soprano/` folder for reference.*
