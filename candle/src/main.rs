use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder};
// use candle_transformers::models::qwen2::{Config as QwenConfig, ModelForCausalLM as QwenModel};
use clap::Parser;
use hf_hub::{api::sync::Api, Repo, RepoType};
use hound;
use tokenizers::Tokenizer;

mod vocos;
use vocos::{SopranoDecoder, VocosConfig};
mod qwen;
use qwen::{Config as QwenConfig, ModelForCausalLM as QwenModel};

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    #[arg(short, long, default_value = "Hello world! This is a test of the Soprano text to speech model port to Rust using Candle.")]
    prompt: String,

    #[arg(long, default_value = "output.wav")]
    output: String,

    #[arg(long)]
    cpu: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = if args.cpu {
        Device::Cpu
    } else {
        Device::new_cuda(0).unwrap_or(Device::Cpu)
    };
    println!("Using device: {:?}", device);

    // 1. Load Tokenizer
    println!("Loading tokenizer...");
    let api = Api::new()?;
    let repo = api.repo(Repo::new("ekwek/Soprano-80M".to_string(), RepoType::Model));
    let tokenizer_filename = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;

    // 2. Load Qwen Model
    println!("Loading Qwen model...");
    let config_filename = repo.get("config.json")?;
    let config: QwenConfig = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;

    let model_filename = repo.get("model.safetensors")?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_filename], DType::F32, &device)? };
    let mut qwen = QwenModel::new(&config, vb)?;

    // 3. Load Vocos Model
    println!("Loading Vocos model...");
    // We generated vocos_config.json and decoder.safetensors in Phase 1 (convert.py)
    // Assuming they are in the current directory or generated path.
    // In Phase 1 we saved them to `candle/`. But this binary runs from `candle/`?
    // Cargo run usually runs from the package root.

    // Check if files exist, if not look in local `candle/` folder (if run from repo root)
    let (vocos_config_path, vocos_weights_path) = if std::path::Path::new("vocos_config.json").exists() {
        ("vocos_config.json", "decoder.safetensors")
    } else if std::path::Path::new("candle/vocos_config.json").exists() {
        ("candle/vocos_config.json", "candle/decoder.safetensors")
    } else {
        // Fallback: Use Phase 1 output directly if not found
        anyhow::bail!("Could not find vocos_config.json. Did you run convert.py?");
    };

    let vocos_config: VocosConfig = serde_json::from_str(&std::fs::read_to_string(vocos_config_path)?)?;
    let vb_vocos = unsafe { VarBuilder::from_mmaped_safetensors(&[vocos_weights_path], DType::F32, &device)? };
    let vocos = SopranoDecoder::new(&vocos_config, vb_vocos)?;

    // 4. Inference - Qwen
    println!("Generating tokens...");
    let prompt = format!("[STOP][TEXT]{}[START]", args.prompt);
    let tokens = tokenizer.encode(prompt, true).map_err(E::msg)?;
    let input_ids = Tensor::new(tokens.get_ids(), &device)?.unsqueeze(0)?;

    // Generate loop
    let mut generated_tokens = Vec::new();
    let mut generated_hidden_states = Vec::new();
    let mut current_input = input_ids;
    let mut pos = 0;

    println!("Generating tokens (this may take a while)...");
    for _ in 0..200 { // Limit to 200 tokens for test
        let (logits, hidden_states) = qwen.forward(&current_input, pos)?;
        let logits = logits.squeeze(0)?.get(logits.dim(1)? - 1)?; // Last token logits

        // Greedy decoding
        let next_token = logits.argmax(0)?.to_scalar::<u32>()?;
        generated_tokens.push(next_token);

        // Save hidden state for this token
        // hidden_states is (B, L, H) -> (1, L, H)
        // We want the last hidden state corresponding to the newly generated token?
        // Wait, qwen forward returns hidden states for the input tokens.
        // If we pass input tokens, we get their hidden states.
        // We want the hidden state of the token we just predicted? No, usually the hidden state that *produced* the prediction.
        // In autoregressive models, h_t is used to predict x_{t+1}.
        // So for token x_{t+1}, the code (audio info) is in h_t.
        // tts.py says: "hidden_state = response['hidden_state']". This likely refers to the hidden state of the last token processed.

        let last_hidden = hidden_states.squeeze(0)?.get(hidden_states.dim(1)? - 1)?; // (H)
        generated_hidden_states.push(last_hidden);

        // Check for stop condition
        if let Some(eos) = tokenizer.get_vocab(true).get("<|endoftext|>") {
            if next_token == *eos {
                break;
            }
        }

        current_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
        pos += current_input.dim(1)?;
    }

    // 5. Extract Audio Tokens / Hidden States
    println!("Extracting audio codes...");
    // We filter tokens that are in the audio range, and keep their corresponding hidden states.
    // Or does the model only generate audio tokens after [START]?
    // Soprano prompt format: [STOP][TEXT]...[START] -> Audio tokens follow.
    // So all generated tokens should be considered audio tokens until stop?
    // tts.py seems to take all hidden states from the response.
    // But it does verify if they are valid?

    // Let's assume all generated tokens are part of the audio sequence.

    if generated_hidden_states.is_empty() {
        anyhow::bail!("No tokens generated.");
    }

    let hidden_dim = generated_hidden_states[0].dim(0)?;
    let hidden_tensor = Tensor::stack(&generated_hidden_states, 0)?; // (L, H)
    let hidden_tensor = hidden_tensor.unsqueeze(0)?.transpose(1, 2)?; // (1, H, L) as expected by Vocos (B, C, T)

    // 6. Decode
    println!("Decoding audio...");
    // Vocos forward expects (B, C, T)
    let audio_vec = vocos.forward(&hidden_tensor)?;

    // 7. Save to WAV
    println!("Saving to {}...", args.output);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 32000, // Soprano uses 32kHz? tts.py: wavfile.write(out_path, 32000, ...)
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(&args.output, spec)?;
    for sample in audio_vec {
        writer.write_sample(sample)?;
    }
    writer.finalize()?;

    println!("Done!");

    Ok(())
}
