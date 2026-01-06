use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
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
    #[arg(
        short,
        long,
        default_value = "Hello world! This is a test of the Soprano text to speech model port to Rust using Candle."
    )]
    prompt: String,

    #[arg(long, default_value = "output.wav")]
    output: String,

    #[arg(long)]
    cpu: bool,
}

fn clean_text(text: &str) -> String {
    // Basic cleaning: lowercase and remove non-alphabetical/basic punctuation
    let text = text.to_lowercase();
    let mut cleaned = String::new();
    for c in text.chars() {
        if c.is_alphanumeric() || " !$%&'*+,-./0123456789<>?_".contains(c) {
            cleaned.push(c);
        }
    }
    cleaned
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
    let tokenizer_path = if std::path::Path::new("tokenizer.json").exists() {
        std::path::PathBuf::from("tokenizer.json")
    } else {
        let api = Api::new()?;
        let repo = api.repo(Repo::new("ekwek/Soprano-80M".to_string(), RepoType::Model));
        repo.get("tokenizer.json")?
    };
    let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

    // 2. Load Qwen Model
    println!("Loading Qwen model...");
    let (config_path, model_path) = if std::path::Path::new("config.json").exists()
        && std::path::Path::new("model.safetensors").exists()
    {
        (
            std::path::PathBuf::from("config.json"),
            std::path::PathBuf::from("model.safetensors"),
        )
    } else {
        let api = Api::new()?;
        let repo = api.repo(Repo::new("ekwek/Soprano-80M".to_string(), RepoType::Model));
        (repo.get("config.json")?, repo.get("model.safetensors")?)
    };

    let config: QwenConfig = serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[model_path], DType::F32, &device)? };
    let mut qwen = QwenModel::new(&config, vb)?;

    // 3. Load Vocos Model
    println!("Loading Vocos model...");
    // We generated vocos_config.json and decoder.safetensors in Phase 1 (convert.py)
    // Assuming they are in the current directory or generated path.
    // In Phase 1 we saved them to `candle/`. But this binary runs from `candle/`?
    // Cargo run usually runs from the package root.

    // Check if files exist, if not look in local `candle/` folder (if run from repo root)
    let (vocos_config_path, vocos_weights_path) =
        if std::path::Path::new("vocos_config.json").exists() {
            ("vocos_config.json", "decoder.safetensors")
        } else if std::path::Path::new("candle/vocos_config.json").exists() {
            ("candle/vocos_config.json", "candle/decoder.safetensors")
        } else {
            // Fallback: Use Phase 1 output directly if not found
            anyhow::bail!("Could not find vocos_config.json. Did you run convert.py?");
        };

    let vocos_config: VocosConfig =
        serde_json::from_str(&std::fs::read_to_string(vocos_config_path)?)?;
    let vb_vocos =
        unsafe { VarBuilder::from_mmaped_safetensors(&[vocos_weights_path], DType::F32, &device)? };
    let vocos = SopranoDecoder::new(&vocos_config, vb_vocos)?;

    // 4. Inference - Qwen
    println!("Generating tokens...");
    let cleaned_prompt = clean_text(&args.prompt);
    let prompt = format!("[STOP][TEXT]{}[START]", cleaned_prompt);
    let tokens = tokenizer.encode(prompt, false).map_err(E::msg)?; // Use false for add_special_tokens
    println!("Prompt tokens: {:?}", tokens.get_ids());
    let input_ids = Tensor::new(tokens.get_ids(), &device)?.unsqueeze(0)?;

    // Generate loop
    let mut generated_tokens = Vec::new();
    let mut generated_hidden_states = Vec::new();
    let mut current_input = input_ids;
    let mut all_tokens = tokens.get_ids().to_vec();
    let mut pos = 0;

    let temperature = 0.3f32;
    let repetition_penalty = 1.2f32;

    println!("Generating tokens (this may take a while)...");
    for iter in 0..256 {
        // Generate more tokens
        let (logits, hidden_states) = qwen.forward(&current_input, pos)?;

        // Debug: print hidden states info on first iteration
        if iter == 0 {
            let hs_shape = hidden_states.dims();
            println!("DEBUG: Hidden states shape: {:?}", hs_shape);
            let last_pos_hs = hidden_states.squeeze(0)?.get(hidden_states.dim(1)? - 1)?;
            let hs_vec = last_pos_hs.to_vec1::<f32>()?;
            println!(
                "DEBUG: Hidden states at last position (first 10): {:?}",
                &hs_vec[..10]
            );
            let min_val = hs_vec.iter().cloned().fold(f32::INFINITY, f32::min);
            let max_val = hs_vec.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            println!(
                "DEBUG: Hidden states range: [{:.4}, {:.4}]",
                min_val, max_val
            );
        }

        let mut logits = logits.squeeze(0)?.get(logits.dim(1)? - 1)?; // Last token logits

        // Repetition penalty
        if repetition_penalty != 1.0 {
            let mut logits_vec = logits.to_vec1::<f32>()?;
            for &t in all_tokens.iter() {
                let t = t as usize;
                if t < logits_vec.len() {
                    if logits_vec[t] < 0.0 {
                        logits_vec[t] *= repetition_penalty;
                    } else {
                        logits_vec[t] /= repetition_penalty;
                    }
                }
            }
            logits = Tensor::from_vec(logits_vec, logits.dims(), logits.device())?;
        }

        pos += current_input.dim(1)?;

        // Sampling
        let next_token = if temperature > 0.0 {
            let prs = candle_nn::ops::softmax_last_dim(&(logits / (temperature as f64))?)?;
            let prs_vec = prs.to_vec1::<f32>()?;

            // Simple sampling
            let mut r = rand::random::<f32>();
            let mut token = 0;
            for (i, &p) in prs_vec.iter().enumerate() {
                r -= p;
                if r <= 0.0 {
                    token = i as u32;
                    break;
                }
            }
            token
        } else {
            logits.argmax(0)?.to_scalar::<u32>()?
        };

        // Check for stop condition BEFORE collecting hidden state
        // Python filters out EOS token hidden states
        if next_token == 3 {
            // [STOP] id is 3
            break;
        }

        generated_tokens.push(next_token);
        all_tokens.push(next_token);

        // Collect hidden states only for non-EOS tokens
        let last_hidden = hidden_states.squeeze(0)?.get(hidden_states.dim(1)? - 1)?;
        generated_hidden_states.push(last_hidden);

        current_input = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
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
    let hidden_tensor = hidden_tensor.unsqueeze(0)?.transpose(1, 2)?.contiguous()?; // (1, H, L) as expected by Vocos (B, C, T)

    // 6. Decode
    println!("Decoding audio...");
    // Vocos forward expects (B, C, T)
    let audio_vec = vocos.forward(&hidden_tensor)?;

    if audio_vec.is_empty() {
        anyhow::bail!("Decoded audio is empty.");
    }

    // Python takes audio from the END: audio[-(lengths[i]*TOKEN_SIZE - TOKEN_SIZE):]
    // This means we take (num_tokens - 1) * token_size samples from the end
    let token_size = 2048;
    let num_tokens = generated_hidden_states.len();
    let expected_samples = if num_tokens > 1 {
        (num_tokens - 1) * token_size
    } else {
        audio_vec.len()
    };
    let audio_vec = if audio_vec.len() > expected_samples {
        // Take from the END of the audio, like Python does
        audio_vec[audio_vec.len() - expected_samples..].to_vec()
    } else {
        audio_vec
    };

    let min_audio = audio_vec.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_audio = audio_vec.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!(
        "Audio stats: len={}, min={:.4}, max={:.4}",
        audio_vec.len(),
        min_audio,
        max_audio
    );
    println!("Generated tokens: {:?}", generated_tokens);

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
