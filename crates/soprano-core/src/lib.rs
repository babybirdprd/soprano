//! Soprano Core - High-fidelity Text-to-Speech Library
//!
//! This crate provides the core TTS functionality including:
//! - `SopranoTTS` struct for inference
//! - Streaming audio generation
//! - Text normalization

mod qwen;
mod text;
mod vocos;

pub use text::clean_text;

use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use fancy_regex::Regex;
use hf_hub::{api::sync::Api, Repo, RepoType};
use qwen::{Config as QwenConfig, ModelForCausalLM as QwenModel};
use std::path::PathBuf;
use tokenizers::Tokenizer;
use vocos::{SopranoDecoder, VocosConfig};

/// Configuration for inference
#[derive(Debug, Clone)]
pub struct InferConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub max_tokens_per_sentence: usize,
}

impl Default for InferConfig {
    fn default() -> Self {
        Self {
            temperature: 0.3,
            top_p: 0.95,
            repetition_penalty: 1.2,
            max_tokens_per_sentence: 512,
        }
    }
}

/// Streaming configuration
#[derive(Debug, Clone)]
pub struct StreamConfig {
    /// Number of tokens to generate before yielding audio chunk
    pub chunk_size: usize,
    /// Decoder receptive field (tokens of context needed)
    pub receptive_field: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            chunk_size: 5,
            receptive_field: 4,
        }
    }
}

/// Main TTS engine
pub struct SopranoTTS {
    qwen: QwenModel,
    vocos: SopranoDecoder,
    tokenizer: Tokenizer,
    device: Device,
    token_size: usize,
}

impl SopranoTTS {
    /// Create a new SopranoTTS instance
    pub fn new(device: Device) -> Result<Self> {
        // Load tokenizer
        let tokenizer_path = Self::get_model_file("tokenizer.json")?;
        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(E::msg)?;

        // Load Qwen model
        let config_path = Self::get_model_file("config.json")?;
        let model_path = Self::get_model_file("model.safetensors")?;
        let config: QwenConfig = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
        let vb =
            unsafe { VarBuilder::from_mmaped_safetensors(&[&model_path], DType::F32, &device)? };
        let qwen = QwenModel::new(&config, vb)?;

        // Load Vocos decoder
        let vocos_config_path = Self::get_model_file("vocos_config.json")?;
        let vocos_weights_path = Self::get_model_file("decoder.safetensors")?;
        let vocos_config: VocosConfig =
            serde_json::from_str(&std::fs::read_to_string(&vocos_config_path)?)?;
        let vb_vocos = unsafe {
            VarBuilder::from_mmaped_safetensors(&[&vocos_weights_path], DType::F32, &device)?
        };
        let vocos = SopranoDecoder::new(&vocos_config, vb_vocos)?;

        Ok(Self {
            qwen,
            vocos,
            tokenizer,
            device,
            token_size: 2048, // Samples per audio token
        })
    }

    fn get_model_file(filename: &str) -> Result<PathBuf> {
        // Check local first
        let local_path = PathBuf::from(filename);
        if local_path.exists() {
            return Ok(local_path);
        }

        // Download from HuggingFace
        let api = Api::new()?;
        let repo = api.repo(Repo::new("ekwek/Soprano-80M".to_string(), RepoType::Model));
        Ok(repo.get(filename)?)
    }

    /// Preprocess text into sentences
    pub fn preprocess_text(&self, text: &str, min_length: usize) -> Vec<String> {
        let cleaned = clean_text(text);
        let re_sentence = Regex::new(r"(?<=[.!?])\s+").unwrap();
        let sentences: Vec<&str> = re_sentence.split(&cleaned).map(|s| s.unwrap()).collect();

        let mut processed = Vec::new();
        let mut current_sentence = String::new();

        for s in sentences {
            if current_sentence.is_empty() {
                current_sentence = s.to_string();
            } else {
                current_sentence.push(' ');
                current_sentence.push_str(s);
            }

            if current_sentence.len() >= min_length {
                processed.push(current_sentence);
                current_sentence = String::new();
            }
        }

        if !current_sentence.is_empty() {
            if let Some(last) = processed.last_mut() {
                last.push(' ');
                last.push_str(&current_sentence);
            } else {
                processed.push(current_sentence);
            }
        }

        processed
    }

    /// Generate audio from text (non-streaming)
    pub fn infer(&mut self, text: &str, config: &InferConfig) -> Result<Vec<f32>> {
        let sentences = self.preprocess_text(text, 30);
        let mut all_audio = Vec::new();

        for sentence in sentences {
            let audio = self.generate_sentence(&sentence, config)?;
            all_audio.extend(audio);
        }

        Ok(all_audio)
    }

    /// Generate audio for a single sentence
    fn generate_sentence(&mut self, text: &str, config: &InferConfig) -> Result<Vec<f32>> {
        let prompt = format!("[STOP][TEXT]{}[START]", text);
        let tokens = self.tokenizer.encode(prompt, false).map_err(E::msg)?;
        let mut current_input = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;
        let mut all_tokens = tokens.get_ids().to_vec();
        let mut pos = 0;

        let mut generated_hidden_states = Vec::new();

        self.qwen.clear_kv_cache();

        for _iter in 0..config.max_tokens_per_sentence {
            let (logits, hidden_states) = self.qwen.forward(&current_input, pos)?;
            let mut logits = logits.squeeze(0)?.get(logits.dim(1)? - 1)?;

            // Apply repetition penalty
            if config.repetition_penalty != 1.0 {
                let mut logits_vec = logits.to_vec1::<f32>()?;
                for &t in all_tokens.iter() {
                    let t = t as usize;
                    if t < logits_vec.len() {
                        if logits_vec[t] < 0.0 {
                            logits_vec[t] *= config.repetition_penalty;
                        } else {
                            logits_vec[t] /= config.repetition_penalty;
                        }
                    }
                }
                logits = Tensor::from_vec(logits_vec, logits.dims(), logits.device())?;
            }

            pos += current_input.dim(1)?;

            // Sample next token
            let next_token = if config.temperature > 0.0 {
                self.sample_top_p(&logits, config.temperature, config.top_p)?
            } else {
                logits.argmax(0)?.to_scalar::<u32>()?
            };

            // EOS token
            if next_token == 3 {
                break;
            }

            all_tokens.push(next_token);
            let last_hidden = hidden_states.squeeze(0)?.get(hidden_states.dim(1)? - 1)?;
            generated_hidden_states.push(last_hidden);
            current_input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
        }

        if generated_hidden_states.is_empty() {
            return Ok(Vec::new());
        }

        // Decode to audio
        let hidden_tensor = Tensor::stack(&generated_hidden_states, 0)?;
        let hidden_tensor = hidden_tensor.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;

        let audio_vec = self.vocos.forward(&hidden_tensor)?;

        // Trim first token's worth of audio
        let num_tokens = generated_hidden_states.len();
        let expected_samples = if num_tokens > 1 {
            (num_tokens - 1) * self.token_size
        } else {
            audio_vec.len()
        };

        let audio_vec = if audio_vec.len() > expected_samples {
            audio_vec[audio_vec.len() - expected_samples..].to_vec()
        } else {
            audio_vec
        };

        Ok(audio_vec)
    }

    /// Sample from logits using top-p (nucleus) sampling
    fn sample_top_p(&self, logits: &Tensor, temperature: f32, top_p: f32) -> Result<u32> {
        let prs = candle_nn::ops::softmax_last_dim(&(logits / (temperature as f64))?)?;
        let mut prs_vec = prs.to_vec1::<f32>()?;

        let mut sorted_prs: Vec<(usize, f32)> =
            prs_vec.iter().enumerate().map(|(i, &p)| (i, p)).collect();
        sorted_prs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let mut cumulative_p = 0.0;
        let mut cutoff_idx = sorted_prs.len();
        for (i, (_, p)) in sorted_prs.iter().enumerate() {
            cumulative_p += p;
            if cumulative_p > top_p {
                cutoff_idx = i + 1;
                break;
            }
        }

        for i in cutoff_idx..sorted_prs.len() {
            prs_vec[sorted_prs[i].0] = 0.0;
        }

        let sum: f32 = prs_vec.iter().sum();
        for p in prs_vec.iter_mut() {
            *p /= sum;
        }

        let mut r = rand::random::<f32>();
        let mut token = 0;
        for (i, &p) in prs_vec.iter().enumerate() {
            r -= p;
            if r <= 0.0 {
                token = i as u32;
                break;
            }
        }

        Ok(token)
    }

    /// Generate audio with streaming - yields chunks as they're generated
    pub fn infer_stream<'a>(
        &'a mut self,
        text: &'a str,
        infer_config: &'a InferConfig,
        stream_config: &'a StreamConfig,
    ) -> impl Iterator<Item = Result<Vec<f32>>> + 'a {
        let sentences = self.preprocess_text(text, 30);
        StreamingIterator::new(self, sentences, infer_config.clone(), stream_config.clone())
    }

    /// Get the sample rate (32kHz)
    pub fn sample_rate(&self) -> u32 {
        32000
    }
}

/// Iterator for streaming audio generation
struct StreamingIterator<'a> {
    tts: &'a mut SopranoTTS,
    sentences: Vec<String>,
    current_sentence_idx: usize,
    infer_config: InferConfig,
    stream_config: StreamConfig,
    // State for current sentence
    all_tokens: Vec<u32>,
    hidden_states_buffer: Vec<Tensor>,
    pos: usize,
    chunk_counter: usize,
    sentence_finished: bool,
    current_input: Option<Tensor>,
    is_prompt_phase: bool,       // True during initial prompt processing
    generated_count: usize,      // Count of generated tokens
    last_yielded_samples: usize, // Track how many valid samples we've yielded
    prev_audio_len: usize,       // Previous audio length for delta calculation
}

impl<'a> StreamingIterator<'a> {
    fn new(
        tts: &'a mut SopranoTTS,
        sentences: Vec<String>,
        infer_config: InferConfig,
        stream_config: StreamConfig,
    ) -> Self {
        Self {
            tts,
            sentences,
            current_sentence_idx: 0,
            infer_config,
            stream_config,
            all_tokens: Vec::new(),
            hidden_states_buffer: Vec::new(),
            pos: 0,
            chunk_counter: 0,
            sentence_finished: true,
            current_input: None,
            is_prompt_phase: true,
            generated_count: 0,
            last_yielded_samples: 0,
            prev_audio_len: 0,
        }
    }

    fn start_new_sentence(&mut self) -> Result<bool> {
        if self.current_sentence_idx >= self.sentences.len() {
            return Ok(false);
        }

        let sentence = &self.sentences[self.current_sentence_idx];
        let prompt = format!("[STOP][TEXT]{}[START]", sentence);
        let tokens = self.tts.tokenizer.encode(prompt, false).map_err(E::msg)?;

        self.tts.qwen.clear_kv_cache();
        self.all_tokens = tokens.get_ids().to_vec();
        self.hidden_states_buffer.clear();
        self.pos = 0;
        self.chunk_counter = self.stream_config.chunk_size; // Start at chunk_size for immediate yield
        self.sentence_finished = false;
        self.is_prompt_phase = true;
        self.generated_count = 0;
        self.last_yielded_samples = 0;
        self.prev_audio_len = 0;
        self.current_input = Some(Tensor::new(tokens.get_ids(), &self.tts.device)?.unsqueeze(0)?);

        Ok(true)
    }

    fn generate_next_chunk(&mut self) -> Result<Option<Vec<f32>>> {
        let chunk_size = self.stream_config.chunk_size;
        let receptive_field = self.stream_config.receptive_field;

        loop {
            if self.sentence_finished {
                if !self.start_new_sentence()? {
                    return Ok(None); // All sentences done
                }
            }

            let current_input = self.current_input.take().unwrap();
            let (logits, hidden_states) = self.tts.qwen.forward(&current_input, self.pos)?;
            let mut logits_last = logits.squeeze(0)?.get(logits.dim(1)? - 1)?;

            // Apply repetition penalty
            if self.infer_config.repetition_penalty != 1.0 {
                let mut logits_vec = logits_last.to_vec1::<f32>()?;
                for &t in self.all_tokens.iter() {
                    let t = t as usize;
                    if t < logits_vec.len() {
                        if logits_vec[t] < 0.0 {
                            logits_vec[t] *= self.infer_config.repetition_penalty;
                        } else {
                            logits_vec[t] /= self.infer_config.repetition_penalty;
                        }
                    }
                }
                logits_last =
                    Tensor::from_vec(logits_vec, logits_last.dims(), logits_last.device())?;
            }

            self.pos += current_input.dim(1)?;

            let next_token = if self.infer_config.temperature > 0.0 {
                self.tts.sample_top_p(
                    &logits_last,
                    self.infer_config.temperature,
                    self.infer_config.top_p,
                )?
            } else {
                logits_last.argmax(0)?.to_scalar::<u32>()?
            };

            let finished = next_token == 3; // EOS

            // Only add hidden states for generated tokens, not prompt tokens
            if !self.is_prompt_phase && !finished {
                let last_hidden = hidden_states.squeeze(0)?.get(hidden_states.dim(1)? - 1)?;
                self.hidden_states_buffer.push(last_hidden);
            }

            // After first forward pass, we're generating tokens
            if self.is_prompt_phase {
                self.is_prompt_phase = false;
            }

            if !finished {
                self.all_tokens.push(next_token);
                self.generated_count += 1;
                self.current_input =
                    Some(Tensor::new(&[next_token], &self.tts.device)?.unsqueeze(0)?);
            }

            // Don't trim buffer - we need all hidden states for proper decoding
            // (Python handles this differently with their receptive field approach)

            // Python: check conditions then increment counter
            // if finished or len(buffer) >= RF + chunk_size:
            //     if finished or chunk_counter == chunk_size:
            let buffer_ready = self.hidden_states_buffer.len() >= receptive_field + chunk_size;
            let should_check = finished || buffer_ready;
            let should_yield = should_check && (finished || self.chunk_counter == chunk_size);

            if should_yield && !self.hidden_states_buffer.is_empty() {
                // Decode ALL accumulated hidden states (like non-streaming)
                let hidden_tensor = Tensor::stack(&self.hidden_states_buffer, 0)?;
                let hidden_tensor = hidden_tensor.unsqueeze(0)?.transpose(1, 2)?.contiguous()?;
                let audio = self.tts.vocos.forward(&hidden_tensor)?;

                let token_size = self.tts.token_size;
                let num_tokens = self.hidden_states_buffer.len();

                // Same logic as non-streaming: take last (n-1)*token_size samples
                let expected_samples = if num_tokens > 1 {
                    (num_tokens - 1) * token_size
                } else {
                    audio.len()
                };

                // Extract valid audio from the end
                let valid_audio = if audio.len() > expected_samples {
                    &audio[audio.len() - expected_samples..]
                } else {
                    &audio[..]
                };

                // Return only NEW samples we haven't yielded yet
                let audio_chunk = if valid_audio.len() > self.last_yielded_samples {
                    let chunk = valid_audio[self.last_yielded_samples..].to_vec();
                    self.last_yielded_samples = valid_audio.len();
                    chunk
                } else {
                    Vec::new()
                };

                self.chunk_counter = 0;

                if finished {
                    self.sentence_finished = true;
                    self.current_sentence_idx += 1;
                }

                self.chunk_counter += 1;

                if !audio_chunk.is_empty() {
                    return Ok(Some(audio_chunk));
                }
            } else {
                self.chunk_counter += 1;
            }

            if finished {
                self.sentence_finished = true;
                self.current_sentence_idx += 1;
            }

            // Safety limit
            if self.generated_count > self.infer_config.max_tokens_per_sentence {
                self.sentence_finished = true;
                self.current_sentence_idx += 1;
            }
        }
    }
}

impl<'a> Iterator for StreamingIterator<'a> {
    type Item = Result<Vec<f32>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.generate_next_chunk() {
            Ok(Some(chunk)) => Some(Ok(chunk)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_config_default() {
        let config = InferConfig::default();
        assert_eq!(config.temperature, 0.3);
        assert_eq!(config.top_p, 0.95);
    }
}
