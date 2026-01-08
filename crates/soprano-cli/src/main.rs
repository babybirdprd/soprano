//! Soprano CLI - Command-line interface for Soprano TTS

use anyhow::Result;
use clap::Parser;
use soprano_core::{InferConfig, SopranoTTS, StreamConfig};
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(name = "soprano")]
#[command(author, version, about = "High-fidelity Text-to-Speech", long_about = None)]
struct Args {
    /// Text to synthesize
    #[arg(short, long)]
    prompt: String,

    /// Output WAV file path
    #[arg(short, long, default_value = "output.wav")]
    output: String,

    /// Force CPU inference
    #[arg(long)]
    cpu: bool,

    /// Enable streaming mode
    #[arg(long)]
    stream: bool,

    /// Chunk size for streaming (tokens per chunk)
    #[arg(long, default_value = "5")]
    chunk_size: usize,

    /// Sampling temperature (0 = greedy, higher = more random)
    #[arg(long, default_value = "0.3")]
    temperature: f32,

    /// Top-p (nucleus) sampling threshold
    #[arg(long, default_value = "0.95")]
    top_p: f32,

    /// Repetition penalty
    #[arg(long, default_value = "1.2")]
    repetition_penalty: f32,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let device = if args.cpu {
        candle_core::Device::Cpu
    } else {
        candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu)
    };
    println!("Using device: {:?}", device);

    println!("Loading models...");
    let mut tts = SopranoTTS::new(device)?;

    let infer_config = InferConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        repetition_penalty: args.repetition_penalty,
        ..Default::default()
    };

    let inference_start = Instant::now();

    let all_audio = if args.stream {
        println!("Streaming mode enabled (chunk_size={})", args.chunk_size);
        let stream_config = StreamConfig {
            chunk_size: args.chunk_size,
            ..Default::default()
        };

        let mut chunks = Vec::new();
        let mut first_chunk_time = None;

        for (i, chunk_result) in tts
            .infer_stream(&args.prompt, &infer_config, &stream_config)
            .enumerate()
        {
            let chunk = chunk_result?;
            if first_chunk_time.is_none() {
                first_chunk_time = Some(inference_start.elapsed());
                println!(
                    "First chunk latency: {:.2}ms ({} samples)",
                    first_chunk_time.unwrap().as_millis(),
                    chunk.len()
                );
            }
            println!("Chunk {}: {} samples", i + 1, chunk.len());
            chunks.push(chunk);
        }

        chunks.into_iter().flatten().collect()
    } else {
        tts.infer(&args.prompt, &infer_config)?
    };

    if all_audio.is_empty() {
        anyhow::bail!("No audio generated.");
    }

    // Audio stats
    let audio_duration_secs = all_audio.len() as f64 / tts.sample_rate() as f64;
    let min_audio = all_audio.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let max_audio = all_audio.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    println!(
        "Audio stats: len={}, min={:.4}, max={:.4}",
        all_audio.len(),
        min_audio,
        max_audio
    );

    // Save to WAV
    println!("Saving to {}...", args.output);
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: tts.sample_rate(),
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };

    let mut writer = hound::WavWriter::create(&args.output, spec)?;
    for sample in &all_audio {
        writer.write_sample(*sample)?;
    }
    writer.finalize()?;

    println!("Done!");

    // Performance metrics
    let inference_elapsed = inference_start.elapsed().as_secs_f64();
    let rtf = audio_duration_secs / inference_elapsed;
    println!("\n=== Performance ===");
    println!("Audio duration: {:.2}s", audio_duration_secs);
    println!("Generation time: {:.2}s", inference_elapsed);
    println!("RTF: {:.1}x realtime", rtf);

    Ok(())
}
