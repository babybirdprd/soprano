//! Soprano Server - HTTP API for Soprano TTS
//! 
//! Provides OpenAI-compatible and ElevenLabs-compatible endpoints for TTS.
//!
//! ## Endpoints
//! 
//! - `POST /v1/audio/speech` - OpenAI-compatible TTS endpoint
//! - `POST /v1/text-to-speech/{voice_id}` - ElevenLabs-compatible endpoint
//! - `GET /health` - Health check

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use std::net::SocketAddr;
use tokio::net::TcpListener;

mod routes;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Soprano TTS Server");
    println!("==================");
    
    // TODO: Initialize TTS model
    // let device = candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu);
    // let tts = soprano_core::SopranoTTS::new(device)?;

    let app = Router::new()
        .route("/health", get(routes::health))
        .route("/v1/audio/speech", post(routes::openai_speech))
        .route("/v1/text-to-speech/:voice_id", post(routes::elevenlabs_speech));

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("Listening on http://{}", addr);

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
