//! Soprano Server - HTTP API for Soprano TTS
//!
//! Provides OpenAI-compatible and ElevenLabs-compatible endpoints for TTS.
//!
//! ## Endpoints
//!
//! - `POST /v1/audio/speech` - OpenAI-compatible TTS endpoint
//! - `POST /v1/text-to-speech/{voice_id}` - ElevenLabs-compatible endpoint
//! - `POST /v1/audio/speech/stream` - SSE streaming endpoint
//! - `GET /ws` - WebSocket endpoint for real-time TTS
//! - `GET /health` - Health check

use anyhow::Result;
use axum::{
    routing::{get, post},
    Router,
};
use clap::Parser;
use soprano_core::SopranoTTS;
use std::net::SocketAddr;
use std::sync::Arc;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

mod routes;
mod sse;
mod websocket;

/// Shared application state
pub struct AppState {
    pub tts: Mutex<SopranoTTS>,
    pub sample_rate: u32,
}

/// Soprano TTS HTTP Server
#[derive(Parser, Debug)]
#[command(name = "soprano-server")]
#[command(about = "HTTP API server for Soprano TTS")]
struct Args {
    /// Host address to bind to
    #[arg(long, default_value = "0.0.0.0")]
    host: String,

    /// Port to listen on
    #[arg(long, default_value = "3000")]
    port: u16,

    /// Force CPU inference (disable CUDA)
    #[arg(long)]
    cpu: bool,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("soprano_server=info".parse().unwrap()),
        )
        .init();

    let args = Args::parse();

    println!("╔══════════════════════════════════════╗");
    println!(
        "║       Soprano TTS Server v{}       ║",
        env!("CARGO_PKG_VERSION")
    );
    println!("╚══════════════════════════════════════╝");
    println!();

    // Initialize device
    let device = if args.cpu {
        info!("Using CPU device");
        candle_core::Device::Cpu
    } else {
        match candle_core::Device::new_cuda(0) {
            Ok(d) => {
                info!("Using CUDA device");
                d
            }
            Err(_) => {
                info!("CUDA not available, falling back to CPU");
                candle_core::Device::Cpu
            }
        }
    };

    // Initialize TTS model
    info!("Loading TTS model...");
    let tts = SopranoTTS::new(device)?;
    let sample_rate = tts.sample_rate();

    let state = Arc::new(AppState {
        tts: Mutex::new(tts),
        sample_rate,
    });
    info!("Model loaded successfully");

    // CORS layer for web clients
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        // Health check
        .route("/health", get(routes::health))
        // OpenAI-compatible endpoints
        .route("/v1/audio/speech", post(routes::openai_speech))
        .route("/v1/audio/speech/stream", post(sse::stream_speech))
        // ElevenLabs-compatible endpoint
        .route(
            "/v1/text-to-speech/{voice_id}",
            post(routes::elevenlabs_speech),
        )
        .route(
            "/v1/text-to-speech/{voice_id}/stream",
            post(sse::stream_speech_elevenlabs),
        )
        // WebSocket endpoint
        .route("/ws", get(websocket::ws_handler))
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", args.host, args.port).parse()?;
    info!("Server listening on http://{}", addr);
    println!();
    println!("Endpoints:");
    println!("  POST /v1/audio/speech          - OpenAI-compatible TTS");
    println!("  POST /v1/audio/speech/stream   - OpenAI SSE streaming");
    println!("  POST /v1/text-to-speech/:id    - ElevenLabs-compatible TTS");
    println!("  GET  /ws                       - WebSocket streaming");
    println!("  GET  /health                   - Health check");
    println!();

    let listener = TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
