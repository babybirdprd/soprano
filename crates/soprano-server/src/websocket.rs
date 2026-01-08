//! WebSocket handler for real-time bidirectional TTS streaming

use crate::AppState;
use axum::{
    extract::ws::{Message, WebSocket, WebSocketUpgrade},
    extract::State,
    response::IntoResponse,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use futures::{SinkExt, StreamExt};
use serde::{Deserialize, Serialize};
use soprano_core::{InferConfig, StreamConfig};
use std::sync::Arc;
use tracing::{error, info};

/// WebSocket upgrade handler
/// GET /ws
pub async fn ws_handler(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    ws.on_upgrade(|socket| handle_socket(socket, state))
}

/// WebSocket message from client
#[derive(Deserialize)]
struct ClientMessage {
    /// Text to synthesize
    text: String,
    /// Whether to stream audio chunks
    #[serde(default = "default_stream")]
    stream: bool,
    /// Chunk size for streaming
    #[serde(default = "default_chunk_size")]
    chunk_size: usize,
    /// Temperature for sampling
    #[serde(default)]
    temperature: Option<f32>,
    /// Top-p for sampling
    #[serde(default)]
    top_p: Option<f32>,
}

fn default_stream() -> bool {
    true
}

fn default_chunk_size() -> usize {
    5
}

/// WebSocket message to client
#[derive(Serialize)]
#[serde(tag = "type")]
enum ServerMessage {
    #[serde(rename = "audio")]
    Audio {
        /// Base64-encoded PCM audio
        data: String,
        /// Sample rate
        sample_rate: u32,
        /// Format description
        format: String,
        /// Number of samples
        samples: usize,
    },
    #[serde(rename = "done")]
    Done { total_samples: usize },
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "ready")]
    Ready { sample_rate: u32, model: String },
}

async fn handle_socket(socket: WebSocket, state: Arc<AppState>) {
    let (mut sender, mut receiver) = socket.split();

    // Send ready message
    let ready = ServerMessage::Ready {
        sample_rate: state.sample_rate,
        model: "soprano-80m".to_string(),
    };
    if let Ok(json) = serde_json::to_string(&ready) {
        let _ = sender.send(Message::Text(json.into())).await;
    }

    info!("WebSocket client connected");

    // Process incoming messages
    while let Some(msg) = receiver.next().await {
        match msg {
            Ok(Message::Text(text)) => {
                // Parse client message
                let client_msg: ClientMessage = match serde_json::from_str(&text) {
                    Ok(msg) => msg,
                    Err(e) => {
                        let err = ServerMessage::Error {
                            message: format!("Invalid JSON: {}", e),
                        };
                        if let Ok(json) = serde_json::to_string(&err) {
                            let _ = sender.send(Message::Text(json.into())).await;
                        }
                        continue;
                    }
                };

                // Validate
                if client_msg.text.is_empty() {
                    let err = ServerMessage::Error {
                        message: "text cannot be empty".to_string(),
                    };
                    if let Ok(json) = serde_json::to_string(&err) {
                        let _ = sender.send(Message::Text(json.into())).await;
                    }
                    continue;
                }

                // Configure inference
                let mut infer_config = InferConfig::default();
                if let Some(temp) = client_msg.temperature {
                    infer_config.temperature = temp;
                }
                if let Some(top_p) = client_msg.top_p {
                    infer_config.top_p = top_p;
                }

                let stream_config = StreamConfig {
                    chunk_size: client_msg.chunk_size,
                    ..Default::default()
                };

                // Generate audio
                let sample_rate = state.sample_rate;

                if client_msg.stream {
                    // Streaming mode
                    let mut total_samples = 0;
                    let mut tts = state.tts.lock().await;

                    for chunk_result in
                        tts.infer_stream(&client_msg.text, &infer_config, &stream_config)
                    {
                        match chunk_result {
                            Ok(audio) => {
                                total_samples += audio.len();
                                let pcm_bytes = samples_to_pcm_bytes(&audio);
                                let base64_audio = BASE64.encode(&pcm_bytes);

                                let msg = ServerMessage::Audio {
                                    data: base64_audio,
                                    sample_rate,
                                    format: "pcm_s16le".to_string(),
                                    samples: audio.len(),
                                };

                                if let Ok(json) = serde_json::to_string(&msg) {
                                    if sender.send(Message::Text(json.into())).await.is_err() {
                                        error!("Failed to send audio chunk");
                                        break;
                                    }
                                }
                            }
                            Err(e) => {
                                let err = ServerMessage::Error {
                                    message: format!("Generation error: {}", e),
                                };
                                if let Ok(json) = serde_json::to_string(&err) {
                                    let _ = sender.send(Message::Text(json.into())).await;
                                }
                                break;
                            }
                        }
                    }

                    // Send done message
                    let done = ServerMessage::Done { total_samples };
                    if let Ok(json) = serde_json::to_string(&done) {
                        let _ = sender.send(Message::Text(json.into())).await;
                    }
                } else {
                    // Non-streaming mode - generate all at once
                    let mut tts = state.tts.lock().await;
                    match tts.infer(&client_msg.text, &infer_config) {
                        Ok(audio) => {
                            let pcm_bytes = samples_to_pcm_bytes(&audio);
                            let base64_audio = BASE64.encode(&pcm_bytes);

                            let msg = ServerMessage::Audio {
                                data: base64_audio,
                                sample_rate,
                                format: "pcm_s16le".to_string(),
                                samples: audio.len(),
                            };

                            if let Ok(json) = serde_json::to_string(&msg) {
                                let _ = sender.send(Message::Text(json.into())).await;
                            }

                            let done = ServerMessage::Done {
                                total_samples: audio.len(),
                            };
                            if let Ok(json) = serde_json::to_string(&done) {
                                let _ = sender.send(Message::Text(json.into())).await;
                            }
                        }
                        Err(e) => {
                            let err = ServerMessage::Error {
                                message: format!("Generation error: {}", e),
                            };
                            if let Ok(json) = serde_json::to_string(&err) {
                                let _ = sender.send(Message::Text(json.into())).await;
                            }
                        }
                    }
                }
            }
            Ok(Message::Binary(_data)) => {
                // Binary messages not supported for now
                let err = ServerMessage::Error {
                    message: "Binary messages not supported, send JSON text".to_string(),
                };
                if let Ok(json) = serde_json::to_string(&err) {
                    let _ = sender.send(Message::Text(json.into())).await;
                }
            }
            Ok(Message::Close(_)) => {
                info!("WebSocket client disconnected");
                break;
            }
            Ok(_) => {}
            Err(e) => {
                error!("WebSocket error: {}", e);
                break;
            }
        }
    }

    info!("WebSocket connection closed");
}

fn samples_to_pcm_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&sample_i16.to_le_bytes());
    }
    bytes
}
