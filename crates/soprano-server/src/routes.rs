//! HTTP route handlers for Soprano Server

#![allow(dead_code)] // API compatibility fields may not all be used

use crate::AppState;
use axum::{
    body::Body,
    extract::{Path, State},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    Json,
};
use serde::{Deserialize, Serialize};
use soprano_core::InferConfig;
use std::io::Cursor;
use std::sync::Arc;

/// Health check endpoint
pub async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        model: "soprano-80m".to_string(),
    })
}

#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    version: String,
    model: String,
}

/// OpenAI-compatible speech request
/// https://platform.openai.com/docs/api-reference/audio/createSpeech
#[derive(Deserialize)]
pub struct OpenAISpeechRequest {
    /// Model to use (accepted: "soprano-80m", "tts-1", "tts-1-hd")
    pub model: String,
    /// Text to synthesize (max 4096 characters)
    pub input: String,
    /// Voice to use (ignored - single voice model)
    #[serde(default)]
    pub voice: Option<String>,
    /// Output format: wav (default), mp3, opus, aac, flac, pcm
    #[serde(default)]
    pub response_format: Option<String>,
    /// Speed multiplier (0.25 to 4.0, default 1.0) - currently ignored
    #[serde(default)]
    pub speed: Option<f32>,
}

/// OpenAI-compatible TTS endpoint
/// POST /v1/audio/speech
pub async fn openai_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<OpenAISpeechRequest>,
) -> Response {
    // Validate input
    if request.input.is_empty() {
        return error_response(
            StatusCode::BAD_REQUEST,
            "input cannot be empty",
            "invalid_request_error",
        );
    }

    if request.input.len() > 4096 {
        return error_response(
            StatusCode::BAD_REQUEST,
            "input cannot exceed 4096 characters",
            "invalid_request_error",
        );
    }

    // Determine output format
    let format = request.response_format.as_deref().unwrap_or("wav");
    if format != "wav" && format != "pcm" {
        return error_response(
            StatusCode::BAD_REQUEST,
            "Only 'wav' and 'pcm' formats are currently supported",
            "invalid_request_error",
        );
    }

    // Generate audio
    let config = InferConfig::default();

    let audio = {
        let mut tts = state.tts.lock().await;
        match tts.infer(&request.input, &config) {
            Ok(audio) => audio,
            Err(e) => {
                return error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("TTS generation failed: {}", e),
                    "server_error",
                );
            }
        }
    };

    if audio.is_empty() {
        return error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            "TTS generated empty audio",
            "server_error",
        );
    }

    // Return audio based on format
    match format {
        "pcm" => {
            // Return raw PCM samples (24kHz 16-bit signed LE per OpenAI spec)
            // We output 32kHz so mention in headers
            let pcm_bytes = samples_to_pcm_bytes(&audio);
            Response::builder()
                .status(StatusCode::OK)
                .header(header::CONTENT_TYPE, "audio/pcm")
                .header("X-Sample-Rate", state.sample_rate.to_string())
                .body(Body::from(pcm_bytes))
                .unwrap()
        }
        _ => {
            // Default: WAV format
            match encode_wav(&audio, state.sample_rate) {
                Ok(wav_bytes) => Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "audio/wav")
                    .body(Body::from(wav_bytes))
                    .unwrap(),
                Err(e) => error_response(
                    StatusCode::INTERNAL_SERVER_ERROR,
                    &format!("WAV encoding failed: {}", e),
                    "server_error",
                ),
            }
        }
    }
}

/// ElevenLabs-compatible speech request
#[derive(Deserialize)]
pub struct ElevenLabsSpeechRequest {
    pub text: String,
    #[serde(default)]
    pub model_id: Option<String>,
    #[serde(default)]
    pub voice_settings: Option<VoiceSettings>,
}

#[derive(Deserialize)]
pub struct VoiceSettings {
    #[serde(default)]
    pub stability: Option<f32>,
    #[serde(default)]
    pub similarity_boost: Option<f32>,
}

/// ElevenLabs-compatible TTS endpoint
/// POST /v1/text-to-speech/{voice_id}
pub async fn elevenlabs_speech(
    Path(_voice_id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(request): Json<ElevenLabsSpeechRequest>,
) -> Response {
    // Validate input
    if request.text.is_empty() {
        return elevenlabs_error("text cannot be empty", "validation_error");
    }

    // Map voice_settings to InferConfig if provided
    let config = InferConfig::default();

    // Generate audio
    let audio = {
        let mut tts = state.tts.lock().await;
        match tts.infer(&request.text, &config) {
            Ok(audio) => audio,
            Err(e) => {
                return elevenlabs_error(&format!("TTS generation failed: {}", e), "server_error");
            }
        }
    };

    if audio.is_empty() {
        return elevenlabs_error("TTS generated empty audio", "server_error");
    }

    // ElevenLabs defaults to mp3, but we only support wav/pcm for now
    match encode_wav(&audio, state.sample_rate) {
        Ok(wav_bytes) => Response::builder()
            .status(StatusCode::OK)
            .header(header::CONTENT_TYPE, "audio/wav")
            .header("X-Request-Id", uuid_simple())
            .body(Body::from(wav_bytes))
            .unwrap(),
        Err(e) => elevenlabs_error(&format!("WAV encoding failed: {}", e), "server_error"),
    }
}

// Helper functions

fn error_response(status: StatusCode, message: &str, error_type: &str) -> Response {
    let body = serde_json::json!({
        "error": {
            "message": message,
            "type": error_type
        }
    });
    Response::builder()
        .status(status)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

fn elevenlabs_error(message: &str, status: &str) -> Response {
    let body = serde_json::json!({
        "detail": {
            "status": status,
            "message": message
        }
    });
    Response::builder()
        .status(StatusCode::BAD_REQUEST)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from(serde_json::to_string(&body).unwrap()))
        .unwrap()
}

fn encode_wav(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>, hound::Error> {
    let mut cursor = Cursor::new(Vec::new());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        writer.write_sample(sample_i16)?;
    }
    writer.finalize()?;

    Ok(cursor.into_inner())
}

fn samples_to_pcm_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&sample_i16.to_le_bytes());
    }
    bytes
}

fn uuid_simple() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    format!("{:032x}", nanos)
}
