//! HTTP route handlers for Soprano Server

use axum::{extract::Path, http::StatusCode, response::IntoResponse, Json};
use serde::{Deserialize, Serialize};

/// Health check endpoint
pub async fn health() -> impl IntoResponse {
    Json(HealthResponse {
        status: "ok".to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

#[derive(Serialize)]
pub struct HealthResponse {
    status: String,
    version: String,
}

/// OpenAI-compatible speech request
#[derive(Deserialize)]
pub struct OpenAISpeechRequest {
    pub model: String,
    pub input: String,
    pub voice: Option<String>,
    pub response_format: Option<String>,
    pub speed: Option<f32>,
}

/// OpenAI-compatible TTS endpoint
/// POST /v1/audio/speech
pub async fn openai_speech(Json(_request): Json<OpenAISpeechRequest>) -> impl IntoResponse {
    // TODO: Implement actual TTS
    // For now, return a placeholder response
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "error": {
                "message": "TTS generation not yet implemented",
                "type": "not_implemented"
            }
        })),
    )
}

/// ElevenLabs-compatible speech request
#[derive(Deserialize)]
pub struct ElevenLabsSpeechRequest {
    pub text: String,
    pub model_id: Option<String>,
    pub voice_settings: Option<VoiceSettings>,
}

#[derive(Deserialize)]
pub struct VoiceSettings {
    pub stability: Option<f32>,
    pub similarity_boost: Option<f32>,
}

/// ElevenLabs-compatible TTS endpoint
/// POST /v1/text-to-speech/{voice_id}
pub async fn elevenlabs_speech(
    Path(_voice_id): Path<String>,
    Json(_request): Json<ElevenLabsSpeechRequest>,
) -> impl IntoResponse {
    // TODO: Implement actual TTS
    (
        StatusCode::NOT_IMPLEMENTED,
        Json(serde_json::json!({
            "detail": {
                "status": "not_implemented",
                "message": "TTS generation not yet implemented"
            }
        })),
    )
}
