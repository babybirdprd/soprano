//! Server-Sent Events (SSE) streaming for real-time audio

#![allow(dead_code)] // API compatibility fields may not all be used

use crate::AppState;
use axum::{
    extract::{Path, State},
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Response,
    },
    Json,
};
use base64::{engine::general_purpose::STANDARD as BASE64, Engine};
use futures::stream;
use serde::Deserialize;
use soprano_core::{InferConfig, StreamConfig};
use std::convert::Infallible;
use std::sync::Arc;

/// SSE streaming request (OpenAI-style with stream flag)
#[derive(Deserialize)]
pub struct StreamSpeechRequest {
    pub model: Option<String>,
    pub input: String,
    #[serde(default)]
    pub voice: Option<String>,
    #[serde(default)]
    pub chunk_size: Option<usize>,
}

/// SSE streaming endpoint for OpenAI-compatible API
/// POST /v1/audio/speech/stream
pub async fn stream_speech(
    State(state): State<Arc<AppState>>,
    Json(request): Json<StreamSpeechRequest>,
) -> Response {
    if request.input.is_empty() {
        return sse_error("input cannot be empty".to_string());
    }

    let chunk_size = request.chunk_size.unwrap_or(5);
    let sample_rate = state.sample_rate;
    let input = request.input.clone();

    // Create SSE stream
    let stream = stream::unfold(
        StreamState::new(state, input, chunk_size),
        move |mut state| async move {
            match state.next_chunk().await {
                ChunkResult::Audio(audio) => {
                    let pcm_bytes = samples_to_pcm_bytes(&audio);
                    let base64_audio = BASE64.encode(&pcm_bytes);

                    let event_data = serde_json::json!({
                        "chunk": base64_audio,
                        "sample_rate": sample_rate,
                        "format": "pcm_s16le",
                        "samples": audio.len()
                    });

                    let event = Event::default().event("audio").data(event_data.to_string());

                    Some((Ok::<_, Infallible>(event), state))
                }
                ChunkResult::Done => {
                    let event = Event::default()
                        .event("done")
                        .data(r#"{"status":"complete"}"#);
                    state.finished = true;
                    Some((Ok(event), state))
                }
                ChunkResult::Error(e) => {
                    let event = Event::default()
                        .event("error")
                        .data(format!(r#"{{"error":"{}"}}"#, e));
                    state.finished = true;
                    Some((Ok(event), state))
                }
                ChunkResult::End => None,
            }
        },
    );

    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

/// ElevenLabs-style streaming endpoint
/// POST /v1/text-to-speech/{voice_id}/stream
pub async fn stream_speech_elevenlabs(
    Path(_voice_id): Path<String>,
    State(state): State<Arc<AppState>>,
    Json(request): Json<ElevenLabsStreamRequest>,
) -> Response {
    let stream_request = StreamSpeechRequest {
        model: request.model_id,
        input: request.text,
        voice: None,
        chunk_size: Some(5),
    };

    stream_speech(State(state), Json(stream_request)).await
}

#[derive(Deserialize)]
pub struct ElevenLabsStreamRequest {
    pub text: String,
    #[serde(default)]
    pub model_id: Option<String>,
}

// Internal streaming state machine

struct StreamState {
    state: Arc<AppState>,
    input: String,
    chunk_size: usize,
    started: bool,
    finished: bool,
    iterator: Option<StreamIteratorWrapper>,
}

impl StreamState {
    fn new(state: Arc<AppState>, input: String, chunk_size: usize) -> Self {
        Self {
            state,
            input,
            chunk_size,
            started: false,
            finished: false,
            iterator: None,
        }
    }

    async fn next_chunk(&mut self) -> ChunkResult {
        if self.finished {
            return ChunkResult::End;
        }

        // Initialize streaming on first call
        if !self.started {
            self.started = true;

            let mut tts = self.state.tts.lock().await;
            let infer_config = InferConfig::default();
            let stream_config = StreamConfig {
                chunk_size: self.chunk_size,
                ..Default::default()
            };

            // We need to collect chunks since we can't store the iterator
            // due to lifetime constraints with the mutex
            let mut chunks = Vec::new();
            for chunk_result in tts.infer_stream(&self.input, &infer_config, &stream_config) {
                match chunk_result {
                    Ok(chunk) => chunks.push(chunk),
                    Err(e) => return ChunkResult::Error(e.to_string()),
                }
            }

            self.iterator = Some(StreamIteratorWrapper { chunks, index: 0 });
        }

        // Get next chunk from iterator
        if let Some(ref mut iter) = self.iterator {
            if let Some(chunk) = iter.next() {
                return ChunkResult::Audio(chunk);
            }
        }

        ChunkResult::Done
    }
}

struct StreamIteratorWrapper {
    chunks: Vec<Vec<f32>>,
    index: usize,
}

impl StreamIteratorWrapper {
    fn next(&mut self) -> Option<Vec<f32>> {
        if self.index < self.chunks.len() {
            let chunk = self.chunks[self.index].clone();
            self.index += 1;
            Some(chunk)
        } else {
            None
        }
    }
}

enum ChunkResult {
    Audio(Vec<f32>),
    Done,
    Error(String),
    End,
}

fn samples_to_pcm_bytes(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let sample_i16 = (sample.clamp(-1.0, 1.0) * 32767.0) as i16;
        bytes.extend_from_slice(&sample_i16.to_le_bytes());
    }
    bytes
}

fn sse_error(message: String) -> Response {
    let stream = stream::once(async move {
        let event = Event::default()
            .event("error")
            .data(format!(r#"{{"error":"{}"}}"#, message));
        Ok::<_, Infallible>(event)
    });

    Sse::new(stream).into_response()
}
