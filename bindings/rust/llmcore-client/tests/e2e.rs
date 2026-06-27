//! End-to-end tests against a spawned `llmcore-bridge` (FakeFacade).
//!
//! Set `LLMCORE_BRIDGE_PYTHON` to a python with `llmcore[bridge]` importable
//! (default `python3`). Unix-only harness (kills via the OS).

use std::net::TcpListener;
use std::process::{Child, Command};
use std::time::{Duration, Instant};

use llmcore_client::{v1, LlmcoreClient};

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// A spawned bridge process; killed on drop.
struct Bridge {
    child: Child,
    endpoint: String,
}

impl Bridge {
    async fn start() -> Bridge {
        Self::start_env(&[]).await
    }

    /// Start with the Tier-2 fake audio surface enabled (advertises tier2.audio).
    async fn start_audio() -> Bridge {
        Self::start_env(&[("LLMCORE_BRIDGE_FAKE_AUDIO", "1")]).await
    }

    async fn start_env(extra_env: &[(&str, &str)]) -> Bridge {
        let grpc_port = free_port();
        let http_port = free_port();
        let grpc_addr = format!("127.0.0.1:{}", grpc_port);
        let http_addr = format!("127.0.0.1:{}", http_port);
        let py = std::env::var("LLMCORE_BRIDGE_PYTHON").unwrap_or_else(|_| "python3".to_string());

        let mut cmd = Command::new(py);
        cmd.args([
            "-m",
            "llmcore.bridge.cli",
            "serve",
            "--transport",
            "grpc,http",
            "--grpc-address",
            &grpc_addr,
            "--http-address",
            &http_addr,
            "--insecure",
            "--log-level",
            "WARNING",
        ])
        .env("LLMCORE_BRIDGE_FAKE", "1");
        for (k, v) in extra_env {
            cmd.env(k, v);
        }
        let child = cmd.spawn().expect("spawn bridge");

        let endpoint = format!("http://{}", grpc_addr);
        let deadline = Instant::now() + Duration::from_secs(25);
        loop {
            assert!(Instant::now() < deadline, "bridge not ready");
            if let Ok(mut c) = LlmcoreClient::connect(endpoint.clone()).await {
                if let Ok(h) = c.health().await {
                    if h.ok {
                        break;
                    }
                }
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
        Bridge { child, endpoint }
    }
}

impl Drop for Bridge {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

async fn connected() -> (LlmcoreClient, Bridge) {
    let b = Bridge::start().await;
    let c = LlmcoreClient::connect(b.endpoint.clone()).await.unwrap();
    (c, b)
}

#[tokio::test]
async fn ensure_compatible() {
    let (mut c, _b) = connected().await;
    let info = c.ensure_compatible(&["tier0"]).await.unwrap();
    assert_eq!(info.contract_version, "llmcore.v1");

    let err = c.ensure_compatible(&["tier2.audio"]).await.unwrap_err();
    assert_eq!(err.code, "capability.missing");
}

#[tokio::test]
async fn chat_and_usage() {
    let (mut c, _b) = connected().await;
    let res = c
        .chat(v1::ChatRequest {
            message: "hello world".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(res.text, "echo: hello world");
    assert_eq!(res.usage.unwrap().prompt_tokens, Some(2));
}

#[tokio::test]
async fn chat_stream_concat() {
    let (mut c, _b) = connected().await;
    let mut s = c
        .chat_stream(v1::ChatRequest {
            message: "stream this please".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let mut parts = String::new();
    let mut done = false;
    while let Some(chunk) = s.message().await.unwrap() {
        if chunk.done {
            done = true;
        } else {
            parts.push_str(&chunk.text);
        }
    }
    assert!(done);
    assert_eq!(parts, "echo: stream this please");
}

#[tokio::test]
async fn count_and_cost() {
    let (mut c, _b) = connected().await;
    let ct = c
        .count_tokens(v1::CountTokensRequest {
            text: "one two three four".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(ct.tokens, 4);

    let ce = c
        .estimate_cost(v1::EstimateCostRequest {
            provider_name: "fake".into(),
            model_name: "fake-1".into(),
            prompt_tokens: 1_000_000,
            completion_tokens: 1_000_000,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(ce.currency, "USD");
    assert_eq!(ce.total_cost, 3.0);
}

#[tokio::test]
async fn catalog() {
    let (mut c, _b) = connected().await;
    assert_eq!(c.list_providers().await.unwrap().providers, vec!["fake"]);
    assert_eq!(
        c.list_models("fake").await.unwrap().models,
        vec!["fake-1", "fake-2"]
    );
    let d = c.get_provider_details(Some("fake".into())).await.unwrap();
    assert_eq!(d.id, "fake-1");
    assert_eq!(d.context_length, 8192);
}

#[tokio::test]
async fn embed_unsupported() {
    let (mut c, _b) = connected().await;
    let err = c
        .embed(v1::EmbedRequest {
            input: vec!["x".into()],
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.category, "ERROR_CATEGORY_UNSUPPORTED");
}

#[tokio::test]
async fn provider_rate_limit_decode() {
    let (mut c, _b) = connected().await;
    let err = c
        .chat(v1::ChatRequest {
            message: "__error__:provider_rate_limited".into(),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(err.category, "ERROR_CATEGORY_PROVIDER");
    assert_eq!(err.code, "provider.rate_limited");
    assert_eq!(err.http_status, Some(429));
    assert!(err.retryable);
    assert_eq!(err.retry_after_ms, Some(2000.0));
    assert_eq!(err.provider.as_deref(), Some("fake"));
}

#[tokio::test]
async fn stream_cancel() {
    let (mut c, _b) = connected().await;
    let mut s = c
        .chat_stream(v1::ChatRequest {
            message: "__cancel__".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    let first = s.message().await.unwrap();
    assert!(first.is_some());
    s.cancel();
    assert!(s.message().await.unwrap().is_none());
}

// ---- audio (Tier 2) ---------------------------------------------------- //
//
// Enum fields are compared by their verified numeric proto values (noted inline)
// to stay robust to prost's enum-variant naming. Oneof frames use the generated
// oneof modules (`audio_in::Frame`, `synth_control::Frame`,
// `voice_agent_client_event::Event`, `ocr_request::Source`) with single-word
// variant names derived from the oneof field names.

async fn connected_audio() -> (LlmcoreClient, Bridge) {
    let b = Bridge::start_audio().await;
    let c = LlmcoreClient::connect(b.endpoint.clone()).await.unwrap();
    (c, b)
}

#[tokio::test]
async fn audio_capabilities_advertised() {
    let (mut c, _b) = connected_audio().await;
    let info = c.get_info().await.unwrap();
    for cap in [
        "tier2.audio",
        "audio.transcribe_stream",
        "audio.synthesize_stream",
        "audio.voice_agent",
        "audio.synthesize",
        "audio.transcribe",
        "audio.generate_image",
        "audio.ocr",
        "audio.analyze_text",
    ] {
        assert!(
            info.capabilities.iter().any(|x| x == cap),
            "missing capability {cap}"
        );
    }
}

#[tokio::test]
async fn transcribe_stream() {
    let (mut c, _b) = connected_audio().await;
    let reqs = tokio_stream::iter(vec![
        v1::AudioIn {
            frame: Some(v1::audio_in::Frame::Audio(b"hello".to_vec())),
        },
        v1::AudioIn {
            frame: Some(v1::audio_in::Frame::Audio(b"world".to_vec())),
        },
        v1::AudioIn {
            frame: Some(v1::audio_in::Frame::Control(3)), // STT_CONTROL_CLOSE
        },
    ]);
    let mut s = c.transcribe_stream(reqs).await.unwrap();
    let mut types: Vec<i32> = Vec::new();
    let mut final_text = String::new();
    while let Some(ev) = s.message().await.unwrap() {
        types.push(ev.r#type);
        if ev.r#type == 2 {
            // FINAL
            final_text = ev.text.clone();
        }
    }
    // INTERIM, INTERIM, FINAL, UTTERANCE_END
    assert_eq!(types, vec![1, 1, 2, 3]);
    assert_eq!(final_text, "hello world");
}

#[tokio::test]
async fn synthesize_stream() {
    let (mut c, _b) = connected_audio().await;
    let pieces = ["foo", "bar", "baz"];
    let mut frames: Vec<v1::SynthControl> = pieces
        .iter()
        .map(|p| v1::SynthControl {
            frame: Some(v1::synth_control::Frame::Text((*p).to_string())),
        })
        .collect();
    frames.push(v1::SynthControl {
        frame: Some(v1::synth_control::Frame::Control(3)), // TTS_CONTROL_CLOSE
    });
    let mut s = c.synthesize_stream(tokio_stream::iter(frames)).await.unwrap();
    let mut chunks: Vec<String> = Vec::new();
    let mut seqs: Vec<i64> = Vec::new();
    while let Some(out) = s.message().await.unwrap() {
        chunks.push(String::from_utf8(out.audio).unwrap());
        seqs.push(out.seq);
    }
    assert_eq!(chunks, vec!["foo", "bar", "baz"]);
    assert_eq!(seqs, vec![0, 1, 2]);
}

#[tokio::test]
async fn voice_agent_duplex() {
    let (mut c, _b) = connected_audio().await;
    // A non-settings leading frame -> the default provider (the fake).
    let reqs = tokio_stream::iter(vec![
        v1::VoiceAgentClientEvent {
            event: Some(v1::voice_agent_client_event::Event::InjectUserMessage(
                "hi there".into(),
            )),
        },
        v1::VoiceAgentClientEvent {
            event: Some(v1::voice_agent_client_event::Event::Audio(vec![1, 2])),
        },
    ]);
    let mut s = c.voice_agent(reqs).await.unwrap();
    let mut events = Vec::new();
    while let Some(ev) = s.message().await.unwrap() {
        events.push(ev);
    }
    assert!(!events.is_empty());
    assert_eq!(events.first().unwrap().r#type, 1); // WELCOME
    assert_eq!(events.last().unwrap().r#type, 17); // CLOSE
    let conv = events
        .iter()
        .find(|e| e.r#type == 3) // CONVERSATION_TEXT
        .expect("conversation-text event");
    assert_eq!(conv.role.as_deref(), Some("user"));
    assert_eq!(conv.content.as_deref(), Some("hi there"));
    let audio = events
        .iter()
        .find(|e| e.r#type == 8) // AUDIO
        .expect("audio event");
    assert_eq!(audio.audio.as_deref(), Some(&b"agent:\x01\x02"[..]));
}

#[tokio::test]
async fn synthesize_unary() {
    let (mut c, _b) = connected_audio().await;
    let r = c
        .synthesize(v1::SynthesizeRequest {
            text: "hello".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(r.audio_data, b"tts:hello");
    assert_eq!(r.model, "fake-tts");
}

#[tokio::test]
async fn transcribe_unary() {
    let (mut c, _b) = connected_audio().await;
    let r = c
        .transcribe(v1::TranscribeRequest {
            audio_data: b"hello world".to_vec(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(r.text, "hello world");
    assert_eq!(r.language.as_deref(), Some("en"));
    assert_eq!(r.segments.len(), 1);
    assert_eq!(r.segments[0].speaker.as_deref(), Some("spk_0"));
}

#[tokio::test]
async fn generate_image_unary() {
    let (mut c, _b) = connected_audio().await;
    let r = c
        .generate_image(v1::GenerateImageRequest {
            prompt: "a cat".into(),
            n: 2,
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(r.images.len(), 2);
    // base64("img:a cat") == "aW1nOmEgY2F0"
    assert_eq!(r.images[0].data.as_deref(), Some("aW1nOmEgY2F0"));
}

#[tokio::test]
async fn ocr_unary() {
    let (mut c, _b) = connected_audio().await;
    let r = c
        .ocr(v1::OcrRequest {
            source: Some(v1::ocr_request::Source::Data(b"PDFBYTES".to_vec())),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(r.model, "fake-ocr");
    assert_eq!(r.pages_processed, 1);
    assert_eq!(r.doc_size_bytes, Some(8));
    assert_eq!(r.pages.len(), 1);
}

#[tokio::test]
async fn analyze_text_unary() {
    let (mut c, _b) = connected_audio().await;
    // No features -> the fake returns model "fake-analyze", no summary/topics.
    let r = c
        .analyze_text(v1::AnalyzeTextRequest {
            text: "some text".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_eq!(r.model.as_deref(), Some("fake-analyze"));
    assert!(r.summary.is_none());
    assert!(r.topics.is_empty());
}
