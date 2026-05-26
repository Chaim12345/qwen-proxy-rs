//! Async SSE streaming using reqwest (async) instead of ureq (blocking).
//!
//! The core architectural fix: replace `ureq` inside `smol::unblock` (which holds
//! a thread-pool thread for the entire stream) with `reqwest` + `bytes_stream`,
//! which is truly async and yields control back to the smol runtime on every chunk.

use bytes::Bytes;
use futures::{Stream, StreamExt};
use std::pin::Pin;
use std::task::{Context, Poll};
use tracing::{debug, error, trace};

/// SSE line produced from the upstream Qwen stream.
pub struct SseLine {
    pub raw: String,
    pub request_id: String,
}

/// An async SSE stream that properly yields between chunks.
pub struct QwenSseStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buf: String,
    request_id: String,
    finished: bool,
}

impl QwenSseStream {
    pub fn new(resp: reqwest::Response) -> Self {
        let request_id = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();
        let status = resp.status();
        let inner = resp.bytes_stream();
        debug!(status = %status, request_id = %request_id, "Qwen SSE stream opened");
        Self {
            inner: Box::pin(inner),
            buf: String::new(),
            request_id,
            finished: false,
        }
    }
}

impl Stream for QwenSseStream {
    type Item = Result<SseLine, String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        if self.finished {
            return Poll::Ready(None);
        }
        loop {
            match self.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => {
                    self.finished = true;
                    if !self.buf.is_empty() {
                        let line = std::mem::take(&mut self.buf);
                        return Poll::Ready(Some(Ok(SseLine {
                            raw: line,
                            request_id: self.request_id.clone(),
                        })));
                    }
                    return Poll::Ready(None);
                }
                Poll::Ready(Some(Err(e))) => {
                    error!(error = %e, "Qwen SSE bytes_stream error");
                    return Poll::Ready(Some(Err(format!("Upstream stream error: {}", e))));
                }
                Poll::Ready(Some(Ok(bytes))) => {
                    let text = String::from_utf8_lossy(&bytes);
                    trace!(len = bytes.len(), "SSE bytes chunk received");
                    self.buf.push_str(&text);
                    let mut lines_to_yield = Vec::new();
                    while let Some(pos) = self.buf.find('\n') {
                        let line = self.buf.drain(..=pos).collect::<String>();
                        let line = line.trim_end_matches('\n').trim_end_matches('\r');
                        lines_to_yield.push(line.to_string());
                    }
                    if !lines_to_yield.is_empty() {
                        let first = lines_to_yield.remove(0);
                        for line in lines_to_yield.into_iter().rev() {
                            self.buf.insert_str(0, &line);
                            self.buf.insert(0, '\n');
                        }
                        return Poll::Ready(Some(Ok(SseLine {
                            raw: first,
                            request_id: self.request_id.clone(),
                        })));
                    }
                }
            }
        }
    }
}

/// Fire a POST and return an async SSE stream.
/// This replaces the blocking `ureq::post` + `smol::unblock` combo.
pub async fn post_sse(
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
) -> Result<QwenSseStream, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .map_err(|e| format!("Failed to build reqwest client: {}", e))?;
    let mut req = client.post(&url).body(body);
    for (k, v) in headers {
        req = req.header(k, v);
    }
    let resp = req.send().await.map_err(|e| format!("HTTP send failed: {}", e))?;
    let status = resp.status();
    if !(200..300).contains(&status.as_u16()) {
        let body_text = resp.text().await.unwrap_or_default();
        let preview: String = body_text.chars().take(500).collect();
        error!(status = %status, body_preview = %preview, "Qwen returned error status");
        if status.as_u16() == 429 || body_text.contains("in progress") {
            return Err("Qwen chat is busy (another message in flight)".to_string());
        }
        return Err(format!("Qwen API returned {}", status));
    }
    Ok(QwenSseStream::new(resp))
}
