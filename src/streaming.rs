//! Async SSE streaming using reqwest (async) instead of ureq (blocking).
//!
//! Fixes vs original streaming.rs:
//!  - Uses VecDeque to buffer lines; no intermediate Vec + push-front reversal.
//!  - poll_next yields one line per call instead of only the first of a batch.
//!  - post_sse() exposed so main.rs can call it directly (was never imported before).
//!  - request_id is Arc<str> — clone per SSE line is a refcount bump, not a heap alloc.

use bytes::Bytes;
use futures::{Stream, StreamExt};
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tracing::{debug, error, trace};

pub struct SseLine {
    pub raw: String,
    pub request_id: Arc<str>,
}

pub struct QwenSseStream {
    inner: Pin<Box<dyn Stream<Item = Result<Bytes, reqwest::Error>> + Send>>,
    buf: String,
    ready: VecDeque<String>,
    request_id: Arc<str>,
    finished: bool,
}

impl QwenSseStream {
    pub fn new(resp: reqwest::Response) -> Self {
        let request_id: Arc<str> = resp
            .headers()
            .get("x-request-id")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .into();
        debug!(status = %resp.status(), request_id = %request_id, "Qwen SSE stream opened");
        Self {
            inner: Box::pin(resp.bytes_stream()),
            buf: String::new(),
            ready: VecDeque::new(),
            request_id,
            finished: false,
        }
    }

    fn drain_lines(&mut self) {
        while let Some(pos) = self.buf.find('\n') {
            let line: String = self.buf.drain(..=pos).collect();
            let line = line.trim_end_matches('\n').trim_end_matches('\r').to_string();
            self.ready.push_back(line);
        }
    }
}

impl Stream for QwenSseStream {
    type Item = Result<SseLine, String>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            if let Some(line) = self.ready.pop_front() {
                return Poll::Ready(Some(Ok(SseLine {
                    raw: line,
                    request_id: Arc::clone(&self.request_id),
                })));
            }
            if self.finished {
                if !self.buf.is_empty() {
                    let line = std::mem::take(&mut self.buf);
                    return Poll::Ready(Some(Ok(SseLine {
                        raw: line,
                        request_id: Arc::clone(&self.request_id),
                    })));
                }
                return Poll::Ready(None);
            }
            match self.inner.as_mut().poll_next(cx) {
                Poll::Pending => return Poll::Pending,
                Poll::Ready(None) => { self.finished = true; }
                Poll::Ready(Some(Err(e))) => {
                    error!(error = %e, "Qwen SSE bytes_stream error");
                    return Poll::Ready(Some(Err(format!("Upstream stream error: {}", e))));
                }
                Poll::Ready(Some(Ok(bytes))) => {
                    trace!(len = bytes.len(), "SSE bytes chunk received");
                    self.buf.push_str(&String::from_utf8_lossy(&bytes));
                    self.drain_lines();
                }
            }
        }
    }
}

pub async fn post_sse(
    url: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
) -> Result<QwenSseStream, String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(180))
        .build()
        .map_err(|e| format!("Failed to build reqwest client: {}", e))?;
    let mut req = client.post(&url).body(body);
    for (k, v) in headers {
        req = req.header(k, v);
    }
    let resp = req
        .send()
        .await
        .map_err(|e| format!("HTTP send failed: {}", e))?;
    let status = resp.status();
    if !status.is_success() {
        let body_text = resp.text().await.unwrap_or_default();
        if status.as_u16() == 429 || body_text.contains("in progress") {
            return Err("Qwen chat is busy (another message in flight)".to_string());
        }
        return Err(format!(
            "Qwen API returned {}: {}",
            status,
            body_text.chars().take(200).collect::<String>()
        ));
    }
    Ok(QwenSseStream::new(resp))
}
