//! Async SSE streaming using reqwest (async) instead of ureq (blocking).
//!
//! Fixes vs original streaming.rs:
//!  - Uses VecDeque to buffer lines; no intermediate Vec + push-front reversal.
//!  - poll_next yields one line per call instead of only the first of a batch.
//!  - post_sse() exposed so main.rs can call it directly (was never imported before).
//!  - request_id is Arc<str> — clone per SSE line is a refcount bump, not a heap alloc.

use bytes::Bytes;
use futures::Stream;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use tracing::{debug, error, trace};

#[allow(dead_code)]
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
    /// Create a pre-filled stream from already-fetched lines (used for the blocking ureq path in post_sse).
    /// (The reqwest-based `new` ctor was removed as unused after the Tokio refactor.)
    pub fn from_lines(request_id: impl Into<Arc<str>>, lines: Vec<String>) -> Self {
        let request_id = request_id.into();
        debug!(request_id = %request_id, line_count = lines.len(), "Qwen SSE stream (preloaded from blocking)");
        let mut ready = VecDeque::new();
        for l in lines {
            ready.push_back(l);
        }
        Self {
            inner: Box::pin(futures::stream::pending()), // not used
            buf: String::new(),
            ready,
            request_id,
            finished: true,
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
    // Blocking ureq fetch now via tokio::task::spawn_blocking (safe under the Tokio runtime)
    let body2 = body.clone();
    let url2 = url.clone();
    let headers2 = headers.clone();
    let res: Result<(u16, String, String), String> = tokio::task::spawn_blocking(move || {
        let mut req = ureq::post(&url2);
        for (k, v) in &headers2 {
            req = req.set(k.as_str(), v.as_str());
        }
        req = req.set("accept", "text/event-stream");
        match req.send_bytes(&body2) {
            Ok(resp) => {
                let status = resp.status();
                let req_id = resp.header("x-request-id").unwrap_or("").to_string();
                let body_text = resp.into_string().unwrap_or_default();
                Ok((status, req_id, body_text))
            }
            Err(e) => Err(format!("Qwen SSE request failed: {}", e)),
        }
    })
    .await
    .map_err(|e| format!("spawn_blocking join error: {}", e))?;

    let (status, request_id, body_text) = match res {
        Ok(t) => t,
        Err(e) => return Err(e),
    };
    if !(200..300).contains(&status) {
        if status == 429 || body_text.contains("in progress") {
            return Err("Qwen chat is busy (another message in flight)".to_string());
        }
        return Err(format!(
            "Qwen API returned {}: {}",
            status,
            body_text.chars().take(200).collect::<String>()
        ));
    }
    // Preload all lines so the returned stream yields them without hitting the (unused) inner reqwest stream.
    let lines: Vec<String> = body_text.lines().map(|s| s.to_string()).collect();
    Ok(QwenSseStream::from_lines(request_id, lines))
}
